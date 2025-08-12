from collections.abc import Sequence
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agents.models import FeasibilityCheck, FinalAnswer, FinalConclusion, NextStep
from src.agents.prompts import GAIAPrompts
from src.agents.tools import tools

# Initialize
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
model_with_tools = model.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}

prompts = GAIAPrompts()


# Graph state
class GraphState(BaseModel):
    """The state of the graph"""

    # History
    history: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list
    )  # Complete history with node info
    coordinator_messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list
    )  # Coordinator-specific messages
    executor_messages: Sequence[BaseMessage] = Field(default_factory=list)  # Executor-specific messages

    # Input
    question: str

    # Feasibility check
    feasibility: FeasibilityCheck | None = None

    # Coordinator state
    next_step: NextStep | None = None
    coordinator_conclusion: FinalConclusion | None = None
    coordinator_iterations: int
    coordinator_max_iterations: int

    # Executor state
    executor_conclusion: FinalConclusion | None = None
    executor_iterations: int
    executor_max_iterations: int

    # Final answer state
    final_answer: FinalAnswer | None = None

    def __getitem__(self, item):
        return getattr(self, item)


# Nodes
def check_feasibility(state: GraphState, config: RunnableConfig):
    """Check if the question is feasible to answer with the available tools"""

    question = state["question"]

    system_message = SystemMessage(content=prompts.get_feasibility_check_prompt(tools), node="feasibility")
    question_message = HumanMessage(content=question, node="feasibility")
    messages = [system_message, question_message]

    structured_model = model.with_structured_output(FeasibilityCheck)
    response = structured_model.invoke(messages, config)

    response_message = AIMessage(content=str(response), node="feasibility")
    messages += [response_message]
    return {
        "history": messages,
        "feasibility": response,
    }


def coordinator_node(state: GraphState, config: RunnableConfig):
    """Determine the next step in the plan and select appropriate tools"""

    coordinator_messages = state["coordinator_messages"]
    new_messages = []

    if not coordinator_messages:
        system_message = SystemMessage(content=prompts.get_coordinator_system_prompt(tools), node="coordinator")
        human_message = HumanMessage(
            content=prompts.get_coordinator_context_prompt(state["question"]), node="coordinator"
        )
        coordinator_messages = [system_message, human_message]
        new_messages = coordinator_messages

    if state["executor_conclusion"]:
        executor_message = AIMessage(
            content=f"Executor conclusion: {state['executor_conclusion'].conclusion}. Complete text: {str(state['executor_conclusion'])}",
            node="executor",
        )
        coordinator_messages += [executor_message]
        new_messages += [executor_message]

    # Check if we've reached max iterations
    if (state["next_step"] and state["next_step"].is_final) or (
        state["coordinator_iterations"] >= state["coordinator_max_iterations"]
    ):
        # Generate final conclusion instead of next step
        human_message = HumanMessage(
            content=prompts.get_coordinator_max_iterations_prompt(state["question"]), node="coordinator"
        )

        structured_model = model.with_structured_output(FinalConclusion)
        response = structured_model.invoke(coordinator_messages + [human_message], config)
        response_message = AIMessage(content=str(response), node="coordinator")

        new_messages += [human_message, response_message]
        return {
            "history": new_messages,
            "coordinator_messages": new_messages,
            "coordinator_conclusion": response,
            "coordinator_iterations": state["coordinator_iterations"] + 1,
        }

    structured_model = model.with_structured_output(NextStep)
    response = structured_model.invoke(coordinator_messages, config)

    response_message = AIMessage(content=str(response), node="coordinator")
    new_messages += [response_message]

    return {
        "history": new_messages,
        "coordinator_messages": new_messages,
        "coordinator_iterations": state["coordinator_iterations"] + 1,
        "next_step": response,
        "executor_messages": [],
        "executor_conclusion": None,
        "executor_iterations": 0,
    }


def executor_node(state: GraphState, config: RunnableConfig):
    """Plan the execution of the current step using ReAct pattern"""
    if not state["next_step"]:
        return {
            "executor_conclusion": FinalConclusion(conclusion="No next step", partial_results=""),
            "executor_iterations": state["executor_iterations"] + 1,
        }

    messages = state["executor_messages"]

    if not messages:
        system_message = SystemMessage(
            content=prompts.get_executor_system_prompt(state["next_step"].tools),
            node="executor",
        )
        human_message = HumanMessage(content=prompts.get_executor_task_prompt(state["next_step"].step), node="executor")
        messages = [system_message, human_message]

    if state["executor_iterations"] >= state["executor_max_iterations"]:
        # Generate final conclusion and return to coordinator
        human_message = HumanMessage(
            content=prompts.get_executor_max_iterations_prompt(state["next_step"].step),
            node="executor",
        )

        messages += [human_message]

        structured_model = model.with_structured_output(FinalConclusion)
        response = structured_model.invoke(messages, config)

        response_message = AIMessage(
            content=f"Executor conclusion: {str(response)}",
            node="executor",
        )

        return {
            "history": [human_message, response_message],
            "executor_conclusion": response
            or FinalConclusion(conclusion="Failed to generate conclusion", partial_results=""),
            "executor_iterations": state["executor_iterations"] + 1,
        }

    selected_tools = [tool for tool in tools if tool.name in state["next_step"].tools]
    model_with_selected_tools = model.bind_tools(selected_tools)

    response_message = model_with_selected_tools.invoke(messages, config)
    response_message.node = "executor"

    return {
        "history": response_message,
        "executor_messages": messages + [response_message],
        "executor_iterations": state["executor_iterations"] + 1,
    }


def tool_node(state: GraphState):
    """Execute tools based on the last message's tool calls"""
    outputs = []
    messages = state["executor_messages"]
    last_message = state["executor_messages"][-1]

    for tool_call in last_message.tool_calls:
        try:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            tool_message = ToolMessage(
                content=str(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                node="tools",
            )
            outputs.append(tool_message)
        except Exception as e:
            tool_message = ToolMessage(
                content=f"Error executing tool {tool_call['name']}: {str(e)}",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                node="tools",
            )
            outputs.append(tool_message)

    return {
        "history": outputs,
        "executor_messages": messages + outputs,
    }


def finalise(state: GraphState, config: RunnableConfig):
    """Generate the final answer based on coordinator history"""
    system_message = SystemMessage(content=prompts.get_finalizer_prompt(), node="finalise")
    messages = [system_message] + state["coordinator_messages"]

    structured_model = model.with_structured_output(FinalAnswer)
    response = structured_model.invoke(messages, config)
    response_message = AIMessage(content=str(response), node="finalise")

    return {"history": response_message, "final_answer": response}


# Edges
def should_continue_after_feasibility(state: GraphState) -> Literal["coordinator", END]:
    """Decide whether to continue with coordination or end"""
    if state["feasibility"] and state["feasibility"].feasible:
        return "coordinator"
    return END


def should_continue_after_coordinator(state: GraphState) -> Literal["executor", "finalise"]:
    """Decide whether to continue with execution or go to final answer"""
    if state["coordinator_conclusion"] or (state["coordinator_iterations"] >= state["coordinator_max_iterations"]):
        return "finalise"
    return "executor"


def should_continue_after_executor(state: GraphState) -> Literal["tools", "coordinator", "executor"]:
    """Decide whether to continue with tools or go back to coordinator"""
    last_message = state["executor_messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    if state["executor_conclusion"]:
        return "coordinator"

    return "executor"


def should_continue_after_tools(state: GraphState) -> Literal["executor"]:
    """Tools always go back to executor"""
    return "executor"


# Graph
def build_graph():
    """Build the graph"""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("check_feasibility", check_feasibility)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("tools", tool_node)
    graph.add_node("finalise", finalise)

    # Set entry point
    graph.set_entry_point("check_feasibility")

    # Add edges
    graph.add_conditional_edges(
        "check_feasibility", should_continue_after_feasibility, {"coordinator": "coordinator", END: END}
    )
    graph.add_conditional_edges(
        "coordinator", should_continue_after_coordinator, {"executor": "executor", "finalise": "finalise"}
    )
    graph.add_conditional_edges(
        "executor",
        should_continue_after_executor,
        {"executor": "executor", "tools": "tools", "coordinator": "coordinator"},
    )
    graph.add_conditional_edges(
        "tools",
        should_continue_after_tools,
        {"executor": "executor"},
    )

    # Finalise node goes to END
    graph.add_edge("finalise", END)

    return graph.compile()


def run_agent(question: str, coordinator_max_iterations: int = 5, executor_max_iterations: int = 3):
    """Run the agent with a question"""
    graph = build_graph()

    initial_state = {
        "question": question,
        "history": [],
        "coordinator_messages": [],
        "executor_messages": [],
        "coordinator_iterations": 0,
        "executor_iterations": 0,
        "coordinator_max_iterations": coordinator_max_iterations,
        "executor_max_iterations": executor_max_iterations,
    }

    # Stream the execution
    print(f"Question: {question}")
    print("=" * 50)

    for step in graph.stream(initial_state):
        for node, output in step.items():
            print(f"\n--- {node.upper()} ---")

            # Print history with node information
            if "history" in output and output["history"]:
                print("\nComplete History (with node info):")
                for msg in output["history"]:
                    node_info = getattr(msg, "node", "unknown") if hasattr(msg, "node") else "unknown"
                    content = getattr(msg, "content", str(msg)) if hasattr(msg, "content") else str(msg)
                    print(f"[{node_info}] {msg.__class__.__name__}: {content}")

            if "coordinator_messages" in output and output["coordinator_messages"]:
                print("\nCoordinator Messages:")
                for msg in output["coordinator_messages"]:
                    if hasattr(msg, "content"):
                        print(f"{msg.__class__.__name__}: {msg.content}")

            if "executor_messages" in output and output["executor_messages"]:
                print("\nExecutor Messages:")
                for msg in output["executor_messages"]:
                    if hasattr(msg, "content"):
                        print(f"{msg.__class__.__name__}: {msg.content}")

            if "executor_conclusion" in output and output["executor_conclusion"]:
                print("\n=== EXECUTOR CONCLUSION ===")
                print(f"Conclusion: {output['executor_conclusion'].conclusion}")
                print(f"Partial Results: {output['executor_conclusion'].partial_results}")
                print(f"Confidence: {output['executor_conclusion'].confidence}")

            if "final_answer" in output and output["final_answer"]:
                print("\n=== FINAL ANSWER ===")
                print(f"Answer: {output['final_answer'].answer}")
                print(f"Confidence: {output['final_answer'].confidence}")
                print(f"Reasoning: {output['final_answer'].reasoning}")
