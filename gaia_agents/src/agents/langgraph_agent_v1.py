from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Set up tools
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# Initialize all tools
tools = [
    DuckDuckGoSearchRun(),
    PubmedQueryRun(),
    SemanticScholarQueryRun(),
    ArxivQueryRun(),
    WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    repl_tool,
]

# Initialize Gemini model
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
model_with_tools = model.bind_tools(tools)

# Create tools lookup
tools_by_name = {tool.name: tool for tool in tools}


# Pydantic models for structured output
class ToolSufficiencyResponse(BaseModel):
    """Response for tool sufficiency check"""

    sufficient: bool = Field(description="Whether the available tools are sufficient to answer the question")
    reasoning: str = Field(description="Brief reasoning for the decision")


class FinalAnswer(BaseModel):
    """Final answer structure"""

    answer: str = Field(description="The comprehensive answer to the user's question")


# Graph state
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    llm_call_count: int
    max_llm_calls: int
    question: str | None = None
    answer: FinalAnswer | None = None
    tool_sufficiency: ToolSufficiencyResponse | None = None


# Node functions
def check_tool_sufficiency(state: AgentState, config: RunnableConfig):
    """Check if available tools are sufficient to answer the question"""

    question = state["question"]
    question_message = HumanMessage(content=f"Question to analyze: {question}")

    # Create system prompt for sufficiency check
    available_tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

    system_prompt = f"""You are an AI assistant that needs to determine if the available tools are sufficient to answer a user's question.

Available tools:
{available_tools_desc}

Your task is to analyze the user's question and determine if these tools provide sufficient capability to answer it comprehensively.

Consider:
- Can the question be answered with web search, academic papers, or computational tools?
- Does the question require real-time data, personal information, or capabilities not available through these tools?
- Can you break down the question into parts that these tools can handle?

Be generous in your assessment - if there's a reasonable path to answer the question using these tools, respond with sufficient=True."""

    messages = [SystemMessage(content=system_prompt), question_message]

    structured_model = model.with_structured_output(ToolSufficiencyResponse)
    response = structured_model.invoke(messages, config)

    # Add response to messages for context
    response_message = AIMessage(
        content=f"Tool sufficiency check: {'Sufficient' if response.sufficient else 'Insufficient'}. Reasoning: {response.reasoning}"
    )

    return {"messages": [question_message, response_message], "tool_sufficiency": response}


def call_model(state: AgentState, config: RunnableConfig):
    """Call the model (ReAct agent LLM node)"""

    system_prompt = SystemMessage(
        content="""You are a helpful AI assistant with access to various tools. Use the tools available to you to answer the user's question comprehensively.

    Think step by step:
    1. Analyze what information you need.
    2. Use appropriate tools to gather that information.
    3. Synthesize the information to provide a complete answer.

    IMPORTANT INSTRUCTIONS:
    - Avoid repeating the same tool call with identical parameters.
    - When calling tools, vary your queries and tool arguments to explore different aspects of the question.
    - Use each tool intelligently and purposefully—avoid redundant or uninformative tool calls.
    - Track which tools you've already used and how, so you don't repeat yourself.

    Be thorough but efficient with your tool usage. Use tools only when needed, and prefer combining information from multiple sources."""
    )

    response = model_with_tools.invoke([system_prompt] + state["messages"], config)

    # Increment LLM call count
    new_count = state.get("llm_call_count", 0) + 1

    return {"messages": [response], "llm_call_count": new_count}


def tool_node(state: AgentState):
    """Execute tools based on the last message's tool calls"""

    outputs = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        try:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            outputs.append(
                ToolMessage(
                    content=f"Error executing tool {tool_call['name']}: {str(e)}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

    return {"messages": outputs}


def final_answer_node(state: AgentState, config: RunnableConfig):
    """Generate final structured answer based on conversation history"""

    system_prompt = SystemMessage(
        content="""You are tasked with providing the most concise possible final answer to the user question based on the conversation history and tool usage.

CRITICAL INSTRUCTIONS FOR CONCISENESS:
- If the question asks for a number, provide ONLY the number (e.g., "5", "23", "147")
- If the question asks for a name, provide ONLY the name (e.g., "John Smith", "Paris")
- If the question asks for a yes/no, provide ONLY "Yes" or "No"
- If the question asks for a date, provide ONLY the date (e.g., "2023-05-15", "March 2020")
- Remove ALL unnecessary words, articles, explanations, or context
- Do NOT include phrases like "The answer is", "Based on the research", "According to", etc.
- Provide the absolute minimum text that directly answers the question
- This is for a benchmark submission where brevity is crucial

If the answer cannot be determined from the conversation history, respond with "Unable to determine" (nothing more)."""
    )

    filtered_messages = []
    for msg in state["messages"]:
        if hasattr(msg, "content") and msg.content and msg.content.strip():
            filtered_messages.append(msg)

    messages = [system_prompt] + filtered_messages

    structured_model = model.with_structured_output(FinalAnswer)
    response = structured_model.invoke(messages, config)

    return {"messages": [AIMessage(content=f"Final Answer: {response.answer}")], "answer": response}


# Edge functions
def should_continue_sufficiency(state: AgentState) -> Literal["sufficient", "insufficient"]:
    """Decide whether tools are sufficient"""
    # Check if we have a tool sufficiency result
    sufficiency = state["tool_sufficiency"]

    if sufficiency and hasattr(sufficiency, "sufficient") and sufficiency.sufficient:
        return "sufficient"
    else:
        return "insufficient"


def should_continue_react(state: AgentState) -> Literal["tools", "final_answer"]:
    """Decide whether to continue with ReAct loop or move to final answer"""

    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "final_answer"


def should_continue_after_tools(state: AgentState) -> Literal["agent", "final_answer"]:
    """Decide after tool execution whether to continue or finalize"""

    llm_call_count = state.get("llm_call_count", 0)
    max_calls = state.get("max_llm_calls", 4)

    # If we've reached the maximum number of LLM calls, go to final answer
    if llm_call_count >= max_calls:
        return "final_answer"

    # Otherwise, continue the ReAct loop (go back to agent)
    return "agent"


def create_react_agent_graph():
    """Create and return the compiled ReAct agent graph"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("check_sufficiency", check_tool_sufficiency)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("final_answer", final_answer_node)

    # Set entry point
    workflow.set_entry_point("check_sufficiency")

    # Add conditional edges
    workflow.add_conditional_edges(
        "check_sufficiency", should_continue_sufficiency, {"sufficient": "agent", "insufficient": END}
    )
    workflow.add_conditional_edges("agent", should_continue_react, {"tools": "tools", "final_answer": "final_answer"})
    workflow.add_conditional_edges(
        "tools", should_continue_after_tools, {"agent": "agent", "final_answer": "final_answer"}
    )

    # Add edges
    workflow.add_edge("final_answer", END)

    return workflow.compile()


def run_agent(question: str, max_llm_calls: int = 4):
    """Run the ReAct agent with a question"""

    graph = create_react_agent_graph()

    initial_state = {"messages": [HumanMessage(content=question)], "llm_call_count": 0, "max_llm_calls": max_llm_calls}

    # Stream the execution
    print(f"Question: {question}")
    print("=" * 50)

    for step in graph.stream(initial_state):
        for node, output in step.items():
            print(f"\n--- {node.upper()} ---")
            if "messages" in output and output["messages"]:
                for msg in output["messages"]:
                    if hasattr(msg, "content"):
                        print(f"{msg.__class__.__name__}: {msg.content}")
                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"Tool calls: {[tc['name'] for tc in msg.tool_calls]}")

            if "final_answer" in output:
                print("\nFINAL STRUCTURED ANSWER:")
                print(f"Answer: {output['final_answer'].answer}")
                print(f"Confidence: {output['final_answer'].confidence}")
                print(f"Sources: {output['final_answer'].sources_used}")
