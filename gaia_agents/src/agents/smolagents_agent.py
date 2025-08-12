"""
Multi-Agent System for GAIA Benchmark using smolagents
Architecture: Coordinator -> Specialized Agents
"""

from typing import Any

from smolagents import CodeAgent, HfApiModel

from src.tools import all_tools


class GAIAMultiAgentSystem:
    """
    Multi-agent system designed for GAIA benchmark tasks.
    Uses a coordinator agent that delegates to specialized agents.
    """

    def __init__(self, model_config: dict[str, Any] | None = None):
        """
        Initialize the multi-agent system.

        Args:
            model_config: Configuration for the language model
                         e.g., {"model_id": "Qwen/Qwen2.5-Coder-32B-Instruct", "provider": "together"}
        """
        model_config = model_config or {}
        self.model = HfApiModel(**model_config)
        # self.model = InferenceClientModel(**model_config)
        self.agents = {}
        self._setup_agents()
        self._setup_coordinator()

    def _setup_agents(self):
        """Set up all specialized agents with their respective tools."""

        # Search Agent - Information retrieval
        search_tools = [
            # Assuming these are your actual tool instances
            # Replace with actual tool references from all_tools
            "wikipedia_search",
            "wikipedia_search_tool",
            "duckduckgo_search",
            "web_search_duckduckgo",
            "arxiv_search",
            "fetch_webpage_content",
        ]

        self.agents["search_agent"] = CodeAgent(
            model=self.model,
            tools=[tool for tool in all_tools if tool.name in search_tools],
            name="search_agent",
            description="Retrieves factual information and background data from various sources including Wikipedia, web search, and academic papers",
            verbosity_level=1,
            max_steps=10,
        )

        # Document Agent - Document processing
        document_tools = ["load_csv_file", "load_excel_file", "read_text_file", "transcribe_audio_file"]

        self.agents["document_agent"] = CodeAgent(
            model=self.model,
            tools=[tool for tool in all_tools if tool.name in document_tools],
            name="document_agent",
            description="Loads and processes structured and unstructured documents including CSV, Excel, text files, and audio transcriptions",
            verbosity_level=1,
            max_steps=8,
        )

        # Vision Agent - Image processing
        vision_tools = ["ocr_tool", "image_captioning_tool", "visual_qa_tool"]

        self.agents["vision_agent"] = CodeAgent(
            model=self.model,
            tools=[tool for tool in all_tools if tool.name in vision_tools],
            name="vision_agent",
            description="Extracts text and meaning from images using OCR, captioning, and visual question answering",
            verbosity_level=1,
            max_steps=6,
        )

        # Reasoning Agent - Logic and analysis
        reasoning_tools = ["analyze_chess_position", "analyze_table_commutativity", "count_items_in_list"]

        self.agents["reasoning_agent"] = CodeAgent(
            model=self.model,
            tools=[tool for tool in all_tools if tool.name in reasoning_tools],
            name="reasoning_agent",
            description="Performs symbolic reasoning, logical pattern recognition, and analytical tasks",
            verbosity_level=1,
            max_steps=8,
        )

        # Language Agent - Text processing
        language_tools = ["reverse_string", "reverse_words_in_string"]

        # Note: Language agent might need additional string manipulation tools
        self.agents["language_agent"] = CodeAgent(
            model=self.model,
            tools=[tool for tool in all_tools if tool.name in language_tools],
            name="language_agent",
            description="Handles low-level text transformations and string manipulations",
            verbosity_level=1,
            max_steps=5,
        )

        # Coding Agent - Python execution and logic
        self.agents["coding_agent"] = CodeAgent(
            model=self.model,
            tools=[],  # Uses implicit code execution capabilities
            name="coding_agent",
            description="Executes Python code and performs computational logic through code interpretation",
            additional_authorized_imports=[
                "pandas",
                "numpy",
                "matplotlib",
                "json",
                "re",
                "datetime",
                "math",
                "statistics",
                "itertools",
            ],
            verbosity_level=1,
            max_steps=10,
        )

    def _setup_coordinator(self):
        """Set up the coordinator agent that manages other agents."""

        # Collect all managed agents
        managed_agents = list(self.agents.values())

        self.coordinator = CodeAgent(
            model=self.model,
            tools=[],  # Coordinator has no direct tools
            managed_agents=managed_agents,
            name="coordinator",
            description="Coordinates and delegates tasks to specialized agents based on task requirements",
            planning_interval=3,  # Plan every 3 steps
            verbosity_level=2,
            max_steps=20,
        )

    def analyze_task(self, task: str) -> dict[str, Any]:
        """
        Analyze a GAIA task to determine which agents might be needed.

        Args:
            task: The task description

        Returns:
            Dictionary with task analysis
        """
        analysis_prompt = f"""
        Analyze this GAIA benchmark task and determine which types of agents would be most useful:
        
        Task: {task}
        
        Available agent types:
        - search_agent: For finding factual information online
        - document_agent: For processing files (CSV, Excel, text, audio)
        - vision_agent: For analyzing images
        - reasoning_agent: For logical analysis and pattern recognition
        - language_agent: For text transformations
        - coding_agent: For computational tasks and data processing
        
        Provide a brief analysis of what agents would be needed and why.
        """

        # Use the coordinator's model for analysis
        response = self.model([{"role": "user", "content": analysis_prompt}])
        return {"analysis": response.content, "task": task}

    def solve_task(self, task: str, context: str | None = None) -> Any:
        """
        Solve a GAIA benchmark task using the multi-agent system.

        Args:
            task: The task to solve
            context: Optional additional context

        Returns:
            The result from the coordinator agent
        """

        # Prepare the enhanced prompt for the coordinator
        enhanced_task = f"""
        You are coordinating a team of specialized agents to solve this GAIA benchmark task.
        
        TASK: {task}
        
        {f"CONTEXT: {context}" if context else ""}
        
        Available agents and their capabilities:
        - search_agent: Retrieves information from Wikipedia, web search, ArXiv
        - document_agent: Processes CSV, Excel, text files, and audio transcriptions  
        - vision_agent: Analyzes images with OCR, captioning, and visual QA
        - reasoning_agent: Performs logical analysis and pattern recognition
        - language_agent: Handles text transformations and string operations
        - coding_agent: Executes Python code for computational tasks
        
        Strategy:
        1. Analyze what type of information or processing is needed
        2. Delegate to appropriate specialized agents
        3. Combine results from multiple agents if needed
        4. Provide a final comprehensive answer
        
        Be systematic and thorough. Use multiple agents when the task requires different types of expertise.
        """

        return self.coordinator.run(enhanced_task)

    def get_agent_info(self) -> dict[str, dict]:
        """Get information about all agents in the system."""
        info = {}
        for name, agent in self.agents.items():
            info[name] = {
                "description": agent.description,
                "tools": [tool.name for tool in agent.tools] if hasattr(agent, "tools") else [],
                "max_steps": agent.max_steps,
            }

        info["coordinator"] = {
            "description": self.coordinator.description,
            "managed_agents": [agent.name for agent in self.coordinator.managed_agents],
            "max_steps": self.coordinator.max_steps,
        }

        return info

    def visualize_system(self):
        """Visualize the multi-agent system structure."""
        if hasattr(self.coordinator, "visualize"):
            return self.coordinator.visualize()
        else:
            print("System Structure:")
            print("Coordinator")
            for agent_name in self.agents.keys():
                print(f"  └── {agent_name}")
