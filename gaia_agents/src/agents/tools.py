# Set up tools

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

from tools.custome_wikipedia_tool import wikipedia_tool

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
    wikipedia_tool,
    SemanticScholarQueryRun(),
    ArxivQueryRun(),
    repl_tool,
]
