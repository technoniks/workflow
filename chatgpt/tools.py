import ast
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.tools import tool
from enums import ToolName


KNOWLEDGE_BASE = {
    "python": "Python is a high-level programming language.",
    "langchain": "LangChain is a framework for building LLM applications.",
    "langgraph": "LangGraph is used to build stateful agent workflows.",
    "rag": "RAG combines retrieval systems with generative language models.",
}

@tool
def calculator(expression: str) -> str:
    """
    A calculator that can evaluate mathematical expressions.
    """
    try:
        # Safely evaluate the expression using ast
        node = ast.parse(expression, mode='eval')
        if not all(isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load)) for n in ast.walk(node)):
            raise ValueError("Invalid expression")
        result = eval(compile(node, '<string>', 'eval'))
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """Get current date and time for a specified timezone."""

    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Datetime error: {e}"

@tool
def search_knowledge(query: str) -> str:
    """Search the local demo knowledge base."""

    query = query.lower()

    results = [
        value
        for key, value in KNOWLEDGE_BASE.items()
        if key in query or query in key
    ]

    if not results:
        return "No matching knowledge found."

    return "\n".join(results)

TOOL_REGISTRY = {
    ToolName.CALCULATOR: calculator,
    ToolName.DATETIME: get_current_datetime,
    ToolName.KNOWLEDGE: search_knowledge,
}

def get_all_tools():
    return list(TOOL_REGISTRY.values())


def get_tool(name: ToolName):
    return TOOL_REGISTRY.get(name)