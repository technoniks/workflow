from typing import Any
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    user_input: str
    history: list[dict[str, Any]]
    context: str
    intent: str
    complexity: str
    plan: list[str]
    current_step: int
    max_steps: int
    deadline: float
    status: str
    results: list[dict[str, Any]]
    answer: str
    error: str
