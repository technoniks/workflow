import operator
from typing import Annotated, Literal, TypedDict, get_args, get_origin, get_type_hints


class AgentState(TypedDict):
   question: str
   route: Literal["search", "answer"]
   steps: Annotated[list[str], operator.add]

def apply_update(state: AgentState, update: dict) -> AgentState:
    hints = get_type_hints(AgentState, include_extras=True)
    new = dict(state)
    for key, value in update.items():
        hint = hints[key]
        if get_origin(hint) is Annotated:
            reducer = get_args(hint)[1]
            new[key] = reducer(state[key], value)
        else:
            new[key] = value
    return new

state: AgentState = {
    "question": "What is Langgraph in one sentence?",
    "route": "search",
    "steps": []
}
state = apply_update(state, {
    "steps": ["searched docs"]
})
state = apply_update(state, {
    "route": "answer", "steps": ["wrote answer"]
})
print(state)