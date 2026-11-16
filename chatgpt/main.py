import time
from router import RouteDecision, classify_request
from state import AgentState
from enums import INTENT, STATUS;
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='qwen2.5:3b',
    temperature=0
)

def log_node(node_name: str) -> None:
    print(f"[workflow] entering node: {node_name}")

def build_context(state: AgentState) -> AgentState:
    log_node("context")
    state.context = ""
    return state

def route_request(state: AgentState) -> AgentState:
    log_node("router")
    decision: RouteDecision = classify_request(
        state.user_input,
        state.context
    )
    state.intent = decision.intent
    state.complexity = decision.complexity
    return state

def budget_check(state: AgentState) -> AgentState:
    log_node("budget")
    if time.monotonic() >= state.deadline:
        state.status = STATUS.TIME_LIMIT.value

    if state.current_step >= state.max_steps:
        state.status = STATUS.STEP_LIMIT.value

    return state

def select_next(state: AgentState) -> str:
    if state.status != STATUS.RUNNING.value:
        print("[workflow] selected route: finalize")
        return "finalize"
    selected_intent = INTENT.get(state.intent)
    print(
        f"[workflow] selected route: {selected_intent.intent} "
        f"(workflow: {selected_intent.workflow})"
    )
    return selected_intent.workflow

def answer_node(state: AgentState) -> AgentState:
    log_node(INTENT.ANSWER.intent)
    result = llm.invoke(state.user_input)
    state.answer = result.content
    state.current_step = state.current_step + 1
    state.status = STATUS.COMPLETED.value
    return state

def rag_node(state: AgentState) -> AgentState:
    log_node(INTENT.RAG.intent)
    result = llm.invoke(
        f"""
        Answer the question using the supplied context.
        If context is empty, say that no retrieved contaxt was provided.
        Context: {state.context}
        Question: {state.user_input}
        """
    )
    state.answer = result.content
    state.current_step = state.current_step + 1
    state.status = STATUS.COMPLETED.value
    return state

def tool_node(state: AgentState) -> AgentState:
    log_node(INTENT.TOOL.intent)
    state.answer = "No tool has been configured yet."
    state.current_step = state.current_step + 1
    state.status = STATUS.COMPLETED.value
    return state

def multi_step_node(state: AgentState) -> AgentState:
    log_node(INTENT.MULTI_STEP.intent)
    state.answer = (
            "The multi-step workflow is not configured yet. "
            "Implement a planner and bounded executor here."
        )
    state.current_step = state.current_step + 1
    state.status = STATUS.COMPLETED.value
    return state

def clarify_node(state: AgentState) -> AgentState:
    log_node(INTENT.CLARIFY.intent)
    state.answer = "Could you provide more details about your request?"
    state.status = STATUS.NEED_CLARIFICATION.value
    return state
    


def finalize_node(state: AgentState) -> AgentState:
    log_node("finalize")
    if state.status == STATUS.TIME_LIMIT.value:
        state.answer = "Execution stopped because the time budget was exhausted."
    elif state.status == STATUS.STEP_LIMIT.value:
        state.answer = "Execution stopped because the step limit was reached."
    else:
        state.answer = state.answer
    return state

# ---------- Graph ----------

builder = StateGraph(AgentState)

# Application nodes
builder.add_node("context", build_context)
builder.add_node("router", route_request)
builder.add_node("budget", budget_check)

# Dicision nodes
builder.add_node(INTENT.ANSWER.intent, answer_node)
builder.add_node(INTENT.RAG.intent, rag_node)
builder.add_node(INTENT.TOOL.intent, tool_node)
builder.add_node(INTENT.MULTI_STEP.intent, multi_step_node)
builder.add_node(INTENT.CLARIFY.intent, clarify_node)
builder.add_node("finalize", finalize_node)

# Node Routing
builder.add_edge(START, "context")
builder.add_edge("context", "router")
builder.add_edge("router", "budget")

builder.add_conditional_edges(
    "budget",
    select_next,
    {
        i.workflow: i.intent
        for i in INTENT
    },
)

for node in [i.intent for i in INTENT]:
    builder.add_edge(node, "finalize")

builder.add_edge("finalize", END)

graph = builder.compile()


def run_agent(
    user_input: str,
    max_steps: int = 8,
    time_budget_seconds: float = 20,
) -> AgentState:
    start = time.monotonic()

    initial_state: AgentState = {
        "user_input": user_input,
        "history": [],
        "context": "",
        "intent": "",
        "complexity": "",
        "plan": [],
        "current_step": 0,
        "max_steps": max_steps,
        "deadline": start + time_budget_seconds,
        "status": STATUS.RUNNING.value,
        "results": [],
        "answer": "",
        "error": "",
    }

    return graph.invoke(
        initial_state,
        config={
            # Graph-level safety net; this is not the same
            # as the application-level workflow step budget.
            "recursion_limit": max_steps + 10,
        },
    )

while True:
    user_input = input("Enter your request (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    response = run_agent(
        user_input,
        max_steps=8,
        time_budget_seconds=10,
    )
    agent_state = AgentState.model_validate(response)
    print("\n\nFinal response:", agent_state.answer)