from enum import Enum, StrEnum

class INTENT(Enum):
    RAG=("rag", "retrieve_answer")
    TOOL=("tool", "execute_tool")
    MULTI_STEP=("multi_step", "plan_execute")
    ANSWER=("answer", "direct_answer")
    CLARIFY=("clarify", "ask_user")

    def __init__(self, intent, workflow):
        self.intent = intent
        self.workflow = workflow

    @staticmethod
    def get(intent):
        for i in INTENT:
            if i.intent == intent:
                return i
        raise ValueError(f"Invalid intent: {intent}")

class COMPLEXITY(StrEnum):
    LOW="low", 
    MEDIUM="medium", 
    HIGH="high",

class STATUS(StrEnum):
    RUNNING="running",
    TIME_LIMIT="time_limit",
    STEP_LIMIT="step_limit",
    COMPLETED="completed",
    NEED_CLARIFICATION="need_clarification"

class ToolName(StrEnum):
    CALCULATOR = "calculator"
    DATETIME = "get_current_datetime"
    KNOWLEDGE = "search_knowledge"