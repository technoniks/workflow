import os
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    """Triage result for one customer support ticket."""
    
    category: Literal["billing", "bug", "feature_request", "other"] = Field(
        description="Which team should handle the ticket"
    )
    priority: int = Field(ge=1, le=5, description="1=lowest, 5=urgent(outage, data loss, money lost)")
    sentiment: str = Field(description="One neutral sentence, max 20 words")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You triage support tickets for {company}.\n"
     "Rules:\n"
     "- Priority 5 only for outages, data loss, or customers losing money.\n"
     "- Feature ideas are never above priority 2.\n"
     "Example: 'App crashes when I upload a PDF' -> bug, priority 4."),
    ("human", "Ticket:\n{ticket}"),
])

model = init_chat_model(
    "qwen2.5:3b",
    model_provider="ollama",
    temperature=0
)
structured = model.with_structured_output(Ticket)

chain = (prompt | structured).with_retry(stop_after_attempt=3)

# --- single ticket ---
def run_single_ticket():
    t = chain.invoke({
        "company": "Acme Cloud",
        "ticket": "You charged my card twice!! Fix it NOW."
    })
    print(type(t).__name__, t.model_dump())

# --- many tickets at once ---
def run_many_tickets():
    tickets = [
        "Would love a dark mode option.",
        "Whole dashboard is down since 9am, our team can't work.",
        "Thanks, the new export feature is great!",
    ]
    results = chain.batch([{"company": "Acme Cloud", "ticket": x} for x in tickets])
    for text, r in zip(tickets, results):
        print(f"[{r.category:15}] p{r.priority} {r.sentiment:8} | {text}")
    
    
def debug_output():
    debug = model.with_structured_output(Ticket, include_raw=True)
    out = (prompt | debug).invoke({"company": "Acme", "ticket": "refund pls"})
    print(out["raw"])            # the AIMessage the model actually sent
    print(out["parsed"])         # Ticket, or None if it failed
    print(out["parsing_error"])  # the error, if any
    
print(prompt.invoke({"company": "Acme", "ticket": "hello"}).to_messages())