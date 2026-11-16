import json
from typing import Literal
from pydantic import BaseModel, Field, ValidationError, field_validator


class Ticket(BaseModel):
    """Triage result for one customer support ticket."""
    
    category: Literal["billing", "bug", "feature_request", "other"] = Field(
        description="which team should handle the ticket"
    )
    priority: int = Field(ge=1, le=5, description="1 = lowest, 5 = urgent")
    summary: str = Field(description="One-sentence summary of the problem")
    tags: list[str] = Field(default_factory=list, description="Short keywords")

    @field_validator("summary")
    @classmethod
    def summary_not_too_long(cls, v: str) -> str:
        if len(v) > 120:
            raise ValueError("Summary must be 120 characters or fewer")
        return v.strip()

# 1. The schema that gets sent to LLM
# print(json.dumps(Ticket.model_json_schema(), indent=2))

# 2. A good LLM response
good = '{"category": "billing", "priority": "4", "summary": "Charged twice for Octomber"}'
t = Ticket.model_validate_json(good)
print(t)
# print(t.model_dump())

# 3. A bad LLM responses
bad = '{"category": "bug", "priority": 2, "summary": "Charged twice"}'
try:
    Ticket.model_validate_json(bad)
except ValidationError as e:
    print(e)