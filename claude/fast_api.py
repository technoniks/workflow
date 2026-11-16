import asyncio
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Agentic Lab")

class TriageRequest(BaseModel):
    text: str = Field(min_length=5, max_length=2000)

class TriageResponse(BaseModel):
    category: Literal["billing", "bug", "other"]
    context: list[str]

async def search_kb(text: str) -> str:
    await asyncio.sleep(1.0)
    return f"kb article matching '{text[:20]}'" 

async def lookup_customer(text: str) -> str:
    await asyncio.sleep(0.8)                       # pretend DB call
    return "customer: premium plan"


def fake_classify(text: str) -> str:              # an LLM replaces this in Phase 1
    t = text.lower()
    if "charge" in t or "invoice" in t:
        return "billing"
    if "error" in t or "crash" in t:
        return "bug"
    return "other"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/triage", response_model=TriageResponse)
async def triage(req: TriageRequest):
    try:
        async with asyncio.timeout(3):
            kb, customer = await asyncio.gather(
                search_kb(req.text), lookup_customer(req.text)
            )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="tools timed out")
    return TriageResponse(category=fake_classify(req.text), context=[kb, customer])

@app.post("/stream")
async def stream(req: TriageRequest):
    async def events():
        for word in f"Received: {req.text}".split():
            await asyncio.sleep(0.2)
            yield f"data: {word}\n\n"
        yield "data: [Done]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
            