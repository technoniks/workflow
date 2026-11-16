import os, asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

primary_llm = init_chat_model(
    "gemini-3.1-flash-lite", 
    model_provider="google_genai", 
    temperature=0,
    max_retries=0
)
backup_llm = init_chat_model(
    "qwen2.5:3b",
    model_provider="ollama",
    temperature=0
)

#-------- 1. Retry + fallback ------------
def run_retry_fallback():
    broken = init_chat_model("model-that-not-exist", model_provider="ollama")
    reliable = (
        broken.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=True
        ).with_fallbacks([backup_llm])
    )
    print("1. FALLBACK:", reliable.invoke("Say, 'Fallback workes', nothing else").content)
# run_retry_fallback()

#------- 2. Streaming text and tool calls --------
@tool
def get_weather(city: str) -> str:
    """Get current whether for a city, e.g. 'Pune'."""
    return f"{city}: 31°C, Sunny"

model_tool = primary_llm.bind_tools([get_weather])
def run_stream():
    print("\n2. STREAM:")
    full = None
    for chunk in model_tool.stream([HumanMessage("What is the weather in Pune")]):
        if chunk.content:
            print(" text: ", chunk.content)
        if chunk.tool_call_chunks:
            print(" tool: ", chunk.tool_call_chunks)
        full = chunk if full is None else full + chunk
    print(" final tool_calls:", full.tool_calls)
# run_stream()

# ------ 3. Streaming events from a whole chain --------
prompt = ChatPromptTemplate.from_template("Write a haiku about {topic}")
chain = prompt | primary_llm

async def stream_events():
    print("\n3. EVENTS:")
    async for ev in chain.astream_events({"topic": "monsoon"}, version="v2"):
        kind = ev["event"]
        if kind == "on_chat_model_stream":
            print([text['text']for text in ev["data"]["chunk"].content], end="\n", flush=True)
        elif kind in ("on_chain_start", "on_chain_end", "on_chat_model_start"):
            print(f"\n  [{kind}] {ev['name']}")
    print()    
# asyncio.run(stream_events())

#------- 4. Timeout --------
async def with_timeout():
    
    try:
        async with asyncio.timeout(5):
            with get_usage_metadata_callback() as cb:
                ai:AIMessage = await primary_llm.ainvoke("Write a 5 lines eassy on Indian history.")
                print(f"\nAI: {ai.content[-1]['text']}")
            print("\n5. USAGE:", cb.usage_metadata) 
    except TimeoutError:
        print("\n4. TIMEOUT: call cancelled -> return a fallback message or 504")

# asyncio.run(with_timeout())

# ---------- 5. Token usage tracking ----------
def run_usage():
    with get_usage_metadata_callback() as cb:
        primary_llm.invoke("Hi")
        primary_llm.invoke("Name three rivers in Maharashtra.")
    print("\n5. USAGE:", cb.usage_metadata) 
# run_usage()

# -------- 6. Tags and metadata (used by Langsmith later)----
def run_tags_metadata():
    ai = primary_llm.invoke(
        "Hello",
        config={
            "run_name": "greeting",
            "tags": ["lesson-1.4"],
            "metadata": {
                "user_id": "u_43"
            }
        }
    )
    print(f"AI: {ai}")
run_tags_metadata()