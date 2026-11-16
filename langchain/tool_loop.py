import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ------1. Define Tools--------
FAKE_ORDERS = {
    "A100": {"status": "delivered", "amount": 49.99, "days_ago": 8},
    "A200": {"status": "shipped", "amount": 120.00, "days_ago": 2}
}

@tool
def get_order_status(order_id: str) -> dict:
    """Look up an order by its ID (e.g. 'A100'). Returns status, amount and days since purchase."""
    
    order = FAKE_ORDERS.get(order_id.upper())
    if order is None:
        raise ValueError(f"Order {order_id} not found")
    return order

@tool
def calculate_refund(amount: float, days_since_purchase: int) -> float:
    """Calculate the refund amount."""
    
    if days_since_purchase <= 7:
        return amount
    elif days_since_purchase <= 30:
        return round(amount * 0.5, 2)
    else:
        return 0.0
    
@tool
def cancel_order(order_id: str) -> dict:
    """Cancel the order by its order id(e.g. A100). Return True if status is shipped and Otherwise False"""
    order = FAKE_ORDERS.get(order_id.upper())
    if order is None:
        raise ValueError(f"Order {order_id} not found")
    if order["status"] == "shipped":
        order["cancelled"] = True
    else:
        order["cancelled"] = False
    return order
    
    
tools = [get_order_status, calculate_refund, cancel_order]
tools_by_name = {t.name: t for t in tools}

# for t in tools:
    # print(f"{t.name}: {t.description}\n args: {t.args}")
    
# ------2. Bind tools to the model--------
# model = init_chat_model(
#     "qwen2.5:3b",
#     model_provider="ollama",
#     temperature=0
# )
model = init_chat_model(
    "gemini-3.1-flash-lite",
    model_provider="google_genai",
    temperature=0
)
# model = ChatGoogleGenerativeAI(
#     model="gemini-3.1-flash-lite",
#     temperature=0
# )
model_with_tools = model.bind_tools(tools)    

# ------3. The Agent loop--------
def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [
        SystemMessage("You are the support agent. \
            Rule 1 - Use tools to look up facts. Never guess order details. \
            Rule 2 - Use tool to if any financial decision reqiored like 'refund' or 'product price'\
            Rule 3 - For general question which not required any tools call, then just reply exactly 'Sorry, I can only help you releted to our products'"),
        HumanMessage(question)
    ]
    for step in range(max_steps):
        ai: AIMessage = model_with_tools.invoke(messages)
        messages.append(ai)
        
        if not ai.tool_calls:
            return ai.content[-1]['text']
        
        for call in ai.tool_calls:
            print(f"#step-{step+1}: {call['name']}({call['args']})")
            try:
                result = tools_by_name[call['name']].invoke(call['args'])
                content = str(result)
                status = "success"
            except Exception as e:
                content = f"ERROR: {e}"
                status = "error"
            print(f"    -> {content}")
            messages.append(
                ToolMessage(content, tool_call_id=call['id'], status=status)
            )
    return "Stopped: too many steps"

# ---------- 4. Try it ----------
for q in [
    "Compare refunds for A100 and A200, use tools for refund",
    "Cancel A100 and A200?",
    "How much refund can I get for order A100?",
    "Refund for order Z999?",
    "What is 2 + 2?",
]:
    print(f"\nQ: {q}")
    print(f"A: {run_agent(q)}")