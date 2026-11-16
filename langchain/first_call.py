from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model = init_chat_model(
  model="qwen2.5:3b",
  model_provider="ollama",
  temperature=0,
)

messages = [
  SystemMessage("you are a concise support agent, answer in 1 sentense max."),
  HumanMessage("What is refund policy")
]

response = model.invoke(messages)

print("TYPE:   ", type(response).__name__)
print("CONTENT:", response.content)
# print("USAGE:  ", response.usage_metadata)        # input/output/total tokens
# print("META:   ", response.response_metadata)     # model name, stop reason, etc.

# 3. Multi-turn: append and resend everything
messages.append(response)                          # the AIMessage
messages.append(HumanMessage("Summarize your last answer in 5 words."))
print("\nFOLLOW-UP:", model.invoke(messages).content)

# 4. Streaming: tokens as they're generated
print("\nSTREAM: ", end="")
for chunk in model.stream([HumanMessage("Count from 1 to 10, comma separated.")]):
    print(chunk.content, end="", flush=True)
print()

# 5. Batch: several independent prompts, run concurrently
questions = ["Capital of France?", "Capital of Japan?", "Capital of India?"]
answers = model.batch([[HumanMessage(q + " One word.")] for q in questions])
for q, a in zip(questions, answers):
    print(f"{q} -> {a.content}")