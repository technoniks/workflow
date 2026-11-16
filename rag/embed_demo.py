import math

from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(model="nomic-embed-text")

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

q = embed.embed_query("How do I get my money back?")
texts = [
    "Customers can request a full refund withing 7 days of purchase.",
    "Pro costs $49 per month and includes SSO",
    "Report outages at status.acmr.dev.",
]
vectors = embed.embed_documents(texts)

print("vector length: ", len(q))
for t, v in zip(texts, vectors):
    print(f"{cosine(q, v):.3f} {t}")