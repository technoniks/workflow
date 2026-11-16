import uuid

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models, QdrantClient
from ingest import load_chunks


embedding = OllamaEmbeddings(model="nomic-embed-text")
# Get embedding dimension
embedding_dim = len(
    embedding.embed_query("hello world")
)
chunks = load_chunks()

client = QdrantClient(
    url="http://localhost:6333",
)

collection_name = "kb"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE
        )
    )
store = QdrantVectorStore(
    client=client,
    embedding=embedding,
    collection_name=collection_name
)

def run_indexing():
    store.add_documents(
        chunks
    )
    print(f"Indexed {len(chunks)} chunks\n")
# run_indexing()

# ---------- 1. Plain similarity search ----------
def run_search():
    for q in ["How do I get my money back?", "Does Pro include single sign-on?", "site is down"]:
        print(f"Q: {q}")
        for doc, score in store.similarity_search_with_score(q, k=2):
            print(f"    {score:.3f} [{doc.metadata['source']} > {doc.metadata.get('section', '')}] {doc.page_content}")
        print()
# run_search()
    
# ---------- 2. Metadata filter ----------
def run_metadat_filtering():
    billing_only = models.Filter(
        must=[models.FieldCondition(
            key="metadata.category", 
            match=models.MatchValue(value="billing")
        )])
    for doc in store.similarity_search("credits and money", k=3, filter=billing_only):
        print(f"\n[{doc.metadata}]\n-> {doc.page_content}")
# run_metadat_filtering()

# ---------- 3. As a retriever (the interface chains and agents use) ----------
def run_retrieval():
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k":3, "fetch_k": 10})
    print("\nRetriever (MMR):")
    for doc in retriever.invoke("what does Enterprise get?"):
        print(f"{doc.metadata.get('section')}: {doc.page_content}\n")
run_retrieval()