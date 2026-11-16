from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def load_chunks(path: str = "rag/kb"):
    docs = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader).load()
    md = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "doc_title"), ("##", "section")])
    rec = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=40,add_start_index=True)
    chunks = []
    for doc in docs:
        for chunk in md.split_text(doc.page_content):
            chunk.metadata["source"] = doc.metadata["source"]
            chunk.metadata["category"] = "billing" if any(
                w in doc.metadata["source"] for w in ("refund", "plans")
            ) else "support"
            chunks.append(chunk)
    return rec.split_documents(chunks)
            