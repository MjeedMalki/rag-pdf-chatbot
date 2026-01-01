import os
from uuid import uuid4
from dotenv import load_dotenv, find_dotenv

import tiktoken

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ==================================================
# Environment
# ==================================================
load_dotenv(find_dotenv())

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")

# ==================================================
# Configuration
# ==================================================
DATA_DIR = "./Data"
CHROMA_DIR = "./docs/chroma"
COLLECTION_NAME = "rag_collection"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
BATCH_SIZE = 50
MAX_CONTEXT_TOKENS = 2000

# ==================================================
# Token-safe context builder
# ==================================================
def build_context(docs, max_tokens=MAX_CONTEXT_TOKENS):
    encoder = tiktoken.encoding_for_model("gpt-4.1-mini")
    total_tokens = 0
    chunks = []

    for doc in docs:
        text = doc.page_content
        tokens = len(encoder.encode(text))

        if total_tokens + tokens > max_tokens:
            break

        chunks.append(text)
        total_tokens += tokens

    return "\n\n".join(chunks)

# ==================================================
# Load Documents (ALL PDFs)
# ==================================================
loader = DirectoryLoader(
    path=DATA_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

pages = loader.load()

print(f"Loaded {len(pages)} pages from PDF files")

# ==================================================
# Chunk Documents
# ==================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

documents = text_splitter.split_documents(pages)
print(f"Split into {len(documents)} chunks")

# ==================================================
# Vector Store
# ==================================================
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 1536 dims
)

vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
    persist_directory=CHROMA_DIR
)

# ==================================================
# Safe Batched Ingestion (ONE TIME)
# ==================================================
if vectordb._collection.count() == 0:
    print("Embedding documents into Chroma...")

    ids = [str(uuid4()) for _ in documents]

    for i in range(0, len(documents), BATCH_SIZE):
        vectordb.add_documents(
            documents=documents[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE]
        )

    print("Embedding complete")

else:
    print("Using existing vector store")

# ==================================================
# Retriever
# ==================================================
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)

# ==================================================
# LLM
# ==================================================
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# ==================================================
# Prompt & Chain
# ==================================================
prompt = ChatPromptTemplate.from_template(
    """
Use the following context to answer the question.
If you don't know the answer, say "I don't know".
Use a maximum of two sentences.

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = prompt | llm

# ==================================================
# Chat Loop
# ==================================================
print("\nRAG Chatbot Ready (type 'q' or 'quit' to exit)\n")

while True:
    question = input("You: ").strip().lower()

    if question in {"q", "quit"}:
        break

    docs = retriever.invoke(question)
    context_text = build_context(docs)

    if not context_text:
        print("Assistant: I don't know.")
        continue

    result = chain.invoke(
        {"context": context_text, "question": question}
    )

    print("\nAssistant:", result.content)
    print(
        "Tokens used:",
        result.response_metadata["token_usage"]["total_tokens"]
    )
    print("-" * 60)
