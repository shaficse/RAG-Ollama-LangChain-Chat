"""
Core RAG runtime for backend queries.

Flow:
1) Initialize Ollama chat + embedding models.
2) Connect to Pinecone vector index.
3) Expose a retrieval tool for the agent.
4) Run agent -> retrieve docs -> produce final answer.

Term Glossary (beginner-friendly):
- LLM: Large Language Model used to generate natural-language answers.
- Embedding: Numeric vector representation of text semantics.
- Vector Store: Database optimized for similarity search over embeddings.
- Retriever: Helper that queries the vector store and returns top-k documents.
- Tool: A Python function the agent can call during reasoning.
- ToolMessage: Special message object produced when a tool is invoked.
- Artifact: Structured non-text payload attached to a ToolMessage (here: raw docs).
- System Prompt: High-level behavior instructions for the assistant.
"""

# `os` is used to read environment variables (for URLs/index names).
import os
from typing import Any, Dict

# Load `.env` values into process environment.
from dotenv import load_dotenv
# Agent helper that lets the model decide when to call tools.
from langchain.agents import create_agent
# ToolMessage is emitted after a tool call. It can carry:
# - readable tool output text
# - structured artifact payload (e.g., list of Documents)
from langchain.messages import ToolMessage
# `@tool` decorator turns a Python function into an agent-usable tool.
from langchain.tools import tool
# Ollama-based chat and embedding models (aligned with ingestion.py).
from langchain_ollama import ChatOllama, OllamaEmbeddings
# Pinecone vector store for semantic retrieval.
from langchain_pinecone import PineconeVectorStore

# Reads variables like OLLAMA_BASE_URL / PINECONE_INDEX_NAME from `.env`.
load_dotenv()

# Keep backend model setup aligned with ingestion.py
# If OLLAMA_BASE_URL is not set, fallback to the given LAN endpoint.
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.10.114:11434")
# Chat model used by the agent for reasoning + final answer generation.
model = ChatOllama(model="qwen3:8b", base_url=ollama_base_url)
# Embedding model must match what was used for indexing documents.
embeddings = OllamaEmbeddings(model="qwen3-embedding", base_url=ollama_base_url)
# Index name fallback chain:
# 1) PINECONE_INDEX_NAME
# 2) INDEX_NAME
# 3) "langchain-docs-2025"
pinecone_index_name = os.getenv(
    "PINECONE_INDEX_NAME", os.getenv("INDEX_NAME", "langchain-docs-2025")
)

# Initialize vector store + retriever
# `PineconeVectorStore` knows how to embed queries and search vector index.
vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
# Retriever config:
# - similarity: nearest-neighbor search
# - k=4: return top 4 relevant chunks
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # This function is registered as a "tool" that the agent can call.
    # response_format="content_and_artifact" means we return:
    # 1) text content for the LLM to read
    # 2) raw data artifact for programmatic use later

    # Retrieve top 4 most similar documents
    retrieved_docs = retriever.invoke(query)

    # Serialize documents into readable text for the model.
    # Each block keeps source + content so final answers can cite references.
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both:
    # - `serialized` for model consumption (tool output text)
    # - `retrieved_docs` as artifact for programmatic access after run
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # System prompt defines the assistant behavior and citation policy.
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # Build an agent that can decide when to call `retrieve_context`.
    # The agent loop usually does:
    # - think about the question
    # - call tools if needed
    # - read tool results
    # - produce final answer
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Agent expects chat-style message list.
    messages = [{"role": "user", "content": query}]

    # Run the full agent loop (LLM reasoning + tool calls + final response).
    response = agent.invoke({"messages": messages})

    # By convention, last message is the assistant's final answer.
    answer = response["messages"][-1].content

    # Collect retrieved docs from tool artifacts so caller can inspect context.
    # Why this works:
    # - When the tool runs, LangChain creates a ToolMessage.
    # - Because we used content_and_artifact, the ToolMessage contains:
    #   * content: serialized text
    #   * artifact: raw retrieved document objects
    context_docs = []
    for message in response["messages"]:
        # Keep only tool-related messages.
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # Artifact should be a list[Document] from retrieve_context.
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":
    # Quick manual test when running this file directly.
    result = run_llm(query="what are deep agents?")
    print(result)
