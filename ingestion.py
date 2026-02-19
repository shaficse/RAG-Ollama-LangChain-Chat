"""
Tutorial-style ingestion pipeline for RAG.

What this script does:
1. Crawl LangChain docs pages using Tavily.
2. Convert each crawled page into a LangChain `Document`.
3. Split long pages into smaller overlapping chunks.
4. Create embeddings for each chunk using Ollama (`qwen3-embedding`).
5. Store embeddings in local Chroma and optionally Pinecone (or both).

How to read this file:
- Top section: setup/configuration and dependency initialization.
- `index_documents_async`: batch indexing logic.
- `main`: end-to-end orchestration.
"""

# `asyncio` provides Python's async/await runtime.
import asyncio
# `os` is used for environment variables like API URLs/keys.
import os
# `ssl` lets us configure certificate behavior for HTTPS requests.
import ssl
# `Dict` and `List` are type hints to make function inputs/outputs explicit.
from typing import Dict, List

import certifi
# `requests` performs HTTP calls for preflight checks.
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)

# Tavily -> chunk -> embed -> vector store ingestion pipeline.
# Loads variables from `.env` into process environment.
# Example: OLLAMA_BASE_URL, PINECONE_API_KEY, TAVILY_API_KEY
load_dotenv()

# Normalize certificate handling across libraries that rely on requests/SSL.
# Some HTTP clients look at SSL_CERT_FILE, others look at REQUESTS_CA_BUNDLE.
# Setting both avoids certificate issues across different dependencies.
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Ollama endpoints for local LLM + embeddings.
# If OLLAMA_BASE_URL is not in your .env file, this fallback is used.
# Syntax note:
# os.getenv("KEY", "fallback") -> returns env var value if present, otherwise fallback.
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.10.114:11434")
# Kept for parity with the rest of the project where chat generation is used.
# It is not used in this ingestion-only script, but useful if you later add QA.
chat_ollama = ChatOllama(model="qwen3:8b", base_url=ollama_base_url)
# Embedding model converts text chunks into vectors (list of floats).
embedding = OllamaEmbeddings(model="qwen3-embedding", base_url=ollama_base_url)

# Default local vector store for development runs.
# Chroma persists vectors locally in ./chroma_db.
local_vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# Extract/Map are initialized for future expansion (targeted extraction/site mapping).
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def build_active_vectorstores() -> Dict[str, object]:
    """
    Build all vector stores that should receive embeddings.

    Behavior:
    - Local Chroma is always enabled.
    - Pinecone is also enabled when credentials/index configuration are present.

    Environment variables:
    - `ENABLE_PINECONE` (optional): true/false (default: true)
    - `PINECONE_INDEX_NAME` (optional): preferred index name
    - `INDEX_NAME` (fallback): used if PINECONE_INDEX_NAME is missing
    - `PINECONE_API_KEY`: required for Pinecone operations
    """
    # Type hint on variable:
    # `stores: Dict[str, object]` means
    # - key type is str (store name)
    # - value type is generic object (store instance)
    stores: Dict[str, object] = {"local_chroma": local_vectorstore}
    log_info("✅ Local Chroma vector store enabled (`./chroma_db`).", Colors.GREEN)

    # Convert env text to boolean:
    # - `.lower()` normalizes case
    # - `in {...}` checks membership in accepted truthy strings
    enable_pinecone = os.getenv("ENABLE_PINECONE", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    if not enable_pinecone:
        log_warning("Pinecone disabled by ENABLE_PINECONE=false")
        return stores

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # Nested getenv fallback:
    # 1) Use PINECONE_INDEX_NAME when present
    # 2) else use INDEX_NAME when present
    # 3) else default to "langchain-docs-2025"
    pinecone_index_name = os.getenv(
        "PINECONE_INDEX_NAME", os.getenv("INDEX_NAME", "langchain-docs-2025")
    )

    if not pinecone_api_key:
        log_warning(
            "Pinecone requested, but PINECONE_API_KEY is missing. "
            "Continuing with local Chroma only."
        )
        return stores

    try:
        stores["pinecone"] = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=embedding,
        )
        log_info(
            f"✅ Pinecone vector store enabled (index: {pinecone_index_name}).",
            Colors.GREEN,
        )
    except Exception as exc:
        log_warning(
            f"Pinecone initialization failed ({exc}). "
            "Continuing with local Chroma only."
        )

    return stores


def run_preflight_checks(timeout_seconds: int = 8) -> bool:
    """
    Validate model connectivity before expensive crawl/chunk/index steps.

    Why this matters:
    - If the embedding model is not reachable, vector preparation will fail.
    - If Ollama is down, you want to fail fast before crawling many pages.
    """
    log_header("PREFLIGHT CHECKS")
    # `rstrip("/")` removes trailing slash only from the right side.
    # This avoids accidental "//api/..." in URL joins.
    ollama_api_base = ollama_base_url.rstrip("/")
    chat_model_name = "qwen3:8b"
    embedding_model_name = "qwen3-embedding"

    # 1) Verify Ollama server is reachable and models are listed.
    try:
        log_info("🔎 Checking Ollama server + model list...", Colors.BLUE)
        # `requests.get(..., timeout=...)` raises if connection/read is too slow.
        tags_response = requests.get(
            f"{ollama_api_base}/api/tags",
            timeout=timeout_seconds,
        )
        # Raises HTTPError for non-2xx status codes.
        tags_response.raise_for_status()
        # Parse JSON body into a Python dict.
        models_payload = tags_response.json().get("models", [])
        # List comprehension syntax:
        # [expression for item in iterable]
        available_models = [str(item.get("name", "")) for item in models_payload]
    except requests.RequestException as exc:
        log_error(
            f"Ollama server check failed at {ollama_api_base}: {exc}. "
            "Make sure Ollama is running and OLLAMA_BASE_URL is correct."
        )
        return False

    def model_available(target_model: str) -> bool:
        """Allow matching by exact name or by base name without tag."""
        # Example: "qwen3:8b" -> base "qwen3"
        target_base = target_model.split(":")[0]
        # any(...) returns True if at least one item is True.
        return any(
            name == target_model or name.split(":")[0] == target_base
            for name in available_models
        )

    if not model_available(chat_model_name):
        log_error(
            f"Chat model '{chat_model_name}' not found in Ollama model list: {available_models}"
        )
        return False
    if not model_available(embedding_model_name):
        log_error(
            f"Embedding model '{embedding_model_name}' not found in Ollama model list: {available_models}"
        )
        return False
    log_success("Ollama server and required models are available.")

    # 2) Do a real chat generation call to verify inference path.
    try:
        log_info("🔎 Running chat inference health check...", Colors.BLUE)
        # POST sends JSON payload to the model inference endpoint.
        chat_response = requests.post(
            f"{ollama_api_base}/api/generate",
            json={
                "model": chat_model_name,
                "prompt": "Reply with OK only",
                "stream": False,
            },
            timeout=timeout_seconds,
        )
        chat_response.raise_for_status()
        # Safely extract "response" from JSON and normalize with str(...).strip().
        chat_text = str(chat_response.json().get("response", "")).strip()
        if not chat_text:
            log_error("Chat model check failed: empty generation response.")
            return False
        log_success("Chat model check passed.")
    except requests.RequestException as exc:
        log_error(f"Chat model check failed: {exc}")
        return False

    # 3) Do a real embedding call and validate vector shape.
    try:
        log_info("🔎 Running embedding health check...", Colors.BLUE)
        embedding_response = requests.post(
            f"{ollama_api_base}/api/embeddings",
            json={
                "model": embedding_model_name,
                "prompt": "health check for vector ingestion",
            },
            timeout=timeout_seconds,
        )
        embedding_response.raise_for_status()
        # Expected shape is a list of numbers like [0.01, -0.2, ...]
        sample_vector = embedding_response.json().get("embedding", [])
        if not sample_vector:
            log_error("Embedding check failed: received an empty vector.")
            return False
        # Runtime type check for first element to verify numeric vector.
        if not isinstance(sample_vector[0], (int, float)):
            log_error("Embedding check failed: vector values are not numeric.")
            return False
        log_success(f"Embedding model check passed. Vector dimension: {len(sample_vector)}")
    except requests.RequestException as exc:
        log_error(f"Embedding model check failed: {exc}")
        return False

    # 3) Readiness note for learners.
    log_success(
        "Preflight checks passed. This is the right time to prepare vectors (crawl -> chunk -> embed -> store)."
    )
    return True


async def index_documents_async(
    documents: List[Document],
    vectorstores: Dict[str, object],
    batch_size: int = 50,
):
    """
    Index chunked documents in asynchronous batches.

    Why batches:
    - Prevents sending all chunks in one huge request.
    - Improves reliability and memory usage.
    - Allows concurrent writes for better throughput.
    """
    # `async def` means this function can use `await` for non-blocking I/O.
    log_header("VECTOR STORAGE PHASE")
    # ", ".join(list_of_strings) merges names into readable text.
    store_names = ", ".join(vectorstores.keys())
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector stores: {store_names}",
        Colors.DARKCYAN,
    )

    # Create batches
    # This list comprehension slices the full list in windows of `batch_size`.
    # Example with batch_size=3: [0:3], [3:6], [6:9], ...
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Inner function to index one batch into one target store.
    # `aadd_documents` will embed page_content and store vectors+metadata.
    async def add_batch_to_store(
        store_name: str, store: object, batch: List[Document], batch_num: int
    ):
        try:
            # `await` pauses this coroutine until async write completes.
            # type: ignore[attr-defined] is used because `store` is typed as object.
            # At runtime, both Chroma and Pinecone vector store support aadd_documents.
            await store.aadd_documents(batch)  # type: ignore[attr-defined]
            log_success(
                f"{store_name}: added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            # We return False instead of raising so other batches can continue.
            log_error(f"{store_name}: failed to add batch {batch_num}/{len(batches)} - {e}")
            return False
        return True

    # Process batches concurrently. Exceptions are captured so one failed batch
    # does not cancel the entire indexing job.
    # Nested comprehension syntax:
    # for each batch -> for each store -> create one async task.
    tasks = [
        add_batch_to_store(store_name, store, batch, batch_num)
        for batch_num, batch in enumerate(batches, start=1)
        for store_name, store in vectorstores.items()
    ]
    # `asyncio.gather` runs all tasks concurrently and returns results in order.
    # `return_exceptions=True` keeps failures as values instead of crashing all.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Only explicit True counts as success; exceptions/False are failures.
    # When `return_exceptions=True`, raised exceptions appear in this list.
    # Generator expression syntax:
    # (1 for result in results if result is True)
    successful = sum(1 for result in results if result is True)
    expected_total = len(batches) * len(vectorstores)

    if successful == expected_total:
        log_success(
            "VectorStore Indexing: All batch writes succeeded across all targets! "
            f"({successful}/{expected_total})"
        )
    else:
        log_warning(
            "VectorStore Indexing: Some writes failed. "
            f"Successful writes: {successful}/{expected_total}"
        )


async def main():
    """
    Orchestrate full ingestion:
    crawl -> transform -> chunk -> index.
    """
    # Entry coroutine for the full workflow.
    log_header("DOCUMENTATION INGESTION PIPELINE")

    # Fail fast if local models are not ready.
    if not run_preflight_checks():
        log_error(
            "Preflight failed. Fix model/server issues first, then run ingestion again."
        )
        return

    # Build all vector store targets (local + optional Pinecone).
    vectorstores = build_active_vectorstores()
    if not vectorstores:
        log_error("No active vector stores configured. Nothing to index.")
        return

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl docs pages first.
    # `extract_depth="advanced"` asks Tavily for richer extracted text per page.
    # Expected env var: TAVILY_API_KEY
    res = tavily_crawl.invoke(
        {
            # Crawl starting URL.
            "url": "https://python.langchain.com/",
            # Follow links this many levels deep.
            "max_depth": 2,
            # Ask Tavily for stronger text extraction quality.
            "extract_depth": "advanced",
        }
    )

    # Convert crawl results into LangChain Documents for downstream splitting.
    # Tavily response shape (simplified):
    # {
    #   "results": [
    #      {"url": "...", "raw_content": "..."},
    #      ...
    #   ]
    # }
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                # Main textual content used for splitting/embedding.
                page_content=tavily_crawl_result_item["raw_content"],
                # Metadata travels with chunks; useful later for source citation.
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Chunking keeps context windows manageable for embedding and retrieval.
    # chunk_size=4000:
    # - Larger chunks keep more context from each page.
    # chunk_overlap=200:
    # - Repeats small boundaries so important text is less likely to be split
    #   in a way that harms retrieval quality.
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    # Returns a new list of smaller Document chunks.
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Persist chunk embeddings in batched async writes.
    # batch_size=500 is a throughput-focused default; reduce this if you hit
    # resource limits or API throttling in your environment.
    await index_documents_async(splitted_docs, vectorstores, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    # Standard Python entrypoint pattern:
    # this block runs only when file is executed directly.
    asyncio.run(main())
