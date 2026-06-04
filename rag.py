"""Core RAG pipeline for the Upwork API Technical Support Bot.

The functions below map one-to-one onto the assignment stages:

    A1  load_document      - read the source documentation
    A1  sanity_check       - verify the file was read correctly
    A2  chunk_text         - split into overlapping 500-char chunks
    A3  build_vector_store - embed locally + persist a FAISS index
    A3  load_vector_store  - reload that index from disk
    B1  retrieve           - fetch the top-k most relevant chunks
    B2  answer_query       - prompt the LLM with the retrieved context + guards

Keeping this logic in one importable module means both the one-off ingestion
script (`ingest.py`) and the Streamlit UI (`app.py`) share the exact same code
path, so what gets indexed is exactly what gets queried.
"""
from __future__ import annotations

import time

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# ==========================================================================
# A1 - Data Ingestion
# ==========================================================================
def load_document(path=None) -> str:
    """Read the source documentation file and return it as one raw string.

    Supports PDF (via PyPDFLoader) as well as plain-text / markdown (via
    TextLoader) so the same pipeline works whether the reference is supplied
    as the original PDF export or a cleaned-up .txt file.
    """
    path = config.SOURCE_DOC_PATH if path is None else path

    if not path.exists():
        raise FileNotFoundError(
            f"Source document not found at '{path}'. "
            "Drop the file into data/ or set SOURCE_DOC_PATH in your .env."
        )

    if path.suffix.lower() == ".pdf":
        # PyPDFLoader yields one Document per page; join them into a single
        # string so chunking treats the document as one continuous text.
        pages = PyPDFLoader(str(path)).load()
        return "\n".join(page.page_content for page in pages)

    # TextLoader covers .txt and .md.
    docs = TextLoader(str(path), encoding="utf-8").load()
    return docs[0].page_content


def sanity_check(text: str, sample_chars: int = 500) -> None:
    """Print the total character count and a text sample (Assignment A1).

    A cheap but effective way to confirm the file was actually read and
    decoded correctly before we spend time embedding it.
    """
    print("=" * 64)
    print("A1 SANITY CHECK")
    print("=" * 64)
    print(f"Total characters : {len(text):,}")
    print(f"Total words (~)  : {len(text.split()):,}")
    print("-" * 64)
    print(f"First {sample_chars} characters of the document:\n")
    print(text[:sample_chars])
    print("=" * 64)


# ==========================================================================
# A2 - Document Chunking
# ==========================================================================
def chunk_text(text: str) -> list[Document]:
    """Split raw text into overlapping chunks of CHUNK_SIZE characters.

    RecursiveCharacterTextSplitter first tries to break on natural boundaries
    (paragraphs -> lines -> spaces) and only does a hard character cut as a
    last resort. That keeps code snippets and sentences as intact as possible
    while still respecting the 500-character size limit.

    The 50-character overlap means consecutive chunks share their boundary
    text, so a fact (or a code snippet) that straddles a chunk boundary still
    appears whole in at least one chunk and is not lost at retrieval time.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,        # measure size in characters, per the spec
        add_start_index=True,       # store each chunk's char offset for display
    )
    return splitter.create_documents([text])


# ==========================================================================
# A3 - Vector Storage
# ==========================================================================
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local sentence-transformers embedding model.

    Runs entirely on this machine, so the documentation text is never sent to
    any external service for embedding (Assignment Rule #1).
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def build_vector_store(chunks: list[Document]) -> FAISS:
    """Embed the chunks and persist a FAISS index to disk.

    FAISS is an in-process library (no server to run), which makes it fast,
    reproducible, and trivial to ship with the project.
    """
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(config.VECTOR_STORE_DIR))
    return store


def load_vector_store() -> FAISS:
    """Load the previously persisted FAISS index from disk."""
    if not config.VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. Build it first with: python ingest.py"
        )
    return FAISS.load_local(
        str(config.VECTOR_STORE_DIR),
        get_embeddings(),
        # Safe: we generated this index ourselves on this machine.
        allow_dangerous_deserialization=True,
    )


# ==========================================================================
# B1 - Semantic Retrieval
# ==========================================================================
def retrieve(query: str, store: FAISS = None, k: int = None) -> list[Document]:
    """Return the top-k chunks most semantically similar to the query.

    The query is embedded with the same model used for the documents, then
    FAISS finds the nearest vectors -> the most relevant chunks.
    """
    store = load_vector_store() if store is None else store
    k = config.TOP_K if k is None else k
    return store.similarity_search(query, k=k)


# ==========================================================================
# B2 - API Integration & Prompting
# ==========================================================================
# System prompt: forces the persona AND wires in the hallucination guard.
SYSTEM_PROMPT = """You are a Senior Upwork API Consultant. You help developers \
by answering their technical questions about the Upwork API clearly, precisely, \
and professionally.

Follow these rules without exception:
1. Answer ONLY using the information in the "Context" provided below. Do not \
use any prior or outside knowledge.
2. If the answer cannot be found in the Context, you MUST reply with EXACTLY \
this sentence and nothing else:
"I'm sorry, but the provided documentation does not contain that information."
3. Be concise and specific. Quote the exact values from the Context (e.g. \
token TTLs, grant types, endpoints, permissions) when they are relevant.
4. Never invent endpoints, parameters, limits, or behaviours that are not in \
the Context."""

# Two-message prompt: the system persona/guard + the user turn carrying the
# retrieved context and the actual question.
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer as the Senior Upwork API Consultant, using only the Context above.",
        ),
    ]
)


def build_llm() -> ChatOpenAI:
    """Create the OpenAI-compatible chat client from environment config.

    Works against any OpenAI-compatible endpoint (HF router, Ollama, vLLM...).
    """
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", config.LLM_BASE_URL),
            ("LLM_API_KEY", config.LLM_API_KEY),
            ("LLM_MODEL", config.LLM_MODEL),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )

    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )


def answer_query(query: str, store: FAISS = None, llm: ChatOpenAI = None) -> dict:
    """Full RAG round-trip for a single user question.

    Steps: retrieve top-k chunks -> stuff them into the prompt -> call the LLM.
    Returns the answer, the source chunks used, and the API latency.
    """
    store = load_vector_store() if store is None else store
    llm = build_llm() if llm is None else llm

    # B1: fetch the most relevant chunks.
    chunks = retrieve(query, store=store)

    # Join the chunks into a single context block, clearly separated so the
    # model can tell where one snippet ends and the next begins.
    context = "\n\n---\n\n".join(c.page_content for c in chunks)
    messages = PROMPT.format_messages(context=context, question=query)

    # Time ONLY the API round-trip, which is the "latency" the UI must show.
    start = time.perf_counter()
    response = llm.invoke(messages)
    latency = time.perf_counter() - start

    return {
        "answer": response.content,
        "sources": chunks,
        "latency": latency,
    }
