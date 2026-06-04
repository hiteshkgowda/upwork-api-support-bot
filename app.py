"""Streamlit UI for the Upwork API Technical Support Bot (Assignment B3).

Displays, for every question:
    1. The AI's generated answer.
    2. A "Sources" section with the exact snippets used.
    3. The latency (seconds the LLM API took to respond).

Run locally:  streamlit run app.py   (after `python ingest.py`)
On Streamlit Community Cloud the vector store is built automatically on first
boot, and the LLM endpoint/key are read from the app's Secrets.
"""
import os

import streamlit as st

# Bridge Streamlit Cloud "Secrets" into environment variables BEFORE importing
# config, so the same os.getenv-based configuration works locally (via .env)
# and in the cloud (via the Secrets manager). Wrapped in try/except because
# st.secrets raises if no secrets are configured (e.g. plain local runs).
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # noqa: BLE001 - absence of secrets is expected locally
    pass

import config  # noqa: E402 - must come after the secrets->env bridge above
import rag  # noqa: E402

st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading embedding model, vector store and LLM...")
def get_resources():
    """Load the vector store and LLM once and reuse them across reruns.

    `@st.cache_resource` keeps these heavy objects in memory so we don't reload
    the embedding model / FAISS index on every interaction.

    If the FAISS index doesn't exist yet (e.g. a fresh Streamlit Cloud
    container), build it on the fly from the source document so the app is
    self-bootstrapping and needs no separate `ingest.py` run in the cloud.
    """
    if not config.VECTOR_STORE_DIR.exists():
        text = rag.load_document()
        rag.build_vector_store(rag.chunk_text(text))
    store = rag.load_vector_store()

    # The LLM is optional: if no endpoint is configured the app still works as
    # a retrieval demo (shows the source snippets), and starts generating
    # answers the moment a reachable endpoint + key are added to Secrets.
    try:
        llm = rag.build_llm()
        llm_error = None
    except Exception as exc:  # noqa: BLE001 - missing endpoint is a valid state
        llm, llm_error = None, str(exc)
    return store, llm, llm_error


st.title("🤖 Upwork API Technical Support Bot")
st.caption(
    "A RAG assistant that answers strictly from the Upwork API Technical "
    "Reference — no hallucinations."
)

# --- Sidebar: show the active configuration (nothing secret) --------------
with st.sidebar:
    st.header("Configuration")
    st.markdown(f"**Embedding model**\n\n`{config.EMBEDDING_MODEL}`")
    st.markdown(f"**LLM model**\n\n`{config.LLM_MODEL or 'not set'}`")
    st.markdown(f"**Chunk size / overlap:** {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}")
    st.markdown(f"**Top-K retrieved:** {config.TOP_K}")
    st.divider()
    st.caption(
        "The documentation is embedded locally; only the retrieved snippets "
        "and your question are sent to the LLM endpoint."
    )

# Boot the vector store; surface only a true startup failure (e.g. missing doc).
try:
    store, llm, llm_error = get_resources()
except Exception as exc:  # noqa: BLE001 - surface any startup issue to the user
    st.error(f"Startup error: {exc}")
    st.info("Is the source document present in `data/`?")
    st.stop()

# A missing LLM endpoint is not fatal: warn, but keep the retrieval demo usable.
if llm is None:
    st.warning(
        "⚠️ LLM endpoint not configured, so generated answers are disabled. "
        "Add `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in the app's Secrets "
        "to enable them. Retrieval and the source snippets below still work."
    )

# --- Example questions (the Part C ground-truth questions) ----------------
EXAMPLES = [
    "What is the request-per-second rate limit, and is it per Key or per IP?",
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract details?",
]

st.write("**Try one of the evaluation questions:**")
example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state["query"] = example

query = st.text_input(
    "Ask a question about the Upwork API:",
    key="query",
    placeholder="e.g. How long is an OAuth access token valid for?",
)

if st.button("Ask", type="primary") and query.strip():
    if llm is not None:
        # Full RAG: retrieve + generate the consultant's answer.
        with st.spinner("Consulting the documentation..."):
            result = rag.answer_query(query, store=store, llm=llm)
        sources = result["sources"]

        # 1. The AI's generated answer.
        st.subheader("Answer")
        st.write(result["answer"])

        # 3. Latency (how many seconds the API took to respond).
        st.metric("API latency", f"{result['latency']:.2f} s")
    else:
        # No LLM endpoint: still show what the bot *would* send as context.
        with st.spinner("Retrieving relevant documentation..."):
            sources = rag.retrieve(query, store=store)
        st.info(
            "Showing retrieved sources only — configure an LLM endpoint in "
            "Secrets to get a generated answer."
        )

    # 2. Sources: the exact snippets retrieved (and sent to the model).
    st.subheader("Sources")
    st.caption("Exact snippets retrieved from the documentation and sent to the model.")
    for i, doc in enumerate(sources, start=1):
        offset = doc.metadata.get("start_index", "?")
        with st.expander(f"Source {i}  ·  char offset {offset}"):
            st.text(doc.page_content)
