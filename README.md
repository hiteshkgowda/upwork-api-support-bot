# Upwork API — Technical Support AI Bot (RAG)

A Retrieval-Augmented Generation (RAG) assistant that answers developer
questions about the **Upwork API**, using *only* the official Technical
Reference as its source of truth. It retrieves the most relevant documentation
snippets, feeds them to an LLM acting as a *"Senior Upwork API Consultant,"*
and refuses to answer anything the documentation does not cover (no
hallucinations).

---

## How it maps to the assignment

| Stage | Where | What it does |
|-------|-------|--------------|
| **A1** Data ingestion + sanity check | `rag.load_document`, `rag.sanity_check` | Reads the PDF, prints char count + a sample |
| **A2** Chunking (500 / 50) | `rag.chunk_text` | Splits into 500-char chunks with 50-char overlap |
| **A3** Vector storage | `rag.build_vector_store` | Local `all-MiniLM-L6-v2` embeddings → **FAISS** index on disk |
| **B1** Semantic retrieval | `rag.retrieve` | Fetches the **top-3** most relevant chunks |
| **B2** API integration + prompting | `rag.answer_query`, `rag.SYSTEM_PROMPT` | Persona + **hallucination guard**, OpenAI-compatible LLM call |
| **B3** UI | `app.py` | Streamlit: answer · sources · latency |

### Project structure
```
.
├── config.py          # all settings + secrets, read from environment
├── rag.py             # the RAG pipeline (A1–B2)
├── ingest.py          # one-off: build the vector store (run this first)
├── app.py             # Streamlit UI (B3)
├── requirements.txt
├── .env.example       # template for your endpoint + key
├── TECHNICAL_SUMMARY.md
└── data/
    └── upwork_api_reference.pdf   # the source documentation
```

---

## Setup & run

```bash
# 1. Create a virtual environment (Python 3.12 recommended) and install deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure your LLM endpoint + key (never hard-coded)
cp .env.example .env
#   then edit .env and set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

# 3. Build the vector store from the documentation (runs the A1–A3 stages)
python ingest.py

# 4. Launch the UI
streamlit run app.py
```

The LLM client is **OpenAI-compatible**, so the same code works against the
Hugging Face router, a local/remote **Ollama** server, vLLM, etc. — just point
`LLM_BASE_URL` at your endpoint. (This project was verified end-to-end against a
local Ollama `llama3` using `LLM_BASE_URL=http://localhost:11434/v1`.)

---

## A2 — Why is chunk *overlap* important for technical content?

Overlap means consecutive chunks share a few characters at their boundary
(here, 50). It matters for technical documentation because:

- **Facts and code snippets straddle boundaries.** A single sentence like
  *"TTL for an access token is 24 hours; TTL for a refresh token is 2 weeks"*
  or a multi-line `curl` example can fall exactly on a chunk cut. Without
  overlap, half the fact lands in chunk *N* and half in chunk *N+1*, and
  neither chunk alone retrieves as relevant — so the model never sees the
  complete answer.
- **It preserves local context.** Code depends on the line above it (a header,
  a parameter name, a `grant_type`). Overlap keeps that surrounding context
  attached so a retrieved snippet is self-explanatory.
- **It improves retrieval recall.** Because the boundary text appears in two
  chunks, a query that matches near a cut still finds a chunk that contains the
  full relevant passage.

---

## Part C — Evaluation results

Verified against the bundled documentation (local Ollama `llama3`):

1. **"What is the request-per-second rate limit, and is it per Key or per IP?"**
   → *"I'm sorry, but the provided documentation does not contain that
   information."*
   The supplied reference is the **partial** doc and contains no explicit
   per-second rate limit (only guidance to cache responses), so the
   **hallucination guard correctly fires** instead of inventing a number.
2. **"How long is an OAuth access token valid for?"**
   → **24 hours** — matches the doc (*"TTL for an access token is 24 hours"*,
   `expires_in: 86400`).
3. **"Can I use a Client Credentials Grant to access a user's private contract
   details?"**
   → **No** — the grant is for *enterprise accounts only*, used in
   *server-to-server* scenarios *outside the context of a user*, so it is not
   the mechanism for reading a specific user's private data.

---

## Deploying to Streamlit Community Cloud

The app is cloud-ready: it **builds the FAISS index on first boot** from the
bundled PDF (no separate `ingest.py` run needed) and reads the LLM endpoint/key
from the app's **Secrets**.

1. Push this repo to GitHub (already done if you're reading it there).
2. Go to **https://share.streamlit.io** → *Create app* → pick this repo,
   branch `main`, main file `app.py`.
3. Under **Advanced settings**, set **Python version = 3.12**.
4. Open **Secrets** and paste your endpoint config (TOML):
   ```toml
   LLM_BASE_URL = "https://your-endpoint/v1"
   LLM_API_KEY  = "your_api_key"
   LLM_MODEL    = "your_model_name"
   LLM_TEMPERATURE = "0.1"
   ```
5. **Deploy.** Streamlit installs `requirements.txt`, builds the index, and
   serves the app.

> The endpoint must be **internet-reachable** — a local Ollama
> (`localhost:11434`) works for local runs but not from the cloud. Until a
> reachable endpoint is set in Secrets, the deployed app runs as a retrieval
> demo (it shows the source snippets) and enables generated answers as soon as
> the endpoint is added.

## Security notes (assignment rules)

- **No hard-coded secrets** — the endpoint and key are read from environment
  variables via `.env`, which is git-ignored. `.env.example` documents them.
- **No data uploads** — embeddings are computed by a **local** model, so the
  documentation text never leaves the machine. Only the small set of retrieved
  snippets plus the question are sent to the configured LLM endpoint at query
  time, which is required for RAG.
