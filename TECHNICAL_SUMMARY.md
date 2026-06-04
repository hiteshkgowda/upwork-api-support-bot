# Technical Summary — Upwork API Technical Support Bot

**Author:** Hitesh K · **Stack:** Python · LangChain · FAISS · sentence-transformers · Streamlit

## What I built
A RAG bot that answers Upwork API questions strictly from the official
documentation. The pipeline loads the PDF, chunks it (500 chars / 50 overlap),
embeds the chunks locally with `all-MiniLM-L6-v2`, stores them in a FAISS index,
retrieves the top-3 chunks per query, and asks an OpenAI-compatible LLM —
prompted as a "Senior Upwork API Consultant" — to answer using only that
context, or to return a fixed refusal when the answer isn't present.

## Difficulties faced (and how I solved them)
- **PDF → clean text.** The reference is an image-heavy PDF. I used
  `PyPDFLoader` to pull the text layer and added an A1 sanity check (character
  count + sample) so I could confirm the 43k characters were read correctly
  before spending time on embeddings.
- **Chunking technical content without breaking it.** A naïve fixed-size split
  cuts `curl` examples and JSON responses mid-line. I used
  `RecursiveCharacterTextSplitter` (splits on natural boundaries first) with a
  50-char overlap so facts that straddle a boundary still appear whole in at
  least one chunk.
- **Preventing hallucinations.** The hardest part of a support bot is making it
  say *"I don't know."* I enforced this in the system prompt (answer only from
  context, else an exact refusal sentence) and set a low temperature (0.1). The
  rate-limit question — which the partial doc does **not** answer — correctly
  triggers the refusal instead of a made-up number.
- **API latency & UX.** LLM calls are the slow step, so I time only the API
  round-trip and surface it in the UI, and cache the FAISS index + model in
  Streamlit (`@st.cache_resource`) so they load once, not on every query.
- **Keeping it portable & secret-safe.** I made the LLM client
  OpenAI-compatible (works with Hugging Face, Ollama, vLLM) and read the
  endpoint/key from environment variables — nothing sensitive is hard-coded.

## How I used LLMs to assist development
I used Claude / GPT as a **pair-programmer**, not a code-dump: to compare
LangChain vs. LlamaIndex and FAISS vs. ChromaDB trade-offs, to refine the
system prompt and hallucination-guard wording, and to sanity-check my chunking
parameters and edge cases. Per the assignment's "No Data Uploads" rule, I did
**not** paste the source documentation into any public LLM — embeddings run
locally so the docs never leave my machine. I can explain every line of the
code I committed.

## 3 reasons I'm the best person for the ProAnalyst AI team
1. **I ship working, honest AI — not demos.** I verified the whole pipeline
   end-to-end and prioritized the *anti-hallucination* behaviour that actually
   makes a support bot trustworthy in production, which is exactly what an
   analytics product needs.
2. **Strong data-science fundamentals + ownership.** As a BE CSE (Data Science)
   student I understand embeddings, retrieval, and evaluation from first
   principles, and I run my own business — so I treat code, deadlines, and
   correctness as things I own, not tasks I'm handed.
3. **I learn fast and perform under pressure.** As a national-level basketball
   player I'm used to high-stakes, fast-feedback environments; I pick up new
   tools quickly (this project's stack included) and stay calm and reliable
   when the problem is hard and the clock is running.
