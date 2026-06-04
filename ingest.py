"""One-off ingestion script: build the local vector store from the source doc.

Run once before launching the UI:

    python ingest.py

It walks the Knowledge-Engineering stages in order:
    A1  load + sanity check
    A2  chunk (500 chars / 50 overlap)
    A3  embed locally + persist a FAISS index
"""
import config
import rag


def main() -> None:
    # --- A1: load + sanity check ------------------------------------------
    print(f"Loading source document: {config.SOURCE_DOC_PATH}\n")
    text = rag.load_document()
    rag.sanity_check(text)

    # --- A2: chunk --------------------------------------------------------
    chunks = rag.chunk_text(text)
    print(
        f"\nA2 CHUNKING -> {len(chunks)} chunks "
        f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})"
    )
    print("-" * 64)
    print("Sample chunk [0]:\n")
    print(chunks[0].page_content)
    print("-" * 64)

    # --- A3: embed + persist ---------------------------------------------
    print(
        f"\nA3 VECTOR STORAGE -> embedding with '{config.EMBEDDING_MODEL}' "
        "and building FAISS index..."
    )
    rag.build_vector_store(chunks)
    print(f"Saved vector store to: {config.VECTOR_STORE_DIR}")
    print("\nDone. Launch the UI with:  streamlit run app.py")


if __name__ == "__main__":
    main()
