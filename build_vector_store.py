"""
Embedding + vector store.

Reads the chunk files produced by tag_handbook_record.py / tag_all.py, embeds every chunk
and stores it in Chroma, persisted locally under ./chroma_db.

Embedding has the same provider switch as generation: EMBEDDING_PROVIDER=local uses a
sentence-transformers model that runs on the machine, EMBEDDING_PROVIDER=openai uses the
OpenAI embedding API (needs OPENAI_API_KEY; the cost is tiny at this corpus size).

    export EMBEDDING_PROVIDER=local     # default, and how the shipped store was built
    export EMBEDDING_PROVIDER=openai    # to A/B test the OpenAI embedding API

LOCAL_EMBEDDING_MODEL selects the local model, with no code change for a different size or
family. The configuration behind the reported results is BAAI/bge-base-en-v1.5
(English-only, 109M); intfloat/multilingual-e5-base (278M) is the model it was compared
against in benchmark/results/20261001_embed_bge_vs_e5. Two things have to hold for either
to perform as documented:

    1. Vectors are normalised, so similarity is effectively cosine (see embed_texts).
    2. The retrieval prefix each family expects is applied -- bge on the query side only,
       e5 on both sides. The query side lives in rag_query.py; the passage side is
       resolve_passage_instruction() below.

build_vector_store.py and rag_query.py must run with the same EMBEDDING_PROVIDER and
model: questions and documents embedded by different models land in different vector
spaces, which makes the distances meaningless. This script records the provider and model
in the collection metadata, and rag_query.py checks that at run time and fails on a
mismatch.

Usage:
    python build_vector_store.py tagged_data/32555_chunks.json [tagged_data/other_chunks.json ...]
"""

import json
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()  # picks up OPENAI_API_KEY, EMBEDDING_PROVIDER and other environment variables from .env

# Next to this file, not next to the shell's working directory.
DB_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "rag_mvp"

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local").lower()  # "local" | "openai"
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# e5 models (intfloat/*-e5-*) need a "passage: " prefix on documents, as their model card
# recommends; bge models get nothing on the document side. This has to stay paired with
# BGE_QUERY_INSTRUCTION / E5_QUERY_INSTRUCTION in rag_query.py -- prefixing one side only
# pulls the question and document vector spaces apart.
E5_PASSAGE_INSTRUCTION = "passage: "


def resolve_passage_instruction() -> str:
    """
    Pick the prefix for document text.
    - LOCAL_PASSAGE_INSTRUCTION wins if set, including when set to an empty string.
    - Otherwise it comes from the model name: only e5 needs a passage prefix, bge does not.
    """
    override = os.environ.get("LOCAL_PASSAGE_INSTRUCTION")
    if override is not None:
        return override
    if "e5" in LOCAL_EMBEDDING_MODEL.lower():
        return E5_PASSAGE_INSTRUCTION
    return ""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed with the local model or the OpenAI API, per EMBEDDING_PROVIDER."""
    if EMBEDDING_PROVIDER == "local":
        from sentence_transformers import SentenceTransformer  # lazy import

        print(f"Loading local embedding model ({LOCAL_EMBEDDING_MODEL})... "
              f"(first run downloads the model, requires internet)")
        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        instruction = resolve_passage_instruction()
        if instruction:
            print(f"  (automatically applying the '{instruction.strip()}' prefix to documents)")
            texts = [instruction + t for t in texts]
        # normalize_embeddings=True turns the vectors into unit vectors, which makes the
        # comparison effectively cosine similarity. bge and e5 are both trained and
        # evaluated that way and their model cards recommend it, so this is what it takes
        # to get the documented performance. It also fixes the distance scale: normalised
        # L2 distances fall in 0-2, and runs made without normalising are not comparable.
        # Either way, compare ranks within one run and one question, never absolute
        # distances across runs or models.
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    elif EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI  # lazy import

        print(f"Calling OpenAI embeddings API ({OPENAI_EMBEDDING_MODEL})...")
        client = OpenAI()  # reads OPENAI_API_KEY from env
        # Batch by 100 to keep requests small; unnecessary at this size, but it keeps
        # working as the corpus grows.
        embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
            embeddings.extend([d.embedding for d in resp.data])
        return embeddings

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}'. Only 'local' or 'openai' are supported."
        )


def embedding_model_name() -> str:
    return LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "local" else OPENAI_EMBEDDING_MODEL


def load_chunks(paths: list[str]) -> list[dict]:
    chunks = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        chunks.extend(data)
    return chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_vector_store.py <chunks.json> [more_chunks.json ...]")
        sys.exit(1)

    chunk_paths = sys.argv[1:]
    chunks = load_chunks(chunk_paths)
    print(f"Loaded {len(chunks)} chunks from {len(chunk_paths)} file(s)")

    client = chromadb.PersistentClient(path=DB_DIR)
    # Reset the collection each run so re-running this script doesn't
    # duplicate chunks -- convenient while iterating during MVP debugging.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Record the provider and model on the collection so rag_query.py can catch a query
    # embedded with a different one.
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": embedding_model_name(),
        },
    )

    texts = [c["content"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    # Chroma metadata values must be str/int/float/bool (no None, no nested
    # objects) -- filter/convert accordingly.
    metadatas = []
    for c in chunks:
        meta = {}
        for k, v in c.items():
            if k in ("content",):
                continue
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = json.dumps(v, ensure_ascii=False)
        metadatas.append(meta)

    print(f"Embedding {len(texts)} chunks (provider={EMBEDDING_PROVIDER})...")
    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in Chroma collection '{COLLECTION_NAME}' at {DB_DIR}/ "
          f"(embedding_provider={EMBEDDING_PROVIDER}, model={embedding_model_name()})")


if __name__ == "__main__":
    main()