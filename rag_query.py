"""
Retrieval + generation + citation.

Finds the top-k chunks closest to a question in the Chroma store built by
build_vector_store.py, has an LLM answer from those chunks only, and lists the sources it
used. The system prompt also requires it to say so when the chunks do not answer the
question (abstention).

Generation goes through OpenAI. The LLM_PROVIDER switch is kept, but openai is the only
implementation: adding another means one more function next to _generate_openai() and a
branch in generate_answer().

Embedding has the same kind of switch, EMBEDDING_PROVIDER, and it must match the value
used by build_vector_store.py. If they differ, question vectors and document vectors live
in different spaces and retrieval is meaningless, so this script compares its settings
against the provider/model stored in the collection and fails immediately on a mismatch.

    export EMBEDDING_PROVIDER=local    # default, and how the shipped store was built
    export EMBEDDING_PROVIDER=openai   # to A/B test the OpenAI embedding API

Changing LOCAL_EMBEDDING_MODEL also changes the retrieval prefix, which is picked from the
model name. This is nothing but a fixed string prepended to the text, following each model
card's recommendation:

    - BAAI/bge-* : an instruction on the query side only; documents get nothing, so
      build_vector_store.py is left alone.
    - intfloat/*-e5-* : "query: " on questions and "passage: " on documents, the latter
      handled by resolve_passage_instruction() in build_vector_store.py.

Prefix one side only and the two vector spaces drift apart, so always change both files
together. Model size needs no code change -- the switch handles small, base and large --
and LOCAL_QUERY_INSTRUCTION overrides the choice (set it empty to force no prefix).

    export LOCAL_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"           # default, English, 109M
    export LOCAL_EMBEDDING_MODEL="intfloat/multilingual-e5-base"    # the e5 comparison, 278M
    export LOCAL_QUERY_INSTRUCTION=""                               # force no prefix

Needed for generation:
    pip install openai
    export OPENAI_API_KEY="sk-..."     # https://platform.openai.com/api-keys

Everything sent to the model -- SYSTEM_PROMPT, the in-prompt labels such as
[Reference material] and [Question], and the notices printed to the terminal -- is in
English, because the embedding model (bge-base-en-v1.5) is English-only.

Usage:
    python rag_query.py "How many credit points is Fundamentals of Software Development, and what does it cover?"
    python rag_query.py "What is the assessment structure for this subject?" --top-k 3
"""

import argparse
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv

# Load .env into os.environ if it exists; without it this is a no-op, so plain exports
# work just as well.
load_dotenv()

# Next to this file, not next to the shell's working directory.
DB_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "rag_mvp"

# Default for --top-k, measured rather than chosen: over k5 vs k8 (3 runs each) recall was
# 0.60 vs 0.93 and fully correct answers 3/10 vs 7/10, with no regression at k=8.
DEFAULT_TOP_K = 8

# Course-enrolment scoping: chunks from a subject the student is enrolled in can get a
# distance bonus, because in a 96-subject corpus unrelated subjects push gold out of top-k.
# A soft boost, not a hard filter, so questions about subjects the student is not taking
# stay answerable from the Handbook.
# Default 0: at top_k=8 recall is already 0.91 without it and the boost changes nothing.
# It only matters at top_k=5, where 0.17 is the measured value: the safe range is
# 0.16-0.18, and at 0.19 and above questions about unenrolled subjects start failing.
BOOST_ENROLLED = float(os.getenv("BOOST_ENROLLED", "0"))
# The boost only reorders what was already fetched, so lifting a chunk that sat below
# top_k means fetching more than top_k, re-ranking, then cutting.
OVERFETCH = 4

# Which provider to use, and each provider's default model. Both can be overridden by env var.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()  # only "openai" is implemented
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

API_KEY_ENV_VAR = {"openai": "OPENAI_API_KEY"}

# Must match build_vector_store.py. If they differ, question and document vectors sit in
# different embedding spaces and retrieval is meaningless, so check_embedding_config()
# compares these against the collection metadata at run time and raises on a mismatch.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local").lower()  # "local" | "openai"
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# bge models (BAAI/bge-*) need this instruction on the query side only, as their model
# card recommends; documents get no prefix, which is why embed_texts() in
# build_vector_store.py is left alone.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# e5 models (intfloat/*-e5-*) need "query: " on questions; the matching "passage: " prefix
# for documents is E5_PASSAGE_INSTRUCTION in build_vector_store.py. The two always go
# together.
E5_QUERY_INSTRUCTION = "query: "

SYSTEM_PROMPT = """\
You are an assistant that answers UTS students' questions about subject information.
You must follow the rules below.

1. Base your answer only on the content provided in [Reference material]. Do not fill gaps with your own prior knowledge.
2. If [Reference material] does not contain enough information to answer the question -- or one part of a \
multi-part question -- clearly say so for that part specifically and explain what information is missing. If the \
question has multiple parts, judge each part independently: never use information that answers one part (e.g. an \
individual assessment task's points or weight) to answer a different, unrelated part (e.g. the subject's overall \
credit points) just because the numbers happen to look similar. Material that is on the same topic as the question \
but does not actually answer what is being asked does not count as enough information -- in that case say the \
answer is not available rather than substituting general advice or related facts. Never make anything up (abstention).
3. At the end of your answer, list the sources you actually used under a "Sources:" heading. \
Format each entry as "Subject name (subject_code) - section - URL", and use the exact source_url value given in the \
reference material for the URL (do not invent one). If the URL starts with synthetic://, note in parentheses that it \
is an internal path for synthetic research data, not a real UTS Canvas link. Do not cite material you didn't use.
4. If information comes from multiple sources, distinguish which content came from which source.
5. Treat two sources as being in conflict only when they state different facts about the same specific item \
(e.g. different due dates, or a different weight, for the same assessment task). If one source is simply more \
detailed than the other without contradicting it, do not describe this as a discrepancy -- just present the most \
specific, relevant information for the question. If sources genuinely conflict, prioritize whichever is more \
appropriate given the type of question, and briefly explain why.
6. Answer only what was asked. Do not add commentary about the structure, format, or level of detail of the \
source material unless it is necessary to answer the question or to explain a genuine conflict (per rule 5).
7. Distinguish "credit points" (the fixed value assigned to a whole subject, e.g. "6 credit points") from the \
"points" or "weight" assigned to an individual assessment task (e.g. "worth 15 points", "15%"). These are different \
things -- never use one to answer a question about the other.
8. If the question is about a specific subject (by code or name) and none of the [Reference material] is about \
that subject, say explicitly that no subject with that code/name was found in the available sources -- do not \
only say that the requested detail (e.g. its prerequisites) is missing. Do not claim the subject does not exist \
at UTS, and do not answer with material about a different subject.
"""


@lru_cache(maxsize=1)
def enrolled_subjects() -> dict[str, str]:
    """
    The subjects the student is enrolled in, as {code: name}. Derived from the corpus
    rather than a config file: having Canvas data for a subject means the student has
    access to it, i.e. is enrolled. Adding Canvas chunks extends the enrolment by itself.
    """
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION_NAME)
    got = collection.get(where={"source_type": "canvas"}, include=["metadatas"])
    return {m["subject_code"]: m["subject_name"] for m in got["metadatas"] if m.get("subject_code")}


def check_embedding_config(collection) -> None:
    """
    Check this script's settings against the embedding_provider/model that
    build_vector_store.py recorded on the collection. On a mismatch the distances would be
    quietly meaningless, so raise instead and name the cause.
    """
    stored = collection.metadata or {}
    stored_provider = stored.get("embedding_provider")
    stored_model = stored.get("embedding_model")
    current_model = LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "local" else OPENAI_EMBEDDING_MODEL

    if stored_provider is None:
        return  # collection built before this metadata existed; nothing to check against

    if stored_provider != EMBEDDING_PROVIDER or stored_model != current_model:
        raise RuntimeError(
            "Embedding config mismatch: this Chroma collection was built with "
            f"provider={stored_provider}/model={stored_model}, but the current "
            f"rag_query.py settings are provider={EMBEDDING_PROVIDER}/model={current_model}. "
            "Retrieval is meaningless if these differ -- either set EMBEDDING_PROVIDER "
            "to match, or re-run build_vector_store.py with the current settings to "
            "rebuild the collection."
        )


def resolve_query_instruction() -> str:
    """
    Pick the prefix for the question embedding.
    - LOCAL_QUERY_INSTRUCTION wins if set, including when set to an empty string.
    - Otherwise it comes from the model name: the bge instruction for "bge", "query: " for
      "e5", no prefix for anything else.
    - Never called for the openai provider; embed_query() only uses it on the local path.
    """
    override = os.environ.get("LOCAL_QUERY_INSTRUCTION")
    if override is not None:
        return override
    model_lower = LOCAL_EMBEDDING_MODEL.lower()
    if "bge" in model_lower:
        return BGE_QUERY_INSTRUCTION
    if "e5" in model_lower:
        return E5_QUERY_INSTRUCTION
    return ""


def embed_query(question: str) -> list[float]:
    if EMBEDDING_PROVIDER == "local":
        from sentence_transformers import SentenceTransformer  # lazy import

        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        instruction = resolve_query_instruction()
        text = instruction + question
        # normalize_embeddings=True must match build_vector_store.py; normalising one side
        # only would put questions and documents in different spaces. The bge model card
        # recommends it explicitly, and it is harmless for other sentence-transformers
        # models.
        return model.encode([text], normalize_embeddings=True).tolist()[0]

    elif EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI  # lazy import

        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[question])
        return resp.data[0].embedding

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}'. Only 'local' or 'openai' are supported."
        )


def retrieve(question: str, top_k: int, filter_source_type: str | None = None):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    check_embedding_config(collection)

    query_embedding = [embed_query(question)]
    where = {"source_type": filter_source_type} if filter_source_type else None
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k * OVERFETCH if BOOST_ENROLLED else top_k,
        where=where,
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({"content": doc, "metadata": meta, "distance": dist})

    # Re-ranking belongs here, inside one query's results. Doing it after the merge in
    # retrieve_multi() would compare absolute distances across sub-queries, which is the
    # mistake that function's docstring warns about.
    if BOOST_ENROLLED:
        enrolled = enrolled_subjects()
        retrieved.sort(
            key=lambda r: r["distance"]
            - (BOOST_ENROLLED if r["metadata"].get("subject_code") in enrolled else 0)
        )
    return retrieved[:top_k]


# A compound question ("...the due dates, as well as the credit points") embedded as one
# vector can lose one of its intents: measured here, the chunk answering "credit points"
# fell past top-8. Raising top-k is no answer, since it stops working as the corpus grows,
# so the question is split into sub-queries that are retrieved separately and merged --
# multi-query retrieval, or query decomposition, a standard technique in the RAG
# literature and the idea behind LangChain's MultiQueryRetriever.
#
# A student has no reason to decide whether their question needs splitting, so main()
# always does it. split_into_sub_queries() returns a one-item list when there is nothing
# to split, which makes single-intent questions behave exactly as before; --single-query
# turns the splitting off for comparison and debugging.
SUB_QUERY_SPLIT_PATTERN = re.compile(
    r"\s*(?:,\s*as well as\s*|,?\s*along with\s*|,?\s*in addition to\s*|\?\s*(?:and|also)\s*)\s*",
    re.IGNORECASE,
)


def split_into_sub_queries(question: str) -> list[str]:
    """
    Split a question at linking phrases such as "as well as", "along with" and "in
    addition to". This is a heuristic over a few common patterns, not real language
    analysis, so a compound question joined by a plain "and" still goes through as one
    query.

    Known limitation: a pronoun in the trailing clause ("this course", "it") loses the
    subject named only in the first clause, which can send that sub-query somewhere
    unrelated. Measured case: "...in Fundamentals of Software Development, as well as the
    credit points for this course?" splits off "the credit points for this course", whose
    distances then run around 0.9. Prepending the first clause to the second was tried
    both in full and truncated, and both were rolled back -- the full clause dominates the
    sub-query's meaning again, and a truncation length is a magic number that does not
    carry to other questions. The coreference problem is therefore left as a known
    limitation; fixing it properly means having an LLM rewrite the question into
    self-contained sub-questions (query rewriting) rather than a regex.
    """
    parts = [p.strip().rstrip("?").strip() for p in SUB_QUERY_SPLIT_PATTERN.split(question)]
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return [question]
    return parts


def retrieve_multi(question: str, top_k: int, filter_source_type: str | None = None):
    """
    Split the question, retrieve top_k for each sub-query and merge the results. With a
    single sub-query this is just retrieve() on the original question.

    The merge must not sort by distance. A short, generic sub-query such as "credit points
    for this course" sits further from every document, so re-sorting the combined results
    by absolute distance dropped that sub-query's chunks wholesale (measured
    2026-09-03). It is the same mistake as comparing distances across different questions,
    made at the merge step. So the merge trusts only each sub-query's internal ranking and
    interleaves round-robin: all the firsts, then all the seconds, and so on. No
    sub-query is excluded just for having a wider distance scale than the others.

    Returns (merged_top_k, sub_queries, per_query_results), where per_query_results holds
    each sub-query's own top_k in the same order as sub_queries. main() prints it, which
    is what makes a merged result explainable per sub-query.
    """
    sub_queries = split_into_sub_queries(question)
    if len(sub_queries) == 1:
        results = retrieve(sub_queries[0], top_k, filter_source_type)
        return results, sub_queries, [results]

    per_query_results = [retrieve(sq, top_k, filter_source_type) for sq in sub_queries]

    merged: dict[str, dict] = {}
    order: list[str] = []
    for rank in range(top_k):
        for results in per_query_results:
            if rank < len(results):
                r = results[rank]
                cid = r["metadata"].get("chunk_id")
                if cid not in merged:
                    merged[cid] = r
                    order.append(cid)
        if len(order) >= top_k:
            break

    ranked = [merged[cid] for cid in order[:top_k]]
    return ranked, sub_queries, per_query_results


def build_context(retrieved: list[dict]) -> str:
    # The chunk schema (tag_handbook_record.py) has no authority_level field: the team
    # agreed to derive official vs operational from source_type by rule instead. That
    # derivation happens here, on the way into the prompt.
    authority_by_source_type = {"handbook": "official", "canvas": "operational"}

    # The labels below ("[Reference material N]" and so on) are prompt content that
    # SYSTEM_PROMPT refers to by name, so rules 1-3 only work while they match exactly.
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        meta = r["metadata"]
        source_type = meta.get("source_type")
        authority = authority_by_source_type.get(source_type, "unknown")
        blocks.append(
            f"[Reference material {i}] (subject_code={meta.get('subject_code')}, "
            f"subject_name={meta.get('subject_name')}, section={meta.get('section')}, "
            f"source_type={source_type}, authority_level={authority}, "
            f"is_synthetic={meta.get('is_synthetic')}, "
            f"source_url={meta.get('source_url')})\n{r['content']}"
        )
    return "\n\n".join(blocks)


def _generate_openai(user_message: str) -> str:
    from openai import OpenAI  # lazy import, so --show-retrieval-only runs without it installed

    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1024,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def generate_answer(question: str, retrieved: list[dict]) -> str:
    context = build_context(retrieved)
    # The [Reference material] and [Question] labels are prompt content paired with
    # SYSTEM_PROMPT, as in build_context() above.
    # Enrolment is input about who is asking, not something to retrieve, so it goes in on
    # every question. As a chunk it would only reach the model when it made top-k.
    enrolment = "\n".join(f"- {code} {name}" for code, name in sorted(enrolled_subjects().items()))
    user_message = (
        f"[Student enrolment]\nThe student asking this question is currently enrolled in:\n{enrolment}\n\n"
        f"[Reference material]\n{context}\n\n[Question]\n{question}"
    )

    if LLM_PROVIDER == "openai":
        return _generate_openai(user_message)
    raise ValueError(
        f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Only 'openai' is supported."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--source-type", choices=["handbook", "canvas"], default=None,
        help="Only search a specific source (for debugging)",
    )
    parser.add_argument(
        "--show-retrieval-only", action="store_true",
        help="Only show retrieval results without calling the LLM (works without an API key)",
    )
    parser.add_argument(
        "--single-query", action="store_true",
        help="Debug/comparison only: disable automatic multi-query splitting and embed the question "
             "as a single vector, the way this script used to work (see retrieve_multi's docstring)",
    )
    args = parser.parse_args()

    # Multi-query is the default and needs no thought from the student: a question with
    # something to split gets split, and a single-intent one behaves exactly as it would
    # without splitting. --single-query is for comparison and debugging.
    if args.single_query:
        retrieved = retrieve(args.question, args.top_k, args.source_type)
    else:
        retrieved, sub_queries, per_query_results = retrieve_multi(args.question, args.top_k, args.source_type)
        if len(sub_queries) > 1:
            print(f"\n[Multi-query] split into {len(sub_queries)} sub-queries:")
            # Print each sub-query's own top-k. The merged list alone does not say which
            # sub-query found what, which once nearly hid a bug in the merge.
            for sq, results in zip(sub_queries, per_query_results):
                print(f"  - {sq!r}")
                for r in results:
                    m = r["metadata"]
                    print(f"      {m.get('chunk_id')}  [{m.get('source_type')}/{m.get('section')}]  dist={r['distance']:.3f}")

    embedding_model = LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "local" else OPENAI_EMBEDDING_MODEL
    instruction_note = ""
    if EMBEDDING_PROVIDER == "local":
        instr = resolve_query_instruction()
        if instr:
            # Print the prefix that was actually applied, so a pasted run log says which
            # one it was.
            instruction_note = f" [query prefix: {instr.strip()!r}]"
    print(f"\n=== Retrieved top-{len(retrieved)} chunks "
          f"(embedding={EMBEDDING_PROVIDER}/{embedding_model}){instruction_note} ===")
    for r in retrieved:
        m = r["metadata"]
        print(f"  {m.get('chunk_id')}  [{m.get('source_type')}/{m.get('section')}]  dist={r['distance']:.3f}")

    if args.show_retrieval_only:
        return

    key_env_var = API_KEY_ENV_VAR.get(LLM_PROVIDER)
    if key_env_var is None or not os.environ.get(key_env_var):
        print(f"\n[Notice] LLM_PROVIDER={LLM_PROVIDER} but the {key_env_var} environment variable is not set, "
              f"so answer generation is skipped.")
        print(f"Set the key with `export {key_env_var}=...` and re-run.")
        sys.exit(0)

    print(f"\n=== Generated answer (provider={LLM_PROVIDER}, model={OPENAI_MODEL}) ===")
    answer = generate_answer(args.question, retrieved)
    print(answer)


if __name__ == "__main__":
    main()