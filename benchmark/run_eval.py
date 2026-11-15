"""
Benchmark runner: runs the 15 questions once per config and writes the answers and
retrieval results to JSONL.

Configs come from the command line rather than being hard-coded. A key is either `top_k`
or an UPPERCASE module setting in rag_query.py (BOOST_ENROLLED, OPENAI_MODEL,
COLLECTION_NAME, ...).

  python3 benchmark/run_eval.py --runs 3 \
      --config k5 top_k=5 \
      --config k8 top_k=8 \
      --config k5_boost top_k=5 BOOST_ENROLLED=0.17

  # quick check: a few questions, one run
  python3 benchmark/run_eval.py --runs 1 --ids 10 15 --config k5 top_k=5

Retrieval is deterministic, so it runs once per (config, question); generation runs
--runs times, because gpt-4o-mini output still varies between runs at temperature 0
(measured 2026-09-27: Q10 gave 5/8 on one run and 6/8 on another).

Output (benchmark/results/<timestamp>/):
  answers.jsonl  one line per (config, question, run), including the RAGAS input fields
                 (question / answer / contexts / reference).
  summary.json   mean recall per config, recall per question, and the number of distinct
                 answers per question.
Faithfulness, abstention and citation scoring are the next step and read this file.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rag_query as rq  # noqa: E402


def norm(answer: str) -> str:
    """Normalisation used when counting distinct answers: case, whitespace and punctuation
    differences count as the same answer. ("Sources: None." == "Sources: None" -- without
    this, Q15 was flagged as unstable on 2026-09-30.)"""
    return " ".join(re.findall(r"[a-z0-9]+", answer.lower()))


def parse_config(tokens: list[str]) -> tuple[str, dict]:
    """['k8', 'top_k=8', 'BOOST_ENROLLED=0.17'] -> ('k8', {'top_k': 8, 'BOOST_ENROLLED': 0.17})"""
    name, *pairs = tokens
    values = {"top_k": rq.DEFAULT_TOP_K}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--config {name}: '{pair}' is not key=value")
        if key == "top_k":
            values[key] = int(raw)
        elif key.isupper() and hasattr(rq, key):
            # Cast to the existing value's type, so a float setting never receives "0.17"
            values[key] = type(getattr(rq, key))(raw)
        else:
            raise SystemExit(f"--config {name}: unknown setting '{key}' (top_k or an UPPERCASE name in rag_query.py)")
    return name, values


def apply(values: dict) -> dict:
    """Apply module settings to rag_query and return the previous values, so they can be
    restored between configs."""
    old = {k: getattr(rq, k) for k in values if k != "top_k"}
    for k, v in values.items():
        if k != "top_k":
            setattr(rq, k, v)
    rq.enrolled_subjects.cache_clear()  # a different COLLECTION_NAME/DB_DIR means a different enrolment
    return old


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", action="append", required=True,
                   metavar="NAME [KEY=VALUE ...]", help="repeatable")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--ids", type=int, nargs="+", help="only these question ids")
    p.add_argument("--questions", default=str(ROOT / "benchmark/questions.json"))
    p.add_argument("--out", help="output dir (default benchmark/results/<timestamp>)")
    args = p.parse_args()

    configs = [parse_config(c) for c in args.config]
    questions = json.load(open(args.questions))["questions"]
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]

    out = Path(args.out or ROOT / "benchmark/results" / datetime.now().strftime("%Y%m%d-%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    prompt_sha = hashlib.sha256(rq.SYSTEM_PROMPT.encode()).hexdigest()[:8]
    summary = {"prompt_sha": prompt_sha, "runs": args.runs, "configs": {}}

    with open(out / "answers.jsonl", "w") as f:
        for name, values in configs:
            old = apply(values)
            recalls, distinct = {}, {}
            for q in questions:
                retrieved, _, _ = rq.retrieve_multi(q["question_en"], values["top_k"])
                got = [r["metadata"]["chunk_id"] for r in retrieved]
                gold = q["gold_chunk_ids"]
                recall = sum(g in got for g in gold) / len(gold) if gold else None
                if recall is not None:
                    recalls[q["id"]] = recall
                answers = set()
                for run in range(1, args.runs + 1):
                    answer = rq.generate_answer(q["question_en"], retrieved)
                    answers.add(norm(answer))
                    f.write(json.dumps({
                        "config": name, "settings": values, "prompt_sha": prompt_sha,
                        "model": rq.OPENAI_MODEL, "run": run,
                        "id": q["id"], "type": q["type"], "expect_abstention": q.get("expect_abstention", False),
                        "question": q["question_en"], "answer": answer,
                        "contexts": [r["content"] for r in retrieved], "retrieved_ids": got,
                        "gold_chunk_ids": gold, "recall": recall,
                        "reference": q["reference"],
                        "reference_source": q.get("reference_source"),
                    }, ensure_ascii=False) + "\n")
                    f.flush()
                distinct[q["id"]] = len(answers)
                print(f"[{name}] Q{q['id']} recall={recall} distinct_answers={len(answers)}/{args.runs}", flush=True)
            apply(old)
            mean = round(sum(recalls.values()) / len(recalls), 3) if recalls else None
            summary["configs"][name] = {"settings": values, "recall_mean": mean,
                                        "recall": recalls, "distinct_answers": distinct}

    json.dump(summary, open(out / "summary.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nprompt_sha={prompt_sha}  ->  {out}")
    for name, s in summary["configs"].items():
        unstable = [i for i, n in s["distinct_answers"].items() if n > 1]
        print(f"  {name:12} recall_mean={s['recall_mean']}  unstable_questions={unstable}")


if __name__ == "__main__":
    main()
