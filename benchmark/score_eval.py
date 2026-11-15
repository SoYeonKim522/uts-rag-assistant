"""
Scores a run produced by run_eval.py (answers.jsonl).

  python3 benchmark/score_eval.py benchmark/results/<run>     # judge defaults to gemini-3.1-flash-lite
  python3 benchmark/score_eval.py <dir> --limit 4             # quick check
  python3 benchmark/score_eval.py <dir> --tag rescore1        # re-score with the same judge, to
                                                              # measure how much the judge varies

Metrics
  RQ1  ragas ContextPrecisionWithReference / ContextRecall, on answerable questions only, once
       per (config, question) because retrieval is identical across runs. Recall over gold chunk
       ids is already computed in run_eval.
       id_precision applies the same formula to gold ids, with no LLM involved, and is reported
       next to the ragas precision.
  RQ2  ragas Faithfulness is scored for every row but aggregated over answerable questions only.
       On an unanswerable question a correct abstention cannot be supported by the context and so
       is penalised (Q15 is a full abstention and sits at 0.667), while copying facts about some
       other subject scores full marks. Unanswerable questions are therefore judged by abstention
       alone; the per-question values are still kept in per_question.
       abstention (full / partial / none): the judge only marks each requested item in the
         question as answered / unavailable / not_addressed (ABSTENTION_PROMPT, a separate call),
         and abstention_label() derives the label from those marks.
         Replaced on 2026-10-06: letting the judge pick the label itself (JUDGE_PROMPT's
         "abstention", now kept only as abstention_direct) gave the right reason with the wrong
         label on partial questions (Q11 x3, Q13 run2, Q9 run1), and asking for the reason first
         flipped the error the other way. Per-item marking agrees 42/42, Q3 aside.
         A sentence telling the judge to separate "part of an item answered, the rest reported
         missing" also catches k5 Q7 ("Quiz 2 due date is not available") as partial. Across the
         135 rows of the 9/29, 9/30 and v3.0 runs, 126 match the label checked by hand; all 9
         mismatches are Q3, where the answer names the page but not the procedure, so the judge
         says full while the gold range is partial.
         **Scoring files older than that date use the previous method** -- the new one is what
         --tag parts produced.
         - unanswerable question: full is a success (abstention success rate)
         - answerable question: full is over-abstention
       citation support (judge): whether each source listed under "Sources:" is among the
         retrieved contexts and actually supports the answer. The ratio is supported citations
         over all citations.
       completeness (judge, a separate call), answerable questions only, per row: the reference is
         split into items and the ratio of items present in the answer is reported.
         **It compares the answer against the reference, not the context.** That is a different
         axis from faithfulness, so the two do not belong in one column of the report:
         faithfulness only asks whether anything was made up and cannot see what was left out
         (Q10 scores 1.0 on faithfulness while answering 4 to 6 of 8 items). This metric covers
         that gap.
         ragas FactualCorrectness(mode="recall") was tried first and dropped: measured on
         2026-09-30, the four atomicity/coverage combinations gave Q10 0.67/0.00/0.25/0.00. Two
         zeros for an answer that does contain 4 to 6 of the 8 items -- returned values, not
         exceptions -- and no per-item verdicts to inspect. Here the per-item verdicts are stored
         in completeness_items so they can be audited.
         The call is kept separate from JUDGE_PROMPT so that seeing the reference cannot sway the
         abstention and citation verdicts.
  partial (benchmark v3.0, Q11-13, type="partial"): excluded from the answerable and unanswerable
       aggregates and reported in its own "partial" block, because the correct behaviour on these
       questions is a "partial" abstention -- answer the part the sources cover and say the rest is
       not there.
         partial_abstention_success_rate (partial) / over_abstention_rate (full) /
         missed_abstention_rate (none)
       faithfulness: the "not in the sources" sentence cannot be supported by the context, so the
       ceiling is below 1. Reported separately.
       completeness: scored against the reference as written, so "did it say what was missing"
       counts as an item too.
       ragas context precision/recall: measured with the "not in the sources" paragraph removed
       from the reference (supported_reference).

The judge is a different model from the generator (gpt-4o-mini), so no model grades its own
answers. Final setup (2026-09-29): generation gpt-4o-mini, judge gemini-3.1-flash-lite with
thinking off. Scoring before 9/28 used a gpt-4.1 judge and must not be mixed into the same
comparison. Judge output can shift even at temperature 0, so read any conclusion together with
the variance across runs.

Output (written next to the input):
  scores_<judge>.jsonl        per-row scores plus the judge's stated reasons
  score_summary_<judge>.json  per-config aggregates plus per-question mean and standard deviation
  The filename carries the judge (plus _limitN for --limit and _<tag> for --tag) so changing the
  judge or running a smoke test cannot overwrite existing scores -- a --limit 4 run once wiped the
  9/28 gpt-4.1 results. Always pass --tag when re-scoring the same folder with the same judge;
  without it the previous file is overwritten.
"""
import argparse
import asyncio
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall, Faithfulness

JUDGE_PROMPT = """\
You are grading an answer produced by a retrieval-augmented assistant for university students.
You are given the student's question, the numbered context passages the assistant retrieved, and its answer.

Return a JSON object with exactly these keys:
- "abstention": one of
    "full"    -- the answer does not provide what the question asks for (e.g. the requested value, ranking or
                 list) and says it is not available. It may still mention related facts or which sources are
                 missing -- if the core request is left unanswered, it is "full";
    "partial" -- the question has several parts; the answer provides at least one requested part and
                 explicitly says another requested part is not available;
    "none"    -- the answer does not say that any requested information is unavailable.
  A question with one core request (e.g. "which subject is easiest", even if it names several subjects) has only
  one part, so it is either "full" or "none" -- never "partial".
  Judge only what the answer explicitly says, not whether you think it is complete: pointing the student to a
  page or resource for further details is NOT an abstention.
- "citations": one entry per source listed under the answer's "Sources:" heading (empty list if none or "None"):
    {"cited": "<the source line, shortened>",
     "in_context": true if this source corresponds to one of the context passages (match subject and section),
     "supports_answer": true if that passage contains information the answer actually uses}
- "reason": one short sentence explaining the abstention label.
"""


ABSTENTION_PROMPT = """\
You are grading an answer produced by a retrieval-augmented assistant for university students.
You are given the student's question, the numbered context passages the assistant retrieved, and its answer.

List the distinct things the QUESTION asks for (one entry per requested item; a question with one core request has
one entry, even if it names several subjects). For each, record what the ANSWER does with it:
  "answered"     -- the answer gives the requested information (even if incomplete);
  "unavailable"  -- the answer explicitly says this requested information is not available / not provided /
                    cannot be confirmed. Related facts the question did not ask for do not make it "answered".
  "not_addressed" -- the answer neither gives it nor says it is unavailable.
If the answer gives some of the information for a requested item but explicitly says another piece of it is not available (e.g. one assessment's due date out of several, or the procedure behind a named page), record both: an "answered" entry for what it gives and an "unavailable" entry for what it says is missing.
Pointing the student to a page or resource for further details is NOT "unavailable".

Return a JSON object with exactly this key:
- "parts": [{"part": "<requested item, shortened>", "status": "answered" | "unavailable" | "not_addressed"}]
"""


def abstention_label(parts: list[dict]) -> str:
    """Derive the label from the per-item statuses the judge marked. Letting the judge pick the
    label directly gave the right reason with the wrong label (2026-10-06; see the module
    docstring)."""
    s = {p.get("status") for p in parts}
    if "unavailable" not in s:
        return "none"
    return "partial" if "answered" in s else "full"


async def abstention_parts(client: AsyncOpenAI, model: str, row: dict, extra: dict) -> list[dict]:
    resp = await client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"}, **extra,
        messages=[{"role": "system", "content": ABSTENTION_PROMPT}, {"role": "user", "content": judge_message(row)}],
    )
    return json.loads(resp.choices[0].message.content).get("parts", [])


COMPLETENESS_PROMPT = """\
You are checking how much of a reference answer a retrieval-augmented assistant's answer covers.

Split the reference answer into the distinct factual items a student would need (a due date, a subject
name, a weighting, a requisite, a learning outcome -- one item each). Ignore any trailing "Source:" or
"Sources:" line: it is provenance, not an item.

For each item, decide whether the assistant's answer states it. Wording may differ; an item counts as
present if the answer conveys the same fact. An item stated with a wrong value is NOT present.

Return a JSON object with exactly this key:
- "items": [{"item": "<the item, shortened>", "in_answer": true or false}]
"""


def completeness_message(row: dict) -> str:
    return (f"[Question]\n{row['question']}\n\n[Reference answer]\n{reference_claims_only(row['reference'])}"
            f"\n\n[Assistant answer]\n{row['answer']}")


async def completeness(client: AsyncOpenAI, model: str, row: dict, extra: dict) -> dict:
    resp = await client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"}, **extra,
        messages=[{"role": "system", "content": COMPLETENESS_PROMPT},
                  {"role": "user", "content": completeness_message(row)}],
    )
    return json.loads(resp.choices[0].message.content)


def reference_claims_only(reference: str) -> str:
    """Drop the trailing "Source: ..." line from a reference, for completeness scoring only.
    That line is provenance, not a factual claim, but it gets split into claims and costs marks.
    It also only appears in Q1-9, not Q10, so leaving it in would favour Q10 -- the question with
    the worst omissions (measured 2026-09-30: Q2 scored 0.5 although the answer covered the whole
    reference). ragas context precision/recall read the same reference and may carry the same
    bias; that is untested."""
    lines = reference.strip().split("\n")
    while lines and lines[-1].strip().lower().startswith(("source:", "sources:")):
        lines.pop()
    return "\n".join(lines).strip()


UNSUPPORTED_MARK = "The available sources do not"


def supported_reference(reference: str) -> str:
    """Drop the "not in the sources" paragraph from a partial question's reference, for ragas
    context precision/recall only. Nothing in the context can support that paragraph, so it cuts
    recall for reasons that have nothing to do with retrieval. Every Q11-13 reference is built the
    same way: the answer paragraph, a "The available sources do not ..." paragraph, then the
    Source line. Completeness keeps the paragraph, because "did it say what was missing" has to
    count as an item."""
    paras = [p for p in reference.split("\n\n") if not p.strip().startswith(UNSUPPORTED_MARK)]
    assert len(paras) < len(reference.split("\n\n")), "partial reference without the unsupported paragraph"
    return "\n\n".join(paras)


def judge_message(row: dict) -> str:
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(row["contexts"], 1))
    return f"[Question]\n{row['question']}\n\n[Context passages]\n{ctx}\n\n[Answer]\n{row['answer']}"


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def judge_client(model: str) -> tuple[AsyncOpenAI, dict]:
    """Send gemini-* models to Gemini's OpenAI-compatible endpoint (GEMINI_API_KEY) and anything
    else to OpenAI. Gemini Flash thinks by default, and thinking tokens both eat into max_tokens
    (truncating the output) and bill at the output rate, so thinking is turned off."""
    # max_retries: 429s are common on free and tier-1 per-minute limits; the SDK waits out
    # retry-after and tries again.
    if model.startswith("gemini"):
        return AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=os.environ["GEMINI_API_KEY"], max_retries=10), \
            {"reasoning_effort": "none"}
    return AsyncOpenAI(max_retries=10), {}


async def judge(client: AsyncOpenAI, model: str, row: dict, extra: dict) -> dict:
    resp = await client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"}, **extra,
        messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": judge_message(row)}],
    )
    return json.loads(resp.choices[0].message.content)


def value(result) -> float | None:
    v = getattr(result, "value", result)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


async def score_all(rows: list[dict], judge_model: str, concurrency: int) -> tuple[list[dict], dict]:
    client, extra = judge_client(judge_model)
    # max_tokens: at the default 1024, per-statement verdicts on a long answer such as Q10 get
    # truncated into an IncompleteOutputException.
    # temperature: ragas defaults to 0.01 (InstructorModelArgs); set to 0 to match the judge calls
    # made here. At 0.01, re-scoring the same answers moved faithfulness on 4 of 45 rows and
    # context precision on one question (2026-10-06).
    llm = llm_factory(judge_model, client=client, max_tokens=4096, temperature=0, **extra)
    faith, crecall, cprec = Faithfulness(llm=llm), ContextRecall(llm=llm), ContextPrecisionWithReference(llm=llm)
    sem = asyncio.Semaphore(concurrency)

    async def guarded(make):
        """make: a function returning a fresh coroutine, since a retry needs a new one."""
        for attempt in range(6):
            async with sem:
                try:
                    return await make()
                except Exception as e:  # one failed row must not lose the whole run; record it
                    if "429" not in str(e) or attempt == 5:
                        return e
            # Per-minute limit the SDK retries could not ride out. Wait outside the semaphore so
            # other calls can go ahead.
            await asyncio.sleep(20 * (attempt + 1))

    async def row_scores(row):
        # Completeness applies to answerable questions only. An unanswerable question's reference
        # is the "no information" sentence itself, which scores a correct abstention full marks and
        # an irrelevant answer zero -- the metric inverts.
        answerable = row["type"] != "unanswerable"
        f, j, ab, comp = await asyncio.gather(
            guarded(lambda: faith.ascore(user_input=row["question"], response=row["answer"], retrieved_contexts=row["contexts"])),
            guarded(lambda: judge(client, judge_model, row, extra)),
            guarded(lambda: abstention_parts(client, judge_model, row, extra)),
            guarded(lambda: completeness(client, judge_model, row, extra)) if answerable else asyncio.sleep(0),
        )
        cites = [] if isinstance(j, Exception) else j.get("citations", [])
        items = [] if (not answerable or isinstance(comp, Exception)) else comp.get("items", [])
        return {
            **{k: row[k] for k in ("config", "run", "id", "type", "expect_abstention", "reference_source")},
            "faithfulness": None if isinstance(f, Exception) else value(f),
            "completeness": None if not items else round(sum(bool(i.get("in_answer")) for i in items) / len(items), 3),
            "completeness_items": items,
            "abstention": None if isinstance(ab, Exception) else abstention_label(ab),
            "abstention_parts": None if isinstance(ab, Exception) else ab,
            # The label the judge picked itself, kept for auditing only; not aggregated.
            "abstention_direct": None if isinstance(j, Exception) else j.get("abstention"),
            "abstention_reason": None if isinstance(j, Exception) else j.get("reason"),
            "citations": cites,
            "citation_support": (sum(bool(c.get("in_context") and c.get("supports_answer")) for c in cites) / len(cites))
                                if cites else None,
            "errors": [repr(x) for x in (f, j, ab, comp) if isinstance(x, Exception)],
        }

    # Context metrics run once per (config, question): retrieval is the same in every run.
    ctx_rows = {}
    for r in rows:
        if r["type"] != "unanswerable":
            ctx_rows.setdefault((r["config"], r["id"]), r)

    async def ctx_scores(r):
        ref = supported_reference(r["reference"]) if r["type"] == "partial" else r["reference"]
        kw = dict(user_input=r["question"], retrieved_contexts=r["contexts"], reference=ref)
        p, c = await asyncio.gather(guarded(lambda: cprec.ascore(**kw)), guarded(lambda: crecall.ascore(**kw)))
        return (r["config"], r["id"]), {"ragas_context_precision": None if isinstance(p, Exception) else value(p),
                                        "ragas_context_recall": None if isinstance(c, Exception) else value(c),
                                        "ctx_errors": [repr(x) for x in (p, c) if isinstance(x, Exception)]}

    scored = await asyncio.gather(*(row_scores(r) for r in rows))
    ctx = dict(await asyncio.gather(*(ctx_scores(r) for r in ctx_rows.values())))
    return scored, ctx


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def id_precision(retrieved: list[str], gold: list[str]) -> float:
    """The same average-precision formula as ragas ContextPrecision, except the verdict is simply
    whether a chunk is a gold id, with no judge. ragas asks of each chunk alone whether it is
    useful for the whole answer, so a question that needs several chunks together (Q10) can score
    0 even when only gold chunks were retrieved. This is the comparison value."""
    v = [int(c in gold) for c in retrieved]
    return sum(sum(v[: i + 1]) / (i + 1) * v[i] for i in range(len(v))) / sum(v) if any(v) else 0.0


def summarize(rows, scored, ctx) -> dict:
    recall = {(r["config"], r["id"]): r["recall"] for r in rows}
    id_prec = {(r["config"], r["id"]): id_precision(r["retrieved_ids"], r["gold_chunk_ids"])
               for r in rows if r["gold_chunk_ids"]}
    by_cfg = defaultdict(list)
    for s in scored:
        by_cfg[s["config"]].append(s)

    out = {}
    for cfg, ss in by_cfg.items():
        per_q = {}
        for qid in sorted({s["id"] for s in ss}):
            qs = [s for s in ss if s["id"] == qid]
            f = [s["faithfulness"] for s in qs if s["faithfulness"] is not None]
            cp = [s["completeness"] for s in qs if s["completeness"] is not None]
            per_q[qid] = {
                "type": qs[0]["type"],
                "faithfulness_mean": mean(f),
                "faithfulness_sd": round(statistics.stdev(f), 3) if len(f) > 1 else None,
                "completeness_mean": mean(cp),
                "completeness_sd": round(statistics.stdev(cp), 3) if len(cp) > 1 else None,
                "abstention": [s["abstention"] for s in qs],
                "citation_support": mean([s["citation_support"] for s in qs]),
                "id_recall": recall.get((cfg, qid)),
                "id_precision": id_prec.get((cfg, qid)),
                **ctx.get((cfg, qid), {}),
            }
        # Partial questions go in neither bucket: their correct behaviour is a "partial"
        # abstention, so mixing them into answerable would contaminate the over- and
        # partial-abstention rates and break comparison with the earlier (v2.0) numbers.
        ans = [s for s in ss if s["type"] not in ("partial", "unanswerable")]
        par = [s for s in ss if s["type"] == "partial"]
        una = [s for s in ss if s["type"] == "unanswerable"]
        answerable_q = [v for v in per_q.values() if v["type"] not in ("partial", "unanswerable")]
        partial_q = [v for v in per_q.values() if v["type"] == "partial"]
        out[cfg] = {
            "rq1": {
                "id_recall_mean": mean([v["id_recall"] for v in answerable_q]),
                "id_precision_mean": mean([v["id_precision"] for v in answerable_q]),
                "ragas_context_precision_mean": mean([v.get("ragas_context_precision") for v in answerable_q]),
                "ragas_context_recall_mean": mean([v.get("ragas_context_recall") for v in answerable_q]),
            },
            "rq2": {
                "faithfulness_answerable_mean": mean([s["faithfulness"] for s in ans]),
                "abstention_success_rate": mean([s["abstention"] == "full" for s in una]) if una else None,
                "over_abstention_rate": mean([s["abstention"] == "full" for s in ans]) if ans else None,
                "partial_abstention_rate_answerable": mean([s["abstention"] == "partial" for s in ans]) if ans else None,
                "citation_support_mean": mean([s["citation_support"] for s in ss]),
                # answer vs reference, a different axis from faithfulness (answer vs context)
                "completeness_answerable_mean": mean([s["completeness"] for s in ans]),
            },
            # partial (v3.0 Q11-13): did it answer the covered part and flag only the rest as missing
            "partial": {
                "partial_abstention_success_rate": mean([s["abstention"] == "partial" for s in par]) if par else None,
                "over_abstention_rate": mean([s["abstention"] == "full" for s in par]) if par else None,
                "missed_abstention_rate": mean([s["abstention"] == "none" for s in par]) if par else None,
                # The "not in the sources" sentence is unsupported by the context and loses marks,
                # so this does not belong in the same column as answerable faithfulness.
                "faithfulness_mean": mean([s["faithfulness"] for s in par]),
                "completeness_mean": mean([s["completeness"] for s in par]),
                "id_recall_mean": mean([v["id_recall"] for v in partial_q]),
                "id_precision_mean": mean([v["id_precision"] for v in partial_q]),
            } if par else None,
            "errors": sum(len(s["errors"]) for s in ss) + sum(len(v.get("ctx_errors", [])) for v in per_q.values()),
            "per_question": per_q,
        }
    return out


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--limit", type=int, help="score only the first N rows (smoke test)")
    p.add_argument("--tag", help="suffix for output filenames -- use when re-scoring with the same judge")
    args = p.parse_args()

    d = Path(args.results_dir)
    rows = [json.loads(l) for l in open(d / "answers.jsonl")]
    if args.limit:
        rows = rows[: args.limit]
    gen_models = {r["model"] for r in rows}
    if args.judge_model in gen_models:
        print(f"warning: judge model {args.judge_model} is also the generator -- self-grading bias", file=sys.stderr)

    tag = args.judge_model + (f"_limit{args.limit}" if args.limit else "") + (f"_{args.tag}" if args.tag else "")
    if (d / f"scores_{tag}.jsonl").exists():
        sys.exit(f"scores_{tag}.jsonl already exists -- pass a different --tag instead of overwriting it")

    scored, ctx = asyncio.run(score_all(rows, args.judge_model, args.concurrency))
    out = d / f"score_summary_{tag}.json"
    with open(d / f"scores_{tag}.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = {"judge_model": args.judge_model, "rows": len(rows), "configs": summarize(rows, scored, ctx)}
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)

    print(f"judge={args.judge_model}  rows={len(rows)}  ->  {out}")
    for cfg, s in summary["configs"].items():
        print(f"\n[{cfg}] errors={s['errors']}")
        print("  RQ1", s["rq1"])
        print("  RQ2", s["rq2"])
        if s["partial"]:
            print("  partial", s["partial"])


if __name__ == "__main__":
    main()
