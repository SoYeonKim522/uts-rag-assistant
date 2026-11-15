"""
Mock Canvas with the Study Assistant built in -- the demo site.

Every page is rendered from the same chunk files that went into the vector store
(canvas_synthetic/, tagged_data/). Nothing is written just for the demo, which means a
citation in an answer can link straight to the page the answer came from.

No dependencies: the server itself is standard library only, and /api/ask imports
rag_query lazily, so the site comes up with no embedding model and no API key. Asking a
question needs the installed requirements and OPENAI_API_KEY.

    python3 demo_server.py        # http://localhost:8000, DEMO_PORT to change the port
"""

import html
import json
import os
import re
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
PORT = int(os.environ.get("DEMO_PORT", 8000))
TOP_K = 8

# Enrolled subject = a subject that has Canvas data (the same rule as
# rag_query.enrolled_subjects()).
CANVAS_DIR = ROOT / "canvas_synthetic"
HANDBOOK_DIR = ROOT / "tagged_data"

# Heuristic behind the UI badge that marks an answer as "not in the material".
# ponytail: phrase matching. This drives one on-screen badge, not a scored metric, so it
# does not need a judge model.
ABSTAIN_PATTERNS = re.compile(
    r"not available|not provided|does not (?:contain|include|specify|provide)|"
    r"no (?:subject|record|information|details?)\b[^.]{0,120}?(?:found|available|in the)|"
    r"is not specified|could not be found|cannot provide|i (?:do not|don't) have|"
    r"sources:\s*none",
    re.IGNORECASE,
)

TITLE_PREFIX = re.compile(r"^\[[^\]]*\]\s*")


@lru_cache(maxsize=1)
def corpus() -> dict:
    subjects: dict[str, dict] = {}

    for path in sorted(CANVAS_DIR.glob("*_canvas_chunks.json")):
        for c in json.loads(path.read_text()):
            s = subjects.setdefault(
                c["subject_code"],
                {"code": c["subject_code"], "name": c["subject_name"], "canvas": [], "handbook": []},
            )
            s["canvas"].append(strip_prefix(c))

    for code, s in subjects.items():
        path = HANDBOOK_DIR / f"{code}_chunks.json"
        if path.exists():
            s["handbook"] = [strip_prefix(c) for c in json.loads(path.read_text())]

    return {"subjects": [subjects[k] for k in sorted(subjects)], "top_k": TOP_K}


def strip_prefix(chunk: dict) -> dict:
    """
    Drop the "[32555 ... | Assignments]" header from the front of content: it is there for
    the embedding, not for the reader.

    Also decode HTML entities. scrape_handbook.py strips tags from Handbook bodies but
    leaves entities alone, so 112 Handbook chunks still carry literal &#39; &#8226;
    &rsquo; and the like (canvas_synthetic is unaffected). This decoding is for display
    only; the stored chunks keep the entities.
    """
    c = dict(chunk)
    c["body"] = html.unescape(TITLE_PREFIX.sub("", c.get("content", "")))
    if c.get("title"):
        c["title"] = html.unescape(c["title"])
    c.pop("content", None)
    return c


def ask(question: str) -> dict:
    import rag_query 

    retrieved, sub_queries, _ = rag_query.retrieve_multi(question, TOP_K)
    answer = rag_query.generate_answer(question, retrieved)
    return {
        "answer": answer,
        "sub_queries": sub_queries,
        "abstained": bool(ABSTAIN_PATTERNS.search(answer)),
        "top_k": TOP_K,
        "sources": [
            {
                "chunk_id": r["metadata"].get("chunk_id"),
                "subject_code": r["metadata"].get("subject_code"),
                "subject_name": r["metadata"].get("subject_name"),
                "source_type": r["metadata"].get("source_type"),
                "page_type": r["metadata"].get("page_type"),
                "section": r["metadata"].get("section"),
                "title": r["metadata"].get("title"),
                "source_url": r["metadata"].get("source_url"),
                "distance": round(r["distance"], 4),
            }
            for r in retrieved
        ],
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT / "demo"), **kw)

    def do_GET(self):
        if self.path.startswith("/api/corpus"):
            return self.send_json(corpus())
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/ask"):
            return self.send_error(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        question = (json.loads(body or b"{}").get("question") or "").strip()
        if not question:
            return self.send_json({"error": "question is required"}, status=400)
        try:
            return self.send_json(ask(question))
        except ModuleNotFoundError as exc:
            return self.send_json(
                {"error": f"'{exc.name}' is not installed. Answering needs the full pipeline: "
                          "pip install -r requirements.txt, then put OPENAI_API_KEY in .env. "
                          "Browsing the subject pages works without either."},
                status=500,
            )
        except Exception as exc:  # show the error in the panel instead of dying mid-demo
            return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/corpus"):
            super().log_message(fmt, *args)


def demo():
    c = corpus()
    assert len(c["subjects"]) == 4, c
    assert all(s["handbook"] and s["canvas"] for s in c["subjects"])
    assert not c["subjects"][0]["canvas"][0]["body"].startswith("["), "header was not stripped"
    bodies = "".join(ch["body"] for s in c["subjects"] for ch in s["canvas"] + s["handbook"])
    assert "&#39;" not in bodies and "&#8226;" not in bodies and "&rsquo;" not in bodies, "HTML entities were not decoded"
    desc = next(ch for ch in corpus()["subjects"][2]["handbook"] if ch["section"] == "Subject description")
    assert "you'll" in desc["body"], desc["body"][:200]
    for text in [
        "This information is not available.",
        "No subject with the code 39999 Advanced Data Programming was found in the available sources.",
        "The marking criteria are not provided in the reference material.",
        "Sources: None",
    ]:
        assert ABSTAIN_PATTERNS.search(text), text
    for text in [
        "Week 2 covers object-oriented design. Sources: Canvas - Modules.",
        "The prerequisites are 32541 Project Management. Sources: Handbook - Requisites.",
    ]:
        assert not ABSTAIN_PATTERNS.search(text), text
    print("ok", [(s["code"], len(s["canvas"]), len(s["handbook"])) for s in c["subjects"]])


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        demo()
    else:
        print(f"http://localhost:{PORT}")
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
