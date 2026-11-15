"""
UTS Handbook scraper — Master of Information Technology (C04295)

Pipeline:
  1. Fetch the course page, parse __NEXT_DATA__ JSON.
  2. Walk curriculumStructure.container recursively.
     - Items with academic_item_type == "subject"  -> record subject code/url.
     - Items with academic_item_type in ("major","stream") -> these are AOS
       pages (e.g. /aos/2026/MAJ02086); fetch them too and recurse into
       their own curriculumStructure the same way.
  3. Deduplicate subject codes across all majors/streams.
  4. Fetch each subject page (with a fixed offering, so assessment info is
     real rather than a placeholder) and pull out the fields we need.
  5. Fetch each subject's "access conditions" page (linked inside
     requisite_detail) to get the real prerequisite-restriction text.
  6. Save everything as JSON, tagged with our metadata schema fields.

Notes:
  - A plain GET + parsing the __NEXT_DATA__ <script> block is enough; no
    Selenium/Playwright needed. The response HTML itself carries no body
    markup (the page renders client-side from that payload), which is why
    the render audit reads the page template rather than the output text —
    see render_audit.py.
  - Each subject costs two GETs: the base page (offering list + page
    template), then the chosen offering's page (its assessment breakdown).
    A base page with no offering selected has selected_assessment: null.
  - The access-conditions page (studentforms.uts.edu.au) is a separate,
    legacy static HTML page — plain GET + tag-stripping is enough there too.
  - Be polite: small delay between requests, reuse a single session.
  - Run this from your own machine — coursehandbook.uts.edu.au and
    studentforms.uts.edu.au are not reachable from this sandbox's network
    allowlist.
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

from render_audit import audit_fields, unaudited_rendered_fields

BASE = "https://coursehandbook.uts.edu.au"
YEAR = 2026
REQUEST_DELAY_SECONDS = 0.5  # be polite to the server
OUTPUT_DIR = Path("handbook_data")

# Offering selection.
#
# assessment data is per-offering: the base page (no offering selected) returns
# selected_assessment: null, so falling back to it loses a subject's assessment
# information entirely. Every subject therefore gets a concrete offering picked
# from its own payload, by this rule:
#
#   City campus first -> Spring, else Autumn, else whatever City offering exists
#   no City offering at all (online/distance-only subjects) -> first offering
#   no offerings at all -> base page, and no assessment chunk for that subject
#
# Spring is the target because the corpus is built around Spring Session; the
# Autumn fallback exists because offering-independent content (description,
# ULOs, requisites, teaching strategies) is identical either way, and for the
# subjects checked the assessment breakdown was identical too. Dropping a
# non-Spring subject would remove a real 2026 MIT subject from the corpus and
# make abstention the correct answer to "what is 32XXX about?", which is wrong.
#
# The offering actually used is recorded in offering_used and baked into
# source_url, so a human can verify any assessment claim on the exact page.
TARGET_OFFERING = "City campus-Spring Session-On campus - Weekly"
PREFERRED_LOCATION = "City campus"
SESSION_PREFERENCE = ("Spring Session", "Autumn Session")

HEADERS = {
    "User-Agent": "uts-rag-assistant-research-scraper/1.0"
}

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)

# Matches the access-conditions link embedded inside requisite_detail HTML,
# e.g. <a href="https://studentforms.uts.edu.au/evop/access/search.cfm?subjectcode=32555">
REQUISITE_LINK_RE = re.compile(
    r'href="(https://studentforms\.uts\.edu\.au/evop/access/search\.cfm\?subjectcode=\d+)"'
)

# The access-conditions page wraps the actual requisite/anti-requisite rules
# in site navigation ("Requisite Waiver • Option selection • ...") before it,
# and a footer (enforcement notes, e-Request instructions, copyright,
# disclaimer links, etc.) after it. Neither adds retrievable information, so
# we trim down to the substantive block: from the first "Anti-requisite(s)"/
# "Requisite(s)" heading through the "...UTS: Handbook." reference note.
REQUISITE_CONTENT_START_RE = re.compile(r"(Anti-requisite\(s\)|Requisite\(s\))")
REQUISITE_CONTENT_END_MARKER = "UTS: Handbook."


# --------------------------------------------------------------------------
# Low-level fetch + parse (coursehandbook.uts.edu.au — Next.js SSR pages)
# --------------------------------------------------------------------------

def fetch_page_props(session: requests.Session, path: str) -> dict:
    """
    GET a handbook page (e.g. '/course/2026/C04295' or '/aos/2026/MAJ02086'
    or '/subject/2026/32555' or '/subject/2026/32555?offering=...') and
    return its whole pageProps dict.

    Subject scraping needs more than pageContent: pageConfig (the page
    template) is what the render audit reads. Course/AOS callers only want
    pageContent and use fetch_page_content() below.
    """
    url = BASE + path
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    # Some pages omit charset in Content-Type, so requests can guess the
    # wrong encoding (e.g. Latin-1 instead of UTF-8), mangling smart quotes
    # and non-breaking spaces ("student’s" -> "studentâs"). Force UTF-8,
    # since these pages are UTF-8-encoded in practice.
    resp.encoding = "utf-8"

    match = NEXT_DATA_RE.search(resp.text)
    if not match:
        raise ValueError(f"__NEXT_DATA__ not found on {url}")

    data = json.loads(match.group(1))
    time.sleep(REQUEST_DELAY_SECONDS)
    return data["props"]["pageProps"]


def fetch_page_content(session: requests.Session, path: str) -> dict:
    """Same GET, but just the pageContent dict (course/AOS pages)."""
    return fetch_page_props(session, path)["pageContent"]


# --------------------------------------------------------------------------
# Access-conditions page (studentforms.uts.edu.au — legacy static HTML)
# --------------------------------------------------------------------------

def fetch_requisite_access_conditions(
    session: requests.Session, requisite_detail_html: str
) -> str | None:
    """
    Extract the access-conditions URL embedded in requisite_detail (if any),
    fetch it, and return the plain-text restriction statement.

    Returns None if no link is present in requisite_detail (i.e. the subject
    page didn't reference an access-conditions page at all).

    Note: requisite_detail itself is boilerplate guidance text shared across
    almost every subject ("go check this page to see your requisites") — it
    is NOT the actual restriction content. The real, subject-specific answer
    lives on this linked page, which is why we fetch it separately here
    rather than treating requisite_detail as sufficient on its own.
    """
    match = REQUISITE_LINK_RE.search(requisite_detail_html or "")
    if not match:
        return None

    url = match.group(1)
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(REQUEST_DELAY_SECONDS)

    # Legacy static page, no JSON — strip tags and collapse whitespace to
    # get the visible restriction text (e.g. "There are no access
    # restrictions enforced for this subject", or a list of actual
    # requisite subjects/conditions).
    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = re.sub(r"\s+", " ", text).strip()
    return _trim_requisite_boilerplate(text)


def _trim_requisite_boilerplate(text: str) -> str:
    """
    Cut the nav/footer boilerplate off the access-conditions page text,
    keeping only the substantive requisite/anti-requisite block.

    Subjects with no requisites at all (e.g. "There are no access
    restrictions enforced for this subject") don't contain the
    "Anti-requisite(s)"/"Requisite(s)" heading, so they pass through
    unchanged — there's no boilerplate to trim in that case anyway.
    """
    start_match = REQUISITE_CONTENT_START_RE.search(text)
    if not start_match:
        return text

    start_idx = start_match.start()
    end_idx = text.find(REQUISITE_CONTENT_END_MARKER, start_idx)
    if end_idx == -1:
        # End marker not found for some reason — keep everything from the
        # start marker onward rather than silently dropping real content.
        return text[start_idx:].strip()

    end_idx += len(REQUISITE_CONTENT_END_MARKER)
    return text[start_idx:end_idx].strip()


# --------------------------------------------------------------------------
# Recursive curriculum-structure walker
# --------------------------------------------------------------------------

@dataclass
class DiscoveredItem:
    code: str
    name: str
    url: str
    item_type: str  # "subject" | "major" | "stream" | "sub_major"


def walk_curriculum_structure(container_list: list) -> list[DiscoveredItem]:
    """
    Recursively walk a curriculumStructure['container'] list (this is the
    *local* structure of one page — course, major, or stream) and return
    every item referenced anywhere inside it (subjects AND aos references).
    Does not follow AOS links itself — that happens one level up.
    """
    found: list[DiscoveredItem] = []

    def _walk(containers):
        for c in containers or []:
            for rel in c.get("relationship", []):
                item_type = rel.get("academic_item_type", {}).get("value", "")
                code = rel.get("academic_item_code", "")
                name = rel.get("academic_item_name", "")
                url = rel.get("academic_item_url", "")
                if code and url:
                    found.append(DiscoveredItem(code, name, url, item_type))
            # nested sub-groups (e.g. "IT Project and Electives" -> "IT Project", "Electives")
            _walk(c.get("container", []))

    _walk(container_list)
    return found


def collect_all_subject_codes(
    session: requests.Session, course_code: str, year: int = YEAR
) -> tuple[dict[str, DiscoveredItem], dict]:
    """
    Starting from the course page, recursively resolve every AOS (major /
    stream / sub_major) it references, and return a deduplicated dict of
    {subject_code: DiscoveredItem} covering every subject reachable from
    the course, plus a resolution log.

    The log matters: an AOS that fails to fetch silently removes every
    subject under it from the corpus, and there is no way to notice that
    afterwards from the subject files alone.
    """
    subjects: dict[str, DiscoveredItem] = {}
    visited_aos: set[str] = set()
    aos_log = {"resolved": [], "failed": []}

    course_content = fetch_page_content(session, f"/course/{year}/{course_code}")
    items = walk_curriculum_structure(course_content["curriculumStructure"]["container"])

    queue = list(items)
    while queue:
        item = queue.pop(0)

        if item.item_type == "subject":
            subjects[item.code] = item
            continue

        # major / stream / sub_major -> AOS page, resolve recursively
        if item.code in visited_aos:
            continue
        visited_aos.add(item.code)

        print(f"  Resolving {item.item_type} {item.code} ({item.name})...")
        try:
            aos_content = fetch_page_content(session, item.url)
        except Exception as exc:
            print(f"  ! Failed to fetch {item.url}: {exc}")
            aos_log["failed"].append(
                {"code": item.code, "name": item.name, "type": item.item_type,
                 "url": item.url, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        aos_log["resolved"].append(
            {"code": item.code, "name": item.name, "type": item.item_type}
        )

        cs = aos_content.get("curriculumStructure")
        if not cs:
            continue
        nested_items = walk_curriculum_structure(cs.get("container", []))
        queue.extend(nested_items)

    return subjects, aos_log


# --------------------------------------------------------------------------
# Subject-page extraction
# --------------------------------------------------------------------------

def choose_offering(offerings: list | None) -> str | None:
    """
    Pick the offering to scrape a subject under. See the TARGET_OFFERING
    comment block for the rule and why it exists. Returns the offering's
    display_name (the value the ?offering= query parameter expects), or None
    when the subject publishes no offering at all.
    """
    published = [o for o in offerings or [] if o.get("display_name")]
    if not published:
        return None

    candidates = [o for o in published if o.get("location") == PREFERRED_LOCATION] or published
    for session_name in SESSION_PREFERENCE:
        for offering in candidates:
            if offering.get("teaching_period") == session_name:
                return offering["display_name"]
    return candidates[0]["display_name"]


def extract_subject_record(
    session: requests.Session,
    code: str,
    url: str,
    year: int = YEAR,
) -> dict:
    """
    Fetch a subject twice: the base page for its offering list and page
    template, then the chosen offering's page so the assessment breakdown is
    real content instead of null. Also resolves the linked access-conditions
    page for the real prerequisite-restriction text.
    """
    base_props = fetch_page_props(session, url)
    base_content = base_props.get("pageContent") or {}

    # The page template is offering-independent, and reading it from the base
    # page keeps the audit comparable across subjects.
    page_config = base_props.get("pageConfig")
    render_audit = audit_fields(page_config, base_content)
    render_audit_at = datetime.now(timezone.utc).isoformat()

    offering_used = choose_offering(base_content.get("offering"))
    content = base_content
    final_url = url
    if offering_used:
        offering_url = f"{url}?offering={quote(offering_used)}"
        try:
            content = fetch_page_props(session, offering_url)["pageContent"]
            final_url = offering_url
        except Exception as exc:
            # Keep the base-page content rather than failing the subject; the
            # missing assessment breakdown is recorded by offering_used=None.
            print(f"  ! Offering fetch failed for {code} ({exc}), keeping base page")
            offering_used = None

    requisite_detail = content.get("requisite_detail")
    try:
        requisite_access_conditions = fetch_requisite_access_conditions(session, requisite_detail)
    except Exception as exc:
        print(f"  ! Access-conditions fetch failed for {code}: {exc}")
        requisite_access_conditions = None

    # The actual per-assessment breakdown (Project/Quiz-test, Groupwork,
    # Weight %) lives in "selected_assessment" — a repeating group whose
    # layout is defined by the "Assessments - Subject - Accordion" module.
    # "assessment_description" is a separate, usually-empty intro text field
    # and is NOT where the Groupwork/Weight data lives.
    selected_assessment = content.get("selected_assessment") or []
    assessment_items = [
        {
            "assessment_type": item.get("assessment_type"),
            "groupwork": item.get("individual"),
            "weight_percent": item.get("weight"),
        }
        for item in selected_assessment
    ]

    return {
        # --- metadata schema fields (shared with synthetic Canvas docs) ---
        "subject_code": content.get("code", code),
        "academic_year": year,
        "source_type": "handbook",
        "week": None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_url": BASE + final_url,
        "offering_used": offering_used,

        # --- render audit (raw-record only; never enters the chunk schema) ---
        # Evidence for which payload fields the page template actually renders.
        # Values below are kept as-is even when the audit says False — the
        # exclusion decision is a policy applied at chunking time
        # (tag_handbook_record.EXCLUDED_FIELDS), not a scraping-time deletion.
        "render_audit": render_audit,
        "render_audit_at": render_audit_at,
        "render_audit_other_fields": sorted(unaudited_rendered_fields(page_config)),

        # --- actual content fields ---
        "title": content.get("title"),
        "credit_points": content.get("credit_points"),
        "description": content.get("description"),
        # Rendered as the "Notes" block under the subject description.
        "notes": content.get("notes"),
        "pre_requisites": content.get("pre_requisites"),
        "requisite_detail": requisite_detail,
        "requisites": content.get("requisites"),
        # Real, subject-specific restriction text resolved from the linked
        # access-conditions page (requisite_detail alone is boilerplate).
        # None if requisite_detail had no such link.
        "requisite_access_conditions": requisite_access_conditions,
        # Intro text field — usually empty/null; kept for completeness.
        "assessment_description": content.get("assessment_description"),
        # Structured breakdown actually shown on the page (Project/Quiz-test,
        # Groupwork, Weight %). This is the field to use for the real content.
        "assessment_items": assessment_items,
        "teaching_strategies": content.get("teaching_strategies"),
        "learning_approach": content.get("learning_approach"),
        "unit_learning_outcomes": content.get("unit_learning_outcomes"),
        # Rendered as the "Additional subject details" accordion (e.g.
        # Engineers Australia competencies). Nested: [{domain, [{description}]}]
        "additional_details_req": content.get("additional_details_req"),
        # Rendered in the right-hand-side panel as "Result type" and
        # "Total Workload hours".
        "grading_schema": content.get("grading_schema"),
        "learning_activity_workload": content.get("learning_activity_workload"),
        "offering": content.get("offering"),
        "parent_academic_org": content.get("parent_academic_org"),
        "study_level_ref": content.get("study_level_ref"),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

ERRORS_PATH = OUTPUT_DIR / "_errors.json"
REPORT_PATH = OUTPUT_DIR / "_scrape_report.json"


def build_report(course_code: str, records: list[dict], errors: list[dict], aos_log: dict) -> dict:
    """Aggregate what was collected, so gaps are visible without re-reading 200 files."""
    offering_target, offering_other, offering_none = [], [], []
    for record in records:
        used = record.get("offering_used")
        code = record["subject_code"]
        if used == TARGET_OFFERING:
            offering_target.append(code)
        elif used:
            offering_other.append({"code": code, "offering": used})
        else:
            offering_none.append(code)

    audit_summary: dict[str, dict] = {}
    for record in records:
        for field, verdict in (record.get("render_audit") or {}).items():
            bucket = audit_summary.setdefault(field, {"true": 0, "false": 0, "null": 0})
            bucket[{True: "true", False: "false", None: "null"}[verdict]] += 1

    other_rendered: dict[str, int] = {}
    for record in records:
        for field in record.get("render_audit_other_fields") or []:
            other_rendered[field] = other_rendered.get(field, 0) + 1

    return {
        "course_code": course_code,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subjects_total": len(records) + len(errors),
        "subjects_ok": len(records),
        "subjects_failed": len(errors),
        "aos": {
            "resolved": aos_log.get("resolved", []),
            "failed": aos_log.get("failed", []),
        },
        "offering": {
            "target": TARGET_OFFERING,
            "target_used_count": len(offering_target),
            "target_used": offering_target,
            "other_offering_count": len(offering_other),
            "other_offering": offering_other,
            "no_offering_count": len(offering_none),
            "no_offering": offering_none,
        },
        "render_audit": audit_summary,
        # Fields the template renders that render_audit.AUDIT_FIELDS doesn't
        # name — a coverage gap check, not an audit result.
        "rendered_fields_outside_audit": other_rendered,
        "empty_assessment_items": [
            r["subject_code"] for r in records if not r.get("assessment_items")
        ],
        "null_requisite_access_conditions": [
            r["subject_code"] for r in records if r.get("requisite_access_conditions") is None
        ],
        "empty_notes": [r["subject_code"] for r in records if not r.get("notes")],
        "empty_additional_details": [
            r["subject_code"] for r in records if not r.get("additional_details_req")
        ],
    }


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def preflight(course_code: str = "C04295", year: int = YEAR) -> dict:
    """
    Resolve the subject list without scraping anything, so a human can check
    it against the Handbook's Course structure before ~600 requests go out.
    """
    session = requests.Session()
    subject_map, aos_log = collect_all_subject_codes(session, course_code, year)

    print(f"\nAOS followed ({len(aos_log['resolved'])}):")
    for entry in aos_log["resolved"]:
        print(f"  {entry['type']:10} {entry['code']:10} {entry['name']}")
    if aos_log["failed"]:
        print(f"\nAOS FAILED ({len(aos_log['failed'])}):")
        for entry in aos_log["failed"]:
            print(f"  {entry['code']:10} {entry['name']}  <- {entry['error']}")

    print(f"\nSubjects ({len(subject_map)}):")
    for code, item in sorted(subject_map.items()):
        print(f"  {code}  {item.name}")

    return {"subjects": subject_map, "aos": aos_log}


def scrape_subjects(
    subject_map: dict[str, DiscoveredItem],
    year: int = YEAR,
    force: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Scrape each subject, one JSON file per subject. Resumable (an existing
    file is skipped unless force) and fault-tolerant (one bad subject does
    not stop the run).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    session = requests.Session()
    records, errors = [], []

    for i, (code, item) in enumerate(sorted(subject_map.items()), start=1):
        out_path = OUTPUT_DIR / f"{code}.json"
        if out_path.exists() and not force:
            print(f"[{i}/{len(subject_map)}] {code} - already scraped, skipping")
            records.append(json.loads(out_path.read_text(encoding="utf-8")))
            continue

        print(f"[{i}/{len(subject_map)}] Fetching subject {code} - {item.name}")
        try:
            record = extract_subject_record(session, code, item.url, year)
        except Exception as exc:
            print(f"  ! Failed to fetch subject {code}: {exc}")
            errors.append({"code": code, "name": item.name, "url": item.url,
                           "error": f"{type(exc).__name__}: {exc}"})
            continue

        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        records.append(record)

    return records, errors


def main(course_code: str = "C04295", year: int = YEAR, force: bool = False,
         only: list[str] | None = None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Resolving all subjects reachable from {course_code}...")
    session = requests.Session()
    subject_map, aos_log = collect_all_subject_codes(session, course_code, year)
    print(f"Found {len(subject_map)} unique subjects.\n")

    if only:
        missing = [c for c in only if c not in subject_map]
        if missing:
            print(f"! Not reachable from {course_code}, skipping: {missing}")
        subject_map = {c: subject_map[c] for c in only if c in subject_map}
        print(f"Pilot run over {len(subject_map)} subjects.\n")

    records, errors = scrape_subjects(subject_map, year, force=force)

    if errors:
        ERRORS_PATH.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_report(course_code, records, errors, aos_log)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. {len(records)} ok, {len(errors)} failed -> {OUTPUT_DIR}/")
    print(f"Report: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", default="C04295")
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--preflight", action="store_true",
                        help="list reachable subjects and AOS, scrape nothing")
    parser.add_argument("--only", nargs="+", metavar="CODE",
                        help="scrape just these subject codes (pilot run)")
    parser.add_argument("--force", action="store_true",
                        help="re-scrape subjects that already have a JSON file")
    args = parser.parse_args()

    if args.preflight:
        preflight(args.course, args.year)
    else:
        main(args.course, args.year, force=args.force, only=args.only)
