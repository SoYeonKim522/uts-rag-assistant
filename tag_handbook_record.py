"""
Tag a single raw Handbook subject record (output of scrape_handbook.py)
into schema-tagged chunks, following the agreed metadata schema.

Usage:
    python tag_handbook_record.py handbook_data/32555.json

Produces:
    tagged_data/32555_chunks.json
"""

import json
import os
import re
import sys
from pathlib import Path

ACADEMIC_YEAR = 2026  # fallback only; the scraper writes academic_year per record

# Switch for the synonym hint added to Assessment chunks ("Also known as:
# assignments..."). On by default, which is how the shipped corpus was built. Turn it off
#     export INCLUDE_SYNONYM_HINT=false
# and re-run this script to measure retrieval without the hint, i.e. whether the embedding
# model works out assignment ~ Project/Quiz-test on its own.
INCLUDE_SYNONYM_HINT = os.environ.get("INCLUDE_SYNONYM_HINT", "true").lower() != "false"

# Fields kept out of the corpus on the evidence of the render audit, which is stored as
# render_audit in each raw record (see render_audit.py). They exist in the __NEXT_DATA__
# payload but are never rendered on the page, so a reader cannot verify a citation against
# source_url -- which RQ2 (faithfulness) requires.
# The audit is the evidence and the exclusion is the policy, so the branch below reads this
# constant, not the audit values.
EXCLUDED_FIELDS = {"learning_approach"}


def is_excluded(record: dict, field: str) -> bool:
    """Apply the exclusion policy, logging one line to stderr per excluded field."""
    if field not in EXCLUDED_FIELDS:
        return False
    print(f"[excluded] {record.get('subject_code')} {field} (not rendered)", file=sys.stderr)
    return True


# --------------------------------------------------------------------------
# HTML -> plain text (lightweight; matches the tag-stripping style already
# used in scrape_handbook.py for the access-conditions page)
# --------------------------------------------------------------------------

def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    text = html
    text = re.sub(r"</li>", "\n", text)
    text = re.sub(r"<li>", "- ", text)
    text = re.sub(r"</p>", "\n\n", text)
    # <br>/<br/> tags carry a line break but leave no closing tag for the
    # generic tag-stripping regex below to hook into — without this, they'd
    # just vanish and glue the surrounding sentences together with no space
    # at all (e.g. "...subject results.If you receive...").
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&#8226;", "-")
    # Normalize NBSP (\xa0, from &nbsp; entities in the source HTML) to a
    # regular ASCII space, so downstream tokenization/embedding doesn't
    # treat visually-identical text as different characters.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Requisite/anti-requisite item list -> one item per line
# --------------------------------------------------------------------------

def format_requisite_text(text: str | None) -> str:
    if not text:
        return text or ""

    text = re.sub(r"\s*(Anti-requisite\(s\))", r"\n\1", text)
    text = re.sub(r"\s*(Requisite\(s\))", r"\n\1", text)
    text = re.sub(r"\s*(Item Details)", r"\n\1", text)
    text = re.sub(r"\s*(Item Type Details)", r"\n\1", text)
    text = re.sub(r"\s+(?=\d+\s+\d{5}\s)", "\n", text)
    text = re.sub(r"\s+(?=\d+[a-z]?\s+(?:Academic|Admission|Other) requisite)", "\n", text)

    return text.strip()


# --------------------------------------------------------------------------
# Standard content header: [subject_code subject_name | section | Due ...]
# Due is only included when it actually applies (not the case for Handbook).
# --------------------------------------------------------------------------

def build_header(subject_code: str, subject_name: str, section: str, due: str | None = None) -> str:
    header = f"[{subject_code} {subject_name} | {section}"
    if due:
        header += f" | Due {due}"
    header += "]"
    return header


# --------------------------------------------------------------------------
# Chunk builder
# --------------------------------------------------------------------------

def make_chunk(record: dict, seq: int, page_type: str, section: str,
               body: str, title: str | None = None) -> dict:
    subject_code = record["subject_code"]
    subject_name = record.get("title") or subject_code
    header = build_header(subject_code, subject_name, section)
    content = f"{header}\n{body}".strip()

    return {
        "source_id": f"HB_{subject_code}",
        "chunk_id": f"HB_{subject_code}_{seq:03d}",
        "subject_code": subject_code,
        "subject_name": subject_name,
        "academic_year": record.get("academic_year", ACADEMIC_YEAR),
        "source_type": "handbook",
        "is_synthetic": False,
        "page_type": page_type,
        "section": section,
        "title": title,
        "week": None,
        "due_date": None,
        "source_url": record.get("source_url"),
        "retrieved_at": record.get("retrieved_at"),
        "updated_at": None,
        "content": content,
    }


def tag_record(record: dict) -> list[dict]:
    # Records scraped before the audit was added must still work. The exclusion policy comes
    # from EXCLUDED_FIELDS rather than from the audit values, so a missing audit changes
    # nothing about the output -- only the evidence is absent.
    if "render_audit" not in record:
        print(f"[warn] {record.get('subject_code')}: no render_audit "
              f"(pre-audit raw record); the exclusion policy still applies.", file=sys.stderr)

    chunks = []
    seq = 1

    if record.get("description"):
        # keeps credit_points searchable
        body = html_to_text(record["description"])
        credit_points = record.get("credit_points")
        if credit_points:
            body = f"Credit points: {credit_points}\n\n{body}"
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Subject description",
            body=body,
        ))
        seq += 1

    # When excluded, the "Subject content" chunk is not created at all and later seq numbers
    # shift up. That is intended; no placeholder is inserted.
    if record.get("learning_approach") and not is_excluded(record, "learning_approach"):
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Subject content",
            body=html_to_text(record["learning_approach"]),
        ))
        seq += 1

    if record.get("teaching_strategies"):
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Learning and teaching activities",
            body=html_to_text(record["teaching_strategies"]),
        ))
        seq += 1

    ulo = record.get("unit_learning_outcomes") or []
    if ulo:
        ulo_sorted = sorted(ulo, key=lambda x: int(x.get("number", 0)))
        lines = [
            f"{item.get('code')}: {html_to_text(item.get('description'))}"
            for item in ulo_sorted
        ]
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Learning outcomes",
            body="\n".join(lines),
            title="Learning outcomes",
        ))
        seq += 1

    requisite_detail_text = html_to_text(record.get("requisite_detail"))
    access_conditions_text = format_requisite_text(record.get("requisite_access_conditions"))
    requisite_body_parts = [p for p in [requisite_detail_text, access_conditions_text] if p]
    if requisite_body_parts:
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Requisites",
            body="\n\n".join(requisite_body_parts),
        ))
        seq += 1

    assessment_items = record.get("assessment_items") or []
    assessment_description = html_to_text(record.get("assessment_description"))
    if assessment_items or assessment_description:
        # Students ask about "assignments", not "assessment". The Handbook text never uses
        # the word assignment -- it only names task types such as Project or Quiz/test -- and
        # retrieval kept missing this chunk for assignment questions, so the synonym is
        # stated in the body. Set INCLUDE_SYNONYM_HINT=false to measure retrieval without it.
        lines = []
        if INCLUDE_SYNONYM_HINT:
            lines.append("Also known as: assignments, coursework tasks (과제).")
        if assessment_description:
            lines.append(assessment_description)
        for item in assessment_items:
            lines.append(
                f"- {item.get('assessment_type')} "
                f"({item.get('groupwork')}): {item.get('weight_percent')}%"
            )
        chunks.append(make_chunk(
            record, seq,
            page_type="assessment",
            section="Assessment information",
            body="\n".join(lines),
        ))
        seq += 1

    # --- The next three sections are rendered on the page but were missing from earlier
    # scrapes. On the page Notes comes right after the description; it is appended here
    # instead so existing chunk_ids do not shift, which keeps older runs comparable.

    notes = html_to_text(record.get("notes"))
    if notes:
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Notes",
            body=notes,
        ))
        seq += 1

    # The "Additional subject details" accordion: [{domain, additional_details_req: [{description}]}]
    additional_lines = []
    for group in record.get("additional_details_req") or []:
        domain = (group.get("domain") or "").strip()
        details = [
            html_to_text(d.get("description"))
            for d in group.get("additional_details_req") or []
        ]
        details = [d for d in details if d]
        if not details:
            continue
        additional_lines.append(f"{domain}\n{chr(10).join(details)}" if domain else "\n".join(details))
    if additional_lines:
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Additional subject details",
            body="\n\n".join(additional_lines),
        ))
        seq += 1

    # Facts shown in the right-hand panel of the page (Result type, Total Workload hours...).
    detail_lines = []
    for label, value in (
        ("Credit points", record.get("credit_points")),
        ("Result type", record.get("grading_schema")),
        ("Total workload hours", record.get("learning_activity_workload")),
        ("Study level", record.get("study_level_ref")),
        ("Faculty", record.get("parent_academic_org")),
        ("Availability", record.get("offering_used")),
    ):
        if value:
            detail_lines.append(f"{label}: {value}")
    if detail_lines:
        chunks.append(make_chunk(
            record, seq,
            page_type="reference",
            section="Subject details",
            body="\n".join(detail_lines),
        ))
        seq += 1

    return chunks


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python tag_handbook_record.py <path_to_raw_subject_record.json>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    record = json.loads(in_path.read_text(encoding="utf-8"))

    chunks = tag_record(record)

    out_dir = Path("tagged_data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{record['subject_code']}_chunks.json"
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Tagged {len(chunks)} chunks from {in_path.name} -> {out_path}")
    for c in chunks:
        print(f"  {c['chunk_id']}  [{c['page_type']}]  {c['section']}")


if __name__ == "__main__":
    main()