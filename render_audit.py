"""
Render audit — which __NEXT_DATA__ payload fields does the page actually show?

Why this exists
---------------
RQ2 (cross-source faithfulness) requires that every indexed sentence be
verifiable by a human who opens `source_url`. The Handbook's __NEXT_DATA__
payload carries CMS fields that the page never displays (confirmed case:
`learning_approach`, which holds the "Topics covered" list). Indexing those
would pass automated faithfulness checks while being unverifiable on the page.

How it works
------------
The original plan was to diff the payload values against the rendered HTML of
the same GET response. That is impossible here: coursehandbook.uts.edu.au ships
a shell document (~1.1 KB of visible text: header, nav, footer) and renders the
body on the client. No payload text is present in the response HTML, so a text
diff marks every field — including the ULO control — as not rendered.

The same response does, however, carry `props.pageProps.pageConfig`: the CMS
page template. It lists the render modules of the page and binds each module
slot to a payload field (`data_source: "data"`) or a literal (`"text"`).
Repeating groups (assessment accordion, additional-details accordion) name
their payload collection in `relative_path` / `full_path`. A payload field bound
to no module is never rendered — that is the evidence recorded here.

This is a stronger claim than string matching: it reads the template that
governs rendering, rather than guessing from output text. It is a claim about
template binding, not pixel visibility — a bound field can still be empty, and
that is reported separately by value.

No browser needed; pageConfig arrives in the same GET the scraper already makes.
"""

import json
import sys

# Payload fields we care about for corpus construction. Kept explicit so the
# audit reports a stable set of columns across every subject.
AUDIT_FIELDS = (
    "description",
    "teaching_strategies",
    "learning_approach",
    "assessment_description",
    "requisite_detail",
    "unit_learning_outcomes",
    "notes",
    "additional_details_req",
    "selected_assessment",
    # Right-hand-side panel facts ("Result type", "Total Workload hours").
    "grading_schema",
    "learning_activity_workload",
)


def rendered_fields(page_config: list | None) -> set[str]:
    """Every payload field the page template binds to a render module."""
    found: set[str] = set()

    def walk(modules):
        for module in modules or []:
            for key, value in module.items():
                if key.endswith("_field_map") and isinstance(value, dict):
                    for slot in value.values():
                        if isinstance(slot, dict) and slot.get("data_source") == "data":
                            found.add(slot.get("data_field", ""))
            # Repeating groups (accordions/cards over a list) name the payload
            # collection they iterate rather than binding it in a field map.
            for path_key in ("relative_path", "full_path"):
                path = module.get(path_key) or ""
                # full_path can look like "additional_details_req[y].additional_details_req"
                for part in path.replace("[x]", "").replace("[y]", "").split("."):
                    part = part.strip()
                    if part:
                        found.add(part)
            walk(module.get("sub_modules"))

    for config in page_config or []:
        for modules in (config.get("sections") or {}).values():
            walk(modules)

    found.discard("")
    return found


def audit_fields(page_config: list | None, record: dict | None = None) -> dict[str, bool | None]:
    """
    {field: True (rendered) | False (not rendered) | None (undecidable)}.

    None only happens when the page carries no pageConfig at all — then we know
    nothing either way and refuse to guess. `record` is accepted so callers can
    pass the extracted record; it is not needed for the binding decision.
    """
    if not page_config:
        return {field: None for field in AUDIT_FIELDS}

    bound = rendered_fields(page_config)
    return {field: field in bound for field in AUDIT_FIELDS}


def unaudited_rendered_fields(page_config: list | None) -> set[str]:
    """
    Fields the template renders that AUDIT_FIELDS doesn't cover — i.e. content
    visible on the page that the scraper may not be collecting. Reported so a
    silent coverage gap surfaces instead of going unnoticed.
    """
    return rendered_fields(page_config) - set(AUDIT_FIELDS)


# --------------------------------------------------------------------------
# CLI: python render_audit.py 32555 32144
# --------------------------------------------------------------------------

def main():
    from scrape_handbook import YEAR, fetch_page_props  # local import: avoids a cycle

    import requests

    codes = sys.argv[1:] or ["32555", "32144"]
    session = requests.Session()

    for code in codes:
        props = fetch_page_props(session, f"/subject/{YEAR}/{code}")
        config = props.get("pageConfig")
        result = audit_fields(config, props.get("pageContent"))
        print(f"\n{code}")
        for field, verdict in result.items():
            mark = {True: "rendered", False: "NOT RENDERED", None: "undecidable"}[verdict]
            print(f"  {field:26} {mark}")
        extra = sorted(unaudited_rendered_fields(config))
        print(f"  other rendered fields: {json.dumps(extra)}")


if __name__ == "__main__":
    main()
