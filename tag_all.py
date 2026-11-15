"""
Tag every raw Handbook record in handbook_data/ into tagged_data/.

    python tag_all.py            # all subjects
    python tag_all.py 32555 ...  # just these codes

Same chunking as tag_handbook_record.py (it is imported, not copied) —
this only adds the loop, the totals, and the page_type distribution.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from tag_handbook_record import tag_record

IN_DIR = Path("handbook_data")
OUT_DIR = Path("tagged_data")


def main():
    only = set(sys.argv[1:])
    # Files starting with "_" are run reports and error logs, not subject records.
    paths = sorted(p for p in IN_DIR.glob("*.json") if not p.name.startswith("_"))
    if only:
        paths = [p for p in paths if p.stem in only]

    OUT_DIR.mkdir(exist_ok=True)
    page_types = Counter()
    sections = Counter()
    total = 0

    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        chunks = tag_record(record)
        out_path = OUT_DIR / f"{record['subject_code']}_chunks.json"
        out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        total += len(chunks)
        page_types.update(c["page_type"] for c in chunks)
        sections.update(c["section"] for c in chunks)

    print(f"\n{len(paths)} subjects -> {total} chunks in {OUT_DIR}/")
    print("\npage_type:")
    for name, count in page_types.most_common():
        print(f"  {name:12} {count}")
    print("\nsection:")
    for name, count in sections.most_common():
        print(f"  {name:30} {count}")


if __name__ == "__main__":
    main()
