"""
Extract text from the corpus PDFs into per-page JSON records.

For 02_esrs_set1_2023_2772.pdf, only pages covering ESRS 1 (pp. 1-46)
and ESRS E1 (pp. 81-126) are extracted. The full delegated regulation
is 338 pages and includes ESRS 2, E2-E5, S1-S4 and G1, which are out
of scope for this corpus (see data/raw/SOURCES.md).
"""
import json
from pathlib import Path

import fitz  # PyMuPDF

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (filename, page_ranges): page_ranges is a list of (start, end) inclusive,
# 1-indexed, matching the PDF's own page order. None means whole document.
SOURCES = [
    ("01_csrd_directive_2022_2464.pdf", None),
    ("02_esrs_set1_2023_2772.pdf", [(1, 46), (81, 126)]),
    ("03_taxonomy_regulation_2020_852.pdf", None),
    ("04_schneider_electric_urd_2024.pdf", None),
    ("05_bnp_paribas_urd_2025.pdf", None),
]


def pages_to_extract(total_pages, page_ranges):
    if page_ranges is None:
        return list(range(1, total_pages + 1))
    pages = []
    for start, end in page_ranges:
        pages.extend(range(start, min(end, total_pages) + 1))
    return pages


def extract_pdf(filename, page_ranges):
    path = RAW_DIR / filename
    doc = fitz.open(str(path))
    total = len(doc)
    pages = pages_to_extract(total, page_ranges)

    records = []
    for p in pages:
        text = doc[p - 1].get_text() or ""
        text = text.strip()
        if not text:
            continue
        records.append({"source": filename, "page": p, "text": text})
    doc.close()
    return records


def main():
    manifest = []
    for filename, page_ranges in SOURCES:
        print(f"Extracting {filename} ...")
        records = extract_pdf(filename, page_ranges)
        out_path = OUT_DIR / f"{Path(filename).stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  -> {len(records)} pages with text written to {out_path}")
        manifest.append({"source": filename, "pages_extracted": len(records)})

    with open(OUT_DIR / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nDone. Manifest written to data/processed/text/_manifest.json")


if __name__ == "__main__":
    main()