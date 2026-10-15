"""
Dump the raw text of selected corpus pages, so that the reference answers in
eval/test_questions.yaml can be written from the documents themselves rather
than from outside knowledge.

This matters for the measurement: context_recall splits a reference answer into
atomic claims and checks each one against the retrieved chunks. A claim that is
true but absent from the corpus is scored as a retrieval miss, penalising the
retriever for something it had no way to find.

Uses PyMuPDF, like src/ingestion/extract_corpus.py, so the text read here is the
same text that was indexed.

Usage, from the repo root:
  python eval/dump_pages.py --find "Article 9" --pdf 03_taxonomy_regulation_2020_852.pdf
  python eval/dump_pages.py --dump Q2
  python eval/dump_pages.py --dump-all

No API calls, no quota, runs offline. Output goes to eval/source_extracts/,
which is git-ignored: it holds raw regulatory text, not project content.
"""
import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF

CORPUS_DIR = Path("data/raw")
OUT_DIR = Path("eval/source_extracts")

CSRD = "01_csrd_directive_2022_2464.pdf"
ESRS = "02_esrs_set1_2023_2772.pdf"
TAXO = "03_taxonomy_regulation_2020_852.pdf"
SCHN = "04_schneider_electric_urd_2024.pdf"
BNPP = "05_bnp_paribas_urd_2025.pdf"

# 1-based inclusive page ranges, one page of margin around the target.
# Located with --find on 2026-09-23.
PAGES = {
    "Q1": [(CSRD, 5, 7), (CSRD, 27, 29), (CSRD, 62, 64)],  # recital 17, Art. 19a, Art. 5(2)
    "Q2": [(ESRS, 7, 12)],                                  # double materiality
    "Q3": [(ESRS, 90, 94)],                                 # E1-6, Scopes 1/2/3
    "Q4": [(TAXO, 16, 18)],                                 # Article 9, six objectives
    "Q5": [(TAXO, 15, 16), (TAXO, 23, 24)],                 # Article 3(b) + Article 17
    "Q6": [(ESRS, 25, 27), (ESRS, 36, 42)],                 # phase-in, Appendix C
    "Q7": [(SCHN, 148, 151)],                               # ESRS side is Q3's range
    "Q8": [(BNPP, 759, 761), (TAXO, 17, 18)],               # GAR table + Article 8
}


def _open(pdf_name: str) -> fitz.Document:
    path = CORPUS_DIR / pdf_name
    if not path.exists():
        raise SystemExit(f"Not found: {path}. Download the corpus per data/raw/SOURCES.md.")
    return fitz.open(str(path))


def find(pdf_name: str, pattern: str) -> None:
    """Print every occurrence of pattern, with its page number and some context.

    Whitespace is collapsed before matching because article headings are often
    centred and broken across lines ("Article\\n9").
    """
    doc = _open(pdf_name)
    rx = re.compile(re.escape(pattern).replace(r"\ ", r"\s+"), re.IGNORECASE)
    hits = 0
    for i, page in enumerate(doc, start=1):
        text = re.sub(r"\s+", " ", page.get_text())
        for m in rx.finditer(text):
            start = max(0, m.start() - 80)
            print(f"p.{i} : ...{text[start:m.start() + 260]}...\n")
            hits += 1
    print(f"{hits} occurrence(s) in {pdf_name} ({len(doc)} pages)")
    doc.close()


def dump(qid: str) -> None:
    if qid not in PAGES:
        raise SystemExit(f"No page range for {qid}. Locate it with --find first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = []
    for pdf_name, start, end in PAGES[qid]:
        doc = _open(pdf_name)
        for i in range(max(1, start), min(end, len(doc)) + 1):
            blocks.append(f"\n=== {pdf_name} | PDF page {i} ===\n{doc[i - 1].get_text()}")
        doc.close()
    out = OUT_DIR / f"{qid}.txt"
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(f"{qid} -> {out} ({out.stat().st_size} bytes)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--find", help="text to search for, e.g. 'Article 17'")
    p.add_argument("--pdf", help="corpus file name, required with --find")
    p.add_argument("--dump", help="question id, e.g. Q2")
    p.add_argument("--dump-all", action="store_true")
    a = p.parse_args()

    if a.find:
        if not a.pdf:
            raise SystemExit("--find requires --pdf")
        find(a.pdf, a.find)
    elif a.dump:
        dump(a.dump.upper())
    elif a.dump_all:
        for qid in PAGES:
            dump(qid)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
