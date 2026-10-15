# eval/dump_pages.py
"""Extraction du texte brut des pages du corpus, pour rediger les gold answers.

Usage (depuis la racine du repo) :
  python eval/dump_pages.py --find "Article 9" --pdf 03_taxonomy_regulation_2020_852.pdf
  python eval/dump_pages.py --dump Q2
  python eval/dump_pages.py --dump-all
"""

import argparse
import re
from pathlib import Path

from pypdf import PdfReader

CORPUS_DIR = Path("data/raw") 
OUT_DIR = Path("eval/source_extracts")

CSRD = "01_csrd_directive_2022_2464.pdf"
ESRS = "02_esrs_set1_2023_2772.pdf"
TAXO = "03_taxonomy_regulation_2020_852.pdf"
SCHN = "04_schneider_electric_urd_2024.pdf"
BNPP = "05_bnp_paribas_urd_2025.pdf"

# Pages 1-based inclusives, telles que notees dans test_questions.yaml,
# avec une page de marge de part et d'autre.
# 1-based inclusive page ranges, one page of margin around the target.
# Verified against the --find output on 2026-09-23.
PAGES = {
    "Q1": [(CSRD, 5, 7), (CSRD, 27, 29), (CSRD, 62, 64)],
    "Q2": [(ESRS, 7, 12)],                 # double materiality
    "Q3": [(ESRS, 90, 94)],                # E1-6, Scope 1/2/3
    "Q4": [(TAXO, 16, 18)],                # Article 9, six objectives
    "Q5": [(TAXO, 15, 16), (TAXO, 23, 24)],  # Article 3(b) + Article 17
    "Q6": [(ESRS, 25, 27), (ESRS, 36, 42)],  # phase-in, Appendix C
    "Q7": [(SCHN, 148, 151)],              # ESRS side comes from Q3
    "Q8": [(BNPP, 759, 761), (TAXO, 17, 18)],  # GAR table + Article 8
}

def _reader(pdf_name: str) -> PdfReader:
    path = CORPUS_DIR / pdf_name
    if not path.exists():
        raise SystemExit(f"Introuvable : {path}. Corrige CORPUS_DIR.")
    return PdfReader(str(path))


def _tolerant(pattern: str) -> str:
    """
    Build a regex that tolerates whitespace between any two characters.

    OJ PDFs encode kerning in a way pypdf reads as real spaces, so the text
    layer contains "Ar ticle", "compl iance", "en vironmentally". A literal
    pattern therefore matches nothing. Inserting \\s* between every character
    of the needle makes it match regardless of where the spurious breaks fall.
    """
    return r"\s*".join(re.escape(c) for c in pattern if not c.isspace())


def find(pdf_name: str, pattern: str) -> None:
    reader = _reader(pdf_name)
    rx = re.compile(_tolerant(pattern), re.IGNORECASE)
    hits = 0
    for i, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "")
        for m in rx.finditer(text):
            start = max(0, m.start() - 80)
            excerpt = text[start:m.start() + 260]
            print(f"p.{i} : ...{excerpt}...\n")
            hits += 1
    print(f"{hits} occurrence(s) dans {pdf_name} ({len(reader.pages)} pages)")


def dump(qid: str) -> None:
    if qid not in PAGES:
        raise SystemExit(f"{qid} n'a pas de pages connues. Utilise --find d'abord.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = []
    for pdf_name, start, end in PAGES[qid]:
        reader = _reader(pdf_name)
        last = len(reader.pages)
        for i in range(max(1, start), min(end, last) + 1):
            text = reader.pages[i - 1].extract_text() or ""
            blocks.append(f"\n=== {pdf_name} | PDF page {i} ===\n{text}")
    out = OUT_DIR / f"{qid}.txt"
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(f"{qid} -> {out} ({out.stat().st_size} octets)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--find")
    p.add_argument("--pdf")
    p.add_argument("--dump")
    p.add_argument("--dump-all", action="store_true")
    a = p.parse_args()

    if a.find:
        if not a.pdf:
            raise SystemExit("--find exige --pdf")
        find(a.pdf, a.find)
    elif a.dump:
        dump(a.dump)
    elif a.dump_all:
        for qid in PAGES:
            dump(qid)
    else:
        p.print_help()


if __name__ == "__main__":
    main()