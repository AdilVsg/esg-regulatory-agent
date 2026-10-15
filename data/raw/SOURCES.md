# Corpus sources

Five English-language source documents, all publicly available. They are not
committed to the repository (see `.gitignore`): download each one and save it
under `data/raw/` with the file name indicated. This file is the reference for
reproducing the corpus.

Regulatory texts are not static: a delegated regulation amending 2023/2772 for
simplification was published in mid-2026. This is expected, and it is exactly
why the agent has a web-search fallback rather than relying on the corpus alone.

## Regulatory texts (3)

### 1. CSRD - Directive (EU) 2022/2464
Corporate Sustainability Reporting Directive. Defines scope, timeline and audit
obligations.

- Consolidated English text (HTML): https://eur-lex.europa.eu/eli/dir/2022/2464/2025-04-17/eng
- Original OJ text (PDF): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2464
- Save as: `01_csrd_directive_2022_2464.pdf`

### 2. ESRS Set 1 - Delegated Regulation (EU) 2023/2772
The detailed reporting standards. The full annex runs to several hundred pages;
only **ESRS 1 (General requirements)** and **ESRS E1 (Climate change)** are
ingested, which kept the corpus small enough to verify answer locations by hand.
`src/ingestion/extract_corpus.py` selects those page ranges from the full PDF.

- Consolidated English text (HTML): https://eur-lex.europa.eu/eli/reg_del/2023/2772/2025-01-01/eng
  (use the PDF download button on that page)
- Save as: `02_esrs_set1_2023_2772.pdf`

### 3. EU Taxonomy Regulation (EU) 2020/852
The six environmental objectives, substantial contribution criteria and the
Do No Significant Harm (DNSH) principle.

- English text (PDF): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32020R0852&from=EN
- Save as: `03_taxonomy_regulation_2020_852.pdf`

## Corporate reports (2)

### 4. Schneider Electric - 2024 Universal Registration Document
Industrial issuer whose sustainability statement is structured around ESRS.

- Direct PDF: https://www.se.com/ww/en/assets/564/document/510443/2024-universal-registration-document.pdf
- Save as: `04_schneider_electric_urd_2024.pdf`

### 5. BNP Paribas - 2025 Universal Registration Document and Annual Financial Report
Banking-sector counterpart, which discloses Taxonomy alignment through the
Green Asset Ratio rather than turnover and CapEx.

- Direct PDF: https://cdn-group.bnpparibas.com/uploads/file/universal-registration-document-annual-financial-report-2025.pdf
- Save as: `05_bnp_paribas_urd_2025.pdf`

## Expected layout

```
data/raw/
├── 01_csrd_directive_2022_2464.pdf
├── 02_esrs_set1_2023_2772.pdf
├── 03_taxonomy_regulation_2020_852.pdf
├── 04_schneider_electric_urd_2024.pdf
└── 05_bnp_paribas_urd_2025.pdf
```

The answer locations for each test question are recorded in the
`expected_source` field of `eval/test_questions.yaml`, verified by hand against
these exact files.

## Known coverage gap

Directive 2013/34/EU (the Accounting Directive) is not in the corpus. The CSRD
amends it rather than restating it, so the numeric size thresholds defining a
"large undertaking" (balance sheet total, net turnover, employees) live in
2013/34/EU, not here.

The corpus can therefore only partly answer scope questions: recital 17 and
Article 5(2) of 2022/2464 give the 500-employee threshold and the staged
application dates, but not the full definition. This was surfaced by Q1 of the
test set. The gap is accepted rather than closed with a sixth document, since
filling it at query time is the web-search branch's job.

To close it, add Directive 2013/34/EU (consolidated English text:
https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02013L0034-20240109)
as `06_accounting_directive_2013_34.pdf` and rerun the ingestion pipeline.

## Text extraction

The first extraction pass used `pypdf`, which corrupted the text of the two EU
Official Journal PDFs (`01_csrd...` and `03_taxonomy...`): their narrow
two-column layout made it insert spurious spaces inside words
(`inf or mation`, `compara bility`), which degraded both embeddings and
retrieval. `src/ingestion/extract_corpus.py` uses PyMuPDF instead, which
extracts these layouts cleanly. Avoid `pypdf` for any script reading these files.
