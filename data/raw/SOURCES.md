# Corpus sources

Five English-language source documents. Download each manually and save under
`data/raw/` using the filename indicated. Files are not committed to git
(see `.gitignore`); this file is the source of truth for reproducing the corpus.

Regulatory texts are not static - a delegated regulation amending 2023/2772
for simplification was published in mid-2026. This is expected and is exactly
why the agent needs a web-search fallback rather than relying on the corpus alone.

## Regulatory texts (3)

### 1. CSRD - Directive (EU) 2022/2464
Corporate Sustainability Reporting Directive. Defines scope, timeline, and audit obligations.

- Consolidated English text (HTML): https://eur-lex.europa.eu/eli/dir/2022/2464/2025-04-17/eng
- Original OJ text (PDF): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2464
- Save as: `01_csrd_directive_2022_2464.pdf`

### 2. ESRS Set 1 - Delegated Regulation (EU) 2023/2772
The detailed reporting standards. Only take **ESRS 1 (General requirements)**
and **ESRS E1 (Climate change)** from Annex I for v1 of the corpus - the full
annex is several hundred pages and will make manual verification impractical.

- Consolidated English text (HTML): https://eur-lex.europa.eu/eli/reg_del/2023/2772/2025-01-01/eng
  (use the "PDF" download button on this page to get ESRS 1 / ESRS E1)
- Save as: `02_esrs_set1_2023_2772.pdf`

### 3. EU Taxonomy Regulation (EU) 2020/852
The six environmental objectives, substantial contribution criteria, and the DNSH principle.

- English text (PDF): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32020R0852&from=EN
- Save as: `03_taxonomy_regulation_2020_852.pdf`

## Corporate reports (2)

### 4. Schneider Electric - 2024 Universal Registration Document
Includes the sustainability report structured around ESRS.

- Direct PDF: https://www.se.com/ww/en/assets/564/document/510443/2024-universal-registration-document.pdf
- Save as: `04_schneider_electric_urd_2024.pdf`
- Note: the 2025 URD (filed March 2026) may since be available at
  https://www.se.com/us/en/download/document/2025-URD/ if you prefer the more recent one -
  check whether it requires an account before relying on it.

### 5. BNP Paribas - 2025 Universal Registration Document and Annual Financial Report
Banking-sector report - gives you a direct link to your current work context at Crédit Agricole.

- Direct PDF: https://cdn-group.bnpparibas.com/uploads/file/universal-registration-document-annual-financial-report-2025.pdf
- Save as: `05_bnp_paribas_urd_2025.pdf`

## Known coverage gap

Directive (EU) 2013/34/EU (the "Accounting Directive") is not in this corpus.
The CSRD (Directive 2022/2464) amends it rather than restating it, so the
full quantitative size thresholds for a "large undertaking" (balance sheet
total, net turnover) are defined in 2013/34/EU itself, not in our corpus.

Our corpus can partially answer scope/threshold questions (e.g. Recital 17
of 2022/2464 gives the 500-employee public-interest-entity threshold, and
Article 5(2) gives the phased application dates), but not the full picture.

This was identified via Q1 in `eval/test_questions.yaml` returning weak
retrieval results even after fixing extraction quality (see note below).
Decision: accept this as a known gap rather than add a 6th document, since
this is exactly the kind of hole the agent's web-search branch should cover
at query time rather than the static corpus trying to be exhaustive.

If this gap needs closing later, add Directive 2013/34/EU (consolidated
English text: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02013L0034-20240109)
as `06_accounting_directive_2013_34.pdf` and re-run the full ingestion pipeline.

## Known extraction issue (fixed)

The first extraction pass used `pypdf`, which corrupted text on
`01_csrd_directive_2022_2464.pdf` and `03_taxonomy_regulation_2020_852.pdf`
(both are EU Official Journal PDFs with narrow two-column layouts). Symptom:
words split with spurious spaces (e.g. "inf or mation", "compara bility"),
which degraded embedding quality and retrieval for those two documents.

Fixed by switching `src/ingestion/extract_corpus.py` to `PyMuPDF` (`fitz`),
which handles this layout correctly. If you ever re-add `pypdf` for another
script, don't reuse it for these two-column EU Official Journal PDFs.

## After downloading

You should have 5 files in `data/raw/`:

data/raw/
├── 01_csrd_directive_2022_2464.pdf
├── 02_esrs_set1_2023_2772.pdf
├── 03_taxonomy_regulation_2020_852.pdf
├── 04_schneider_electric_urd_2024.pdf
└── 05_bnp_paribas_urd_2025.pdf

Before writing any ingestion code, open each PDF once and locate (roughly) where
the answers to the questions in `eval/test_questions.yaml` live. This is what lets
you sanity-check the pipeline's output by eye later instead of guessing.