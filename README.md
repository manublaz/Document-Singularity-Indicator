# Documentary Singularity Indicator (SId)

## Overview

This repository contains the implementation of the **Documentary Singularity Indicator (SId)**, a metric designed to support the selection and evaluation of scientific document corpora for artificial intelligence training. SId quantifies the informational singularity of each document within a corpus, combining lexical originality with bibliometric impact to produce a normalized score in the [0, 1] range.

The project is developed by the Department of Library and Information Science at the Complutense University of Madrid (UCM) and is part of an ongoing line of research into corpus quality and representativeness for large language models and other AI systems.

---

## Research Context

The construction of high-quality, non-redundant corpora is a critical step in training AI systems, particularly in specialized scientific domains. Standard bibliometric indicators measure individual document quality, but they do not capture how much unique, non-redundant information a document contributes to a given corpus. SId addresses this gap by combining two complementary dimensions:

- **O_d (Originality):** A measure of lexical singularity derived from smoothed TF-IDF weighting over n-gram frequency distributions, normalized against the P99 of the corpus distribution to reduce the influence of extreme values.
- **I_d (Impact):** A normalized bibliometric impact score computed as the log-ratio of a document's citation count to the maximum observed in the corpus.

The final score is computed as:

```
SId = alpha * O_d + (1 - alpha) * I_d
```

where `alpha` (default: 0.7) controls the relative weight of originality versus impact.

---

## Versions

### SId v1 (original)

The original formulation applies a nearest-neighbor lexical filter: only n-grams whose closest lexical neighbor within the same document has frequency >= 2 contribute to the score. While theoretically motivated, this filter causes a large fraction of documents to receive a score of zero, reducing the discriminatory power of the index for sparse or short texts.

### SId v2 (current)

The revised formulation eliminates the restrictive nearest-neighbor filter, so all n-grams contribute to the originality component. Additional improvements include:

- Smoothed IDF with a decay factor to penalize high-frequency n-grams more gradually.
- An internal redundancy penalty in the O_d denominator (`log(1 + T_d / Omega)`), which discounts documents that repeat the same n-grams many times.
- P99 normalization of O_d, ensuring that scores are well-calibrated and comparable across corpora of different sizes.
- Explicit, interpretable control of the originality/impact trade-off through the `alpha` parameter.

---

## Repository Structure

```
Document-Singularity-Indicator/
│
├── database/                          Calcia corpus database (split SQL dumps)
│   ├── calcia_part_001.sql
│   ├── calcia_part_002.sql
│   └── ... (65 files total)
│
├── datasets/                          CSV output tables from the SId analysis
│   ├── calcia_bottom50.csv            Bottom 50 documents by SId v2
│   ├── calcia_correlaciones.csv       Pearson correlations with corpus variables
│   ├── calcia_deciles.csv             Summary statistics by SId v2 decile
│   ├── calcia_singularity_results.csv Full results for all documents
│   ├── calcia_stats_comparison.csv    Descriptive statistics: SId v1 vs SId v2
│   ├── calcia_top50.csv               Top 50 documents by SId v2
│   └── generate_figures_en.py         Script to regenerate figures from CSV data
│
├── figures/                           Analysis figures (Spanish and English versions)
│   ├── figura_1.png                   Score distribution (Spanish)
│   ├── figura_1_en.png                Score distribution (English)
│   ├── figura_2.png                   Pearson correlations (Spanish)
│   ├── figura_2_en.png                Pearson correlations (English)
│   ├── figura_3.png                   SId vs bibliometric impact (Spanish)
│   ├── figura_3_en.png                SId vs bibliometric impact (English)
│   ├── figura_4.png                   Score distribution by citation quartile (Spanish)
│   ├── figura_4_en.png                Score distribution by citation quartile (English)
│   ├── figura_5.png                   O_d and I_d decomposition (Spanish)
│   ├── figura_5_en.png                O_d and I_d decomposition (English)
│   ├── figura_6.png                   Alpha sensitivity analysis (Spanish)
│   ├── figura_6_en.png                Alpha sensitivity analysis (English)
│   ├── figura_7.png                   Feature profile by decile (Spanish)
│   ├── figura_7_en.png                Feature profile by decile (English)
│   ├── figura_8.png                   Top/Bottom 20 ranking (Spanish)
│   ├── figura_8_en.png                Top/Bottom 20 ranking (English)
│   ├── figura_9.png                   Synthetic corpus validation (Spanish)
│   └── figura_9_en.png                Synthetic corpus validation (English)
│
├── formula/                           Formal mathematical specification of SId
│   ├── Ecuaciones_ISd.pdf             Formula document (Spanish)
│   ├── Ecuaciones_ISd.tex             LaTeX source (Spanish)
│   ├── Ecuaciones_ISd_en.pdf          Formula document (English)
│   └── Ecuaciones_ISd_en.tex          LaTeX source (English)
│
├── queries/                           PubMed search queries used to build the corpus
│   └── calciadb-queries.txt
│
├── reports/                           Full analysis report
│   └── singularity_report.txt
│
├── calciaDB_install.py                Assembles calcia.db from the SQL dump files
├── calciaDB_pubmed_scraping.py        Retrieves and ingests documents from PubMed
├── requirements.txt                   Python dependencies
├── singularitycalcia_v2.py            Main analysis pipeline (Spanish output)
└── singularitycalcia_v2_en.py         Main analysis pipeline (English output)
```

---

## Database

The corpus database (`calcia.db`) is distributed as 65 split SQL dump files located in the `database/` folder. To reassemble and import it locally, run the provided installation script:

```bash
python calciaDB_install.py
```

This script concatenates the partial SQL files in order and restores the full SQLite database ready for analysis.

### Schema

The main `corpus` table is structured as follows:

| Column       | Type    | Description                                 |
|--------------|---------|---------------------------------------------|
| id           | INTEGER | Primary key                                 |
| url          | TEXT    | Source URL                                  |
| title        | TEXT    | Document title                              |
| author       | TEXT    | Author(s)                                   |
| datepub      | TEXT    | Publication date                            |
| text         | TEXT    | Preprocessed full text                      |
| textoriginal | TEXT    | Original full text (preferred if available) |
| citations    | INTEGER | Citation count                              |
| created_at   | TEXT    | Ingestion timestamp                         |
| updated_at   | TEXT    | Last update timestamp                       |

---

## Corpus Construction

Documents were retrieved from PubMed using the search queries listed in `queries/calciadb-queries.txt`. The retrieval and ingestion pipeline is implemented in `calciaDB_pubmed_scraping.py`, which handles API calls, metadata extraction, full-text preprocessing, and citation count ingestion.

---

## Installation

Python 3.9 or later is required.

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`.

---

## Usage

### Analyze a corpus database

```bash
python singularitycalcia_v2_en.py --db calcia.db
```

### Analyze with custom parameters

```bash
python singularitycalcia_v2_en.py --db calcia.db --alpha 0.7 --ngram 3 --output ./results
```

### Re-analyze from a previous v1 CSV result

```bash
python singularitycalcia_v2_en.py --csv results_v1.csv
```

### Run only the synthetic corpus validation

```bash
python singularitycalcia_v2_en.py --synthetic
```

### Regenerate figures from existing CSV outputs

```bash
python datasets/generate_figures_en.py --data ./datasets --output ./figures
```

### Command-line arguments

| Argument      | Default             | Description                                 |
|---------------|---------------------|---------------------------------------------|
| `--db`        | --                  | Path to calcia.db                           |
| `--csv`       | --                  | Path to a previous v1 results CSV           |
| `--synthetic` | False               | Run only synthetic corpus validation        |
| `--alpha`     | 0.7                 | Originality weight (range: 0.0 to 1.0)      |
| `--ngram`     | 3                   | N-gram size for lexical analysis            |
| `--limit`     | None                | Limit number of documents (for quick tests) |
| `--output`    | ./resultados_SId_v2 | Output directory                            |

---

## Output Files

All outputs are written to the directory specified by `--output`. Pre-computed outputs for the full calcia.db corpus are available in the `datasets/` and `figures/` folders.

### CSV tables

| File                             | Description                                          |
|----------------------------------|------------------------------------------------------|
| `calcia_singularity_results.csv` | Full results for all documents (all columns)         |
| `calcia_top50.csv`               | Top 50 documents by SId v2                           |
| `calcia_bottom50.csv`            | Bottom 50 documents by SId v2 (excluding score = 0) |
| `calcia_deciles.csv`             | Summary statistics by SId v2 decile                  |
| `calcia_stats_comparison.csv`    | Descriptive statistics: SId v1 vs SId v2             |
| `calcia_correlaciones.csv`       | Pearson correlations with external variables         |

### Figures

Each figure is produced in both Spanish (`figura_N.png`) and English (`figura_N_en.png`) and stored in the `figures/` folder.

| Figure | Description                                                       |
|--------|-------------------------------------------------------------------|
| 1      | SId score distribution (histogram and normal fit)                 |
| 2      | Pearson correlations of SId with corpus variables                 |
| 3      | Relationship between SId and bibliometric impact                  |
| 4      | SId score distribution stratified by citation quartile            |
| 5      | Decomposition of SId into O_d (originality) and I_d (impact)     |
| 6      | Sensitivity analysis of SId to the alpha parameter               |
| 7      | Feature profile by SId decile                                     |
| 8      | Ranking of the 20 documents with highest and lowest SId           |
| 9      | Validation on the synthetic corpus (known ground truth)           |

### Report

The full analysis report is available at `reports/singularity_report.txt`. It includes descriptive statistics, Pearson correlations, a decile profile table, and the top and bottom 25 document rankings.

---

## Formal Specification

The mathematical formulation of the SId indicator is available in the `formula/` folder in both Spanish and English, as PDF and LaTeX source files. These documents include the complete derivation of the O_d and I_d components, the normalization procedure, and the rationale for the default parameter values.

---

## Synthetic Corpus Validation

The pipeline includes an internal validation procedure based on a manually constructed corpus of 12 documents organized into four categories with known ground-truth ordering:

- **Category A:** Highly cited, technically dense foundational papers (BERT, ResNet, Transformer).
- **Category B:** Medium-impact specialized studies in bibliometrics, cardiology, and document science.
- **Category C:** Low-impact, lexically redundant introductory texts.
- **Category D:** Short but highly cited abstracts from breakthrough papers.

Expected behavior: SId v2 should assign the highest scores to Category A, intermediate scores to Categories B and D (which involve a trade-off between originality and impact), and the lowest scores to Category C.

---

## Parameters and Recommendations

The default configuration (`alpha=0.7`, `ngram=3`) is recommended for scientific corpora in English or Spanish with full-text documents of at least 200 words. For short abstracts or highly technical corpora with limited citation metadata, consider reducing `alpha` to increase the weight of the impact component.

The alpha sensitivity analysis (Figure 6) should be consulted before applying SId to a new corpus, as the optimal balance between originality and impact varies by domain and document type.

---

## Authors

- Prof. Manuel Blazquez Ochando (manublaz@ucm.es)
- Prof. Juan Jose Prieto Gutierrez (jjpg@ucm.es)
- Prof. Maria Antonia Ovalle Perandones (maovalle@ucm.es)

Department of Library and Information Science, Complutense University of Madrid (UCM).

---

## Citation

If you use this software, dataset, or formula in academic research, please cite as follows:

> Blázquez-Ochando, M.; Ovalle-Perandones, M.A.; Prieto-Gutiérrez, J.J. (2026). *Documentary Singularity Indicator* [Software | Formula]. GitHub. https://github.com/manublaz/Document-Singularity-Indicator

---

## License

This software and dataset are distributed for academic and research use. Contact the authors for other uses.
