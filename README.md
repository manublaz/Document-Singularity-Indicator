# Calcia -- Documentary Singularity Indicator (SId)

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
singularitycalcia_v2.py          Main script: SId v2 analysis pipeline
singularitycalcia_v1.py          Original SId v1 implementation (reference)
singularitycalciaPARALLEL.py     Parallelized variant for large corpora
createCalciaDB.py                Utilities to create and populate calcia.db
pubmedcalcia.py                  PubMed document retrieval and ingestion
scholarcalcia.py                 Google Scholar document retrieval and ingestion
singularityEVAL_MASTER.py        Master evaluation script
singularityEVAL1.py              Evaluation module 1
singularityEVAL2.py              Evaluation module 2
singularityEVAL3.py              Evaluation module 3
singularityEVAL4.py              Evaluation module 4
singularityEVAL5.py              Evaluation module 5
singularityEVAL6.py              Evaluation module 6
singularityEVAL7.py              Evaluation module 7
fix.py                           Data repair utilities
requeriments.txt                 Python dependencies
README.md                        This file
```

---

## Database Schema

The main input is a SQLite database (`calcia.db`) with a `corpus` table structured as follows:

| Column         | Type    | Description                                  |
|---------------|---------|----------------------------------------------|
| id            | INTEGER | Primary key                                  |
| url           | TEXT    | Source URL                                   |
| title         | TEXT    | Document title                               |
| author        | TEXT    | Author(s)                                    |
| datepub       | TEXT    | Publication date                             |
| text          | TEXT    | Preprocessed full text                       |
| textoriginal  | TEXT    | Original full text (preferred if available)  |
| citations     | INTEGER | Citation count                               |
| created_at    | TEXT    | Ingestion timestamp                          |
| updated_at    | TEXT    | Last update timestamp                        |

---

## Installation

Python 3.9 or later is required.

```bash
pip install -r requeriments.txt
```

Core dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`.

---

## Usage

### Analyze a corpus database

```bash
python singularitycalcia_v2.py --db calcia.db
```

### Analyze with custom parameters

```bash
python singularitycalcia_v2.py --db calcia.db --alpha 0.7 --ngram 3 --output ./results
```

### Re-analyze from a previous v1 CSV result

```bash
python singularitycalcia_v2.py --csv results_v1.csv
```

### Run only the synthetic corpus validation

```bash
python singularitycalcia_v2.py --synthetic
```

### Command-line arguments

| Argument    | Default                  | Description                                      |
|------------|--------------------------|--------------------------------------------------|
| `--db`      | --                       | Path to calcia.db                                |
| `--csv`     | --                       | Path to a previous v1 results CSV                |
| `--synthetic` | False                  | Run only synthetic corpus validation             |
| `--alpha`   | 0.7                      | Originality weight (range: 0.0 to 1.0)           |
| `--ngram`   | 3                        | N-gram size for lexical analysis                 |
| `--limit`   | None                     | Limit number of documents (for quick tests)      |
| `--output`  | ./resultados_SId_v2      | Output directory                                 |

---

## Output Files

All outputs are written to the directory specified by `--output`.

### CSV tables

| File                            | Description                                           |
|--------------------------------|-------------------------------------------------------|
| `calcia_singularity_results.csv` | Full results for all documents (all columns)        |
| `calcia_top50.csv`              | Top 50 documents by SId v2                           |
| `calcia_bottom50.csv`           | Bottom 50 documents by SId v2 (excluding score = 0) |
| `calcia_deciles.csv`            | Summary statistics by SId v2 decile                 |
| `calcia_stats_comparison.csv`   | Descriptive statistics: SId v1 vs SId v2            |
| `calcia_correlaciones.csv`      | Pearson correlations with external variables         |

### Figures

| File                           | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `fig1_distribucion.png`        | Score distribution histograms: SId v1 vs SId v2         |
| `fig2_ceros.png`               | Key discriminatory quality metrics: SId v1 vs SId v2    |
| `fig3_correlaciones.png`       | Pearson correlations: SId v1 and SId v2 vs corpus vars  |
| `fig4_scatter_citas.png`       | SId v2 vs bibliometric impact (scatter and boxplot)     |
| `fig5_boxplot_cuartiles.png`   | Score distribution by citation quartile                  |
| `fig6_componentes.png`         | O_d and I_d component decomposition                     |
| `fig7_alpha_sensibilidad.png`  | Sensitivity analysis of the alpha parameter             |
| `fig8_top_bottom.png`          | Top and bottom ranked documents by SId v2               |
| `fig9_deciles_perfil.png`      | Feature profile by SId v2 decile                        |
| `fig10_validacion_sintetica.png` | Synthetic corpus validation (known ground truth)      |

### Report

| File                     | Description                                         |
|-------------------------|-----------------------------------------------------|
| `singularity_report.txt` | Full analysis report with statistics, correlations, and rankings |

---

## Synthetic Corpus Validation

The script includes an internal validation procedure based on a manually constructed corpus of 12 documents organized into four categories with known ground-truth ordering:

- **Category A:** Highly cited, technically dense foundational papers (BERT, ResNet, Transformer).
- **Category B:** Medium-impact specialized studies in bibliometrics, cardiology, and document science.
- **Category C:** Low-impact, lexically redundant introductory texts.
- **Category D:** Short but highly cited abstracts from breakthrough papers.

Expected behavior: SId v2 should assign the highest scores to Category A, intermediate scores to Categories B and D (which involve a trade-off between originality and impact), and the lowest scores to Category C.

---

## Parameters and Recommendations

The default configuration (`alpha=0.7`, `ngram=3`) is recommended for scientific corpora in English or Spanish with full-text documents of at least 200 words. For short abstracts or highly technical corpora with limited citation metadata, consider reducing `alpha` to increase the weight of the impact component.

The `alpha` sensitivity analysis (Figure 7) should be consulted before applying SId to a new corpus, as the optimal balance between originality and impact varies by domain and document type.

---

## Authors

- Prof. Manuel Blazquez Ochando (manublaz@ucm.es)
- Prof. Juan Jose Prieto Gutierrez (jjpg@ucm.es)
- Prof. Maria Antonia Ovalle Perandones (maovalle@ucm.es)

Department of Library and Information Science, Complutense University of Madrid (UCM).

Formula analysis and improvement: Claude Sonnet (Anthropic), 2025.

---

## Citation

If you use this software in academic research, please cite the corresponding publication (in preparation). Until formal publication, reference this repository directly.

---

## License

This software is distributed for academic and research use. Contact the authors for other uses.
