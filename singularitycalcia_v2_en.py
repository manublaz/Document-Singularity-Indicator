#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
singularitycalcia_v2.py  --  SId v2.0
======================================
Documentary Singularity Indicator for AI corpus selection

Autores:
  Prof. Manuel Blazquez Ochando  (manublaz@ucm.es)
  Prof. Juan Jose Prieto Gutierrez (jjpg@ucm.es)
  Prof. Maria Antonia Ovalle Perandones (maovalle@ucm.es)
  Dpto. Biblioteconomia y Documentacion -- UCM

Analisis critico y mejora de formula:
  Claude Sonnet (Anthropic), 2025

Estructura de calcia.db -- tabla corpus:
  id, url, title, author, datepub, text, textoriginal, citations,
  created_at, updated_at

Uso:
  python singularitycalcia_v2.py --db calcia.db
  python singularitycalcia_v2.py --db calcia.db --alpha 0.7 --ngram 3 --output ./resultados
  python singularitycalcia_v2.py --csv resultados_v1.csv
  python singularitycalcia_v2.py --synthetic

Generated outputs (in --output DIR):
  calcia_singularity_results.csv   -- tabla completa (todos los documentos)
  calcia_top50.csv                 -- top-50 by SId v2
  calcia_bottom50.csv              -- bottom-50 (excl. score=0)
  calcia_deciles.csv               -- estadisticas por decil
  calcia_stats_comparison.csv      -- comparativa v1 vs v2
  calcia_correlaciones.csv         -- correlaciones con variables externas
  fig1_distribucion.png            -- histogramas v1 vs v2
  fig2_ceros.png                   -- metricas de calidad clave
  fig3_correlaciones.png           -- correlaciones v1 vs v2
  fig4_scatter_citas.png           -- SId v2 vs citations
  fig5_boxplot_cuartiles.png       -- boxplot por cuartil de citas
  fig6_componentes.png             -- descomposicion O_d / I_d
  fig7_alpha_sensibilidad.png      -- sensibilidad al parametro alpha
  fig8_top_bottom.png              -- ranking top/bottom documentos
  fig9_deciles_perfil.png          -- perfil por decil
  fig10_validacion_sintetica.png   -- validacion corpus sintetico
  singularity_report.txt           -- reporte completo
"""

import re
import math
import sqlite3
import os
import sys
import argparse
from collections import Counter
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. CONSTANTES
# ---------------------------------------------------------------------------

DEFAULT_NGRAM  = 3
DEFAULT_ALPHA  = 0.7
DEFAULT_BETA   = 0.1
MIN_TEXT_LEN   = 30
REPORT_TOP_N   = 50

COLORS = {
    'v1'     : '#6b9ec8',
    'v2'     : '#c0392b',
    'accent' : '#d4a017',
    'deep'   : '#0d1b35',
    'mid'    : '#2d4a7a',
    'light'  : '#e8f0f7',
    'grid'   : '#d0dbe8',
    'green'  : '#27ae60',
    'orange' : '#e67e22',
}

# ---------------------------------------------------------------------------
# 2. STOPWORDS
# ---------------------------------------------------------------------------

STOPWORDS = set("""
the a an and or but in on at to for of with by from is are was were be been
being have has had do does did will would could should may might shall can not
this that these those it its we our us they their them you your he she his her
i my as up if so no than then also into about through which who what when where
how all each both few more most other some such only own same very just because
while although however therefore thus hence since during between after before
above below over under again further here there once any itself themselves
el la los las un una unos unas de del al y e o u pero sino si en con por para
que se su sus le les me mi mis te tu tus nos es son fue ser estar ha han este
esta este ese esa como mas tambien entre donde cuando aunque porque sin sobre
results study method patients data analysis using used significant showed show
found total groups group compared comparison table figure included including
associated association based methods conclusions objective background purpose
conclusion introduction discussion however therefore respectively p vs eg ie
et al ci hr or doi pmc pmid abstract keywords
""".split())

# ---------------------------------------------------------------------------
# 3. PREPROCESAMIENTO
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    if not text or not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\b\d+([.,]\d+)?\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split()
             if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)


def extract_ngrams(text: str, n: int) -> List[str]:
    words = text.split()
    if len(words) < n:
        return []
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


# ---------------------------------------------------------------------------
# 4. CARGA DESDE calcia.db
# ---------------------------------------------------------------------------

def load_from_db(db_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Carga documentos desde tabla corpus de calcia.db."""
    if not os.path.exists(db_path):
        print(f'  ERROR: File not found: {db_path}')
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("PRAGMA table_info(corpus)")
    cols = {row[1] for row in cur.fetchall()}
    print(f'  Detected columns: {sorted(cols)}')

    # Usar textoriginal si existe y tiene contenido, sino text
    if 'textoriginal' in cols:
        text_expr = "COALESCE(NULLIF(TRIM(textoriginal),''), text)"
    else:
        text_expr = 'text'

    sql = f"""
        SELECT
            CAST(id AS TEXT)            AS id,
            COALESCE(title,   '')       AS title,
            COALESCE(author,  '')       AS author,
            COALESCE(datepub, '')       AS datepub,
            {text_expr}                 AS text,
            COALESCE(citations, 0)      AS citations
        FROM corpus
        WHERE text IS NOT NULL AND LENGTH(TRIM(text)) >= {MIN_TEXT_LEN}
        ORDER BY id
    """
    if limit:
        sql += f' LIMIT {limit}'

    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()

    docs = []
    skipped = 0
    for row in rows:
        doc_id, title, author, datepub, text, citations = row
        if not text or len(text.strip()) < MIN_TEXT_LEN:
            skipped += 1
            continue
        docs.append({
            'id'       : str(doc_id),
            'title'    : (title or f'Doc_{doc_id}')[:120],
            'author'   : author or '',
            'datepub'  : datepub or '',
            'text'     : text,
            'citations': max(0, int(citations or 0)),
        })

    print(f'  Valid documents: {len(docs):,}  '
          f'(skipped due to short text: {skipped})')
    return docs


def db_summary(db_path: str) -> None:
    """Imprime estadisticas de la base de datos."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM corpus")
    total = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM corpus WHERE LENGTH(TRIM(COALESCE(text,''))) >= {MIN_TEXT_LEN}")
    with_text = cur.fetchone()[0]
    cur.execute("SELECT MIN(citations), MAX(citations), AVG(citations) FROM corpus WHERE citations > 0")
    row = cur.fetchone()
    cmin, cmax, cavg = (row if row[0] else (0, 0, 0.0))
    cur.execute("SELECT COUNT(*) FROM corpus WHERE citations > 0")
    with_cit = cur.fetchone()[0]
    conn.close()
    print(f'\n  {"=" * 52}')
    print(f'  SUMMARY OF calcia.db')
    print(f'  {"=" * 52}')
    print(f'  Total records         : {total:,}')
    print(f'  With sufficient text  : {with_text:,}')
    print(f'  With citations > 0    : {with_cit:,}')
    if cmax:
        print(f'  Citations min/max/avg : {cmin} / {cmax} / {cavg:.1f}')
    print(f'  {"=" * 52}\n')


# ---------------------------------------------------------------------------
# 5. BARRA DE PROGRESO (sin dependencias externas)
# ---------------------------------------------------------------------------

def _progress(i: int, n: int, label: str = '', bar_len: int = 30,
              t_start: float = None) -> None:
    """Imprime barra de progreso en la misma linea."""
    import time
    pct   = i / n
    filled = int(bar_len * pct)
    bar   = '#' * filled + '-' * (bar_len - filled)
    eta   = ''
    if t_start and i > 0:
        elapsed = time.time() - t_start
        remaining = elapsed / pct * (1 - pct)
        if remaining < 60:
            eta = f'  ETA: {remaining:.0f}s'
        else:
            eta = f'  ETA: {remaining/60:.1f}m'
    print(f'\r    [{bar}] {i:>5}/{n}  {pct*100:>5.1f}%  {label}{eta}   ',
          end='', flush=True)
    if i == n:
        print()


# ---------------------------------------------------------------------------
# 5. IS_d v1  (implementacion fiel — optimizada para corpus grande)
# ---------------------------------------------------------------------------
#
# OPTIMIZACIONES respecto a la implementacion ingenua:
#
#  1. doc_count (cdi) precomputado en build() → O(1) por n-grama
#     (la version original recalculaba sum() sobre todo el corpus en cada _si)
#
#  2. fknn aproximado eficientemente via indice de palabras:
#     Para cada n-grama i, el vecino mas proximo es aquel n-grama j del mismo
#     documento que comparte el mayor numero de palabras con i.
#     Se precomputa un indice word→{ngrams} para localizar candidatos
#     sin iterar sobre todos los n-gramas del documento.
#     Esto reduce la complejidad de O(k²) a O(k·overlap) por documento.
#
#  3. Todos los fknn del documento se calculan en un solo paso (score_doc_fast)
#     antes de calcular los scores, evitando llamadas repetidas.
#
#  Resultado: corpus de 7.500 docs con n=3 termina en ~3-8 minutos
#  (frente a horas en la version O(k²) pura).

class SingularityV1:
    """
    IS_d v1 original (implementacion optimizada para corpus grande).

    IS_d = (sum_i(Si/fi) + beta*log(1+Cd)) / (sum_i(Si) + beta*log(1+Cmax))
    Si = (1/(1+log(1+fi+fknn))) * IDF_i
    N = n-gramas del documento cuyo vecino lexico mas proximo tiene fknn >= 2
    """

    def __init__(self, n: int = 3, beta: float = 0.1):
        self.n           = n
        self.beta        = beta
        self.C           = 0
        self.corpus_ng:  Counter = Counter()   # fi_corpus
        self.doc_count:  Counter = Counter()   # cdi (precomputado)
        self.doc_ng: Dict[str, Counter] = {}   # fi_doc

    def build(self, documents: List[Dict]) -> None:
        import time
        t0 = time.time()
        print('  Indexing corpus (v1)...')
        n = len(documents)
        self.C         = n
        self.corpus_ng = Counter()
        self.doc_count = Counter()
        self.doc_ng    = {}
        for i, doc in enumerate(documents, 1):
            if i % 500 == 0 or i == n:
                _progress(i, n, 'indexando')
            did  = doc['id']
            txt  = preprocess(doc['text'])
            ngs  = extract_ngrams(txt, self.n)
            cntr = Counter(ngs)
            self.doc_ng[did] = cntr
            self.corpus_ng.update(ngs)
            for ng in cntr:
                self.doc_count[ng] += 1
        print(f'  Index built in {time.time()-t0:.1f}s  '
              f'| unique n-grams: {len(self.corpus_ng):,}')

    def _fknn_doc(self, did: str) -> Dict[str, float]:
        """
        Calcula fknn para todos los n-gramas de un documento en un solo paso.
        Devuelve dict {ngram: fknn_value}.

        Estrategia eficiente:
          Construye indice word -> lista de (ngram, freq) presentes en el doc.
          Para cada n-grama i, busca candidatos a vecino en la interseccion
          de los indices de sus palabras. Solo evalua esos candidatos (no todos).
        """
        dngs = self.doc_ng.get(did, {})
        if not dngs:
            return {}

        # Indice: palabra -> lista de ngrams del doc que contienen esa palabra
        word_idx: Dict[str, List[str]] = {}
        for ng in dngs:
            for w in ng.split():
                word_idx.setdefault(w, []).append(ng)

        fknn_map: Dict[str, float] = {}
        for ng, freq_i in dngs.items():
            words_i   = ng.split()
            # Candidatos: n-gramas del doc que comparten al menos 1 palabra
            candidates: Counter = Counter()
            for w in words_i:
                for cand in word_idx.get(w, []):
                    if cand != ng:
                        candidates[cand] += 1

            best_sim = best_freq = 0.0
            set_i = set(words_i)
            for cand, shared in candidates.items():
                set_o = set(cand.split())
                union = set_i | set_o
                sim   = shared / len(union) if union else 0.0
                if sim > best_sim:
                    best_sim  = sim
                    best_freq = float(dngs[cand])

            fknn_map[ng] = best_freq

        return fknn_map

    def score_doc(self, did: str, citations: int, max_cit: int) -> Dict:
        dngs     = self.doc_ng.get(did, {})
        fknn_map = self._fknn_doc(did)

        # N = n-gramas cuyo vecino mas proximo tiene fknn >= 2
        qual = [ng for ng in dngs if fknn_map.get(ng, 0.0) >= 2.0]

        sum_si_fi = sum_si = 0.0
        for ng in qual:
            fi   = self.corpus_ng.get(ng, 0)
            cdi  = self.doc_count.get(ng, 1)
            if fi == 0 or cdi == 0 or self.C <= cdi:
                continue
            idf   = math.log(self.C / cdi)
            fknn  = fknn_map.get(ng, 0.0)
            denom = 1.0 + math.log(1.0 + fi + fknn)
            si    = (1.0 / denom) * idf
            if si > 0 and fi > 0:
                sum_si_fi += si / fi
                sum_si    += si

        num = sum_si_fi + self.beta * math.log(1.0 + citations)
        den = sum_si    + self.beta * math.log(1.0 + max_cit)
        return {
            'score_v1' : num / den if den > 0 else 0.0,
            'N_qual_v1': len(qual),
            'N_total'  : len(dngs),
        }

    def analyze(self, documents: List[Dict]) -> pd.DataFrame:
        import time
        self.build(documents)
        max_cit = max(d['citations'] for d in documents)
        print('  Computing SId v1...')
        rows  = []
        n     = len(documents)
        t0    = time.time()
        for i, doc in enumerate(documents, 1):
            if i % 50 == 0 or i == n:
                _progress(i, n, 'SId v1', t_start=t0)
            r = self.score_doc(doc['id'], doc['citations'], max_cit)
            rows.append({
                'document_id' : doc['id'],
                'title'       : doc['title'],
                'author'      : doc.get('author', ''),
                'datepub'     : doc.get('datepub', ''),
                'citations'   : doc['citations'],
                'score_v1'    : r['score_v1'],
                'N_qual_v1'   : r['N_qual_v1'],
                'N_total'     : r['N_total'],
                'text_length' : len(doc['text']),
            })
        elapsed = time.time() - t0
        print(f'  SId v1 completed in {elapsed:.1f}s  '
              f'({elapsed/n*1000:.1f}ms/doc)')
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. IS_d v2  (formula mejorada)
# ---------------------------------------------------------------------------
#
# IS_d_v2 = alpha * O_d  +  (1-alpha) * I_d
#
# O_d = sum_i [ Si * log(1 + fi_doc) ]
#       / [ |Omega| * log(1 + Td/|Omega|)  +  eps ]
#
# Si  = IDF_suav(i) * decay(fi_corpus)
#     = log((C+1)/(cdi+0.5))  *  1/(1+log(1+fi_corpus))
#
# O_d normalizado a [0,1] usando P99 del corpus como techo.
#
# I_d = log(1+Cd) / log(1+Cmax)
#
# Ventajas sobre v1:
#   - Sin filtro restrictivo: todos los n-gramas contribuyen (no solo N con fknn>=2)
#   - Documentos cortos reciben scores > 0
#   - O_d e I_d ambos en [0,1]: score bien calibrado
#   - alpha controla explicitamente el balance originalidad/impacto
#   - El divisor log(1+Td/|Omega|) penaliza redundancia interna
#   - IDF suavizado es mas robusto frente a vocabulario exotico

class SingularityV2:

    def __init__(self, n: int = 3, alpha: float = 0.7):
        self.n          = n
        self.alpha      = alpha
        self.C          = 0
        self.corpus_ng: Counter = Counter()
        self.doc_count: Counter = Counter()
        self.doc_ng: Dict[str, Counter] = {}

    def build(self, documents: List[Dict]) -> None:
        print('  Building SId v2 index...')
        self.C         = len(documents)
        self.corpus_ng = Counter()
        self.doc_count = Counter()
        self.doc_ng    = {}
        for doc in documents:
            did  = doc['id']
            txt  = preprocess(doc['text'])
            ngs  = extract_ngrams(txt, self.n)
            cntr = Counter(ngs)
            self.doc_ng[did] = cntr
            self.corpus_ng.update(ngs)
            for ng in cntr:
                self.doc_count[ng] += 1
        print(f'    Corpus: {self.C:,} docs  |  '
              f'unique n-grams: {len(self.corpus_ng):,}  |  '
              f'total instances: {sum(self.corpus_ng.values()):,}')

    def _si(self, ngram: str) -> float:
        fi  = self.corpus_ng.get(ngram, 0)
        if fi == 0:
            return 0.0
        cdi = self.doc_count.get(ngram, 1)
        idf = math.log((self.C + 1.0) / (cdi + 0.5))
        if idf <= 0:
            return 0.0
        decay = 1.0 / (1.0 + math.log(1.0 + fi))
        return idf * decay

    def _Od_raw(self, did: str) -> float:
        dngs = self.doc_ng.get(did, {})
        if not dngs:
            return 0.0
        Omega  = len(dngs)
        Td     = sum(dngs.values())
        num    = sum(self._si(ng) * math.log(1.0 + freq)
                     for ng, freq in dngs.items())
        redund = math.log(1.0 + Td / max(Omega, 1))
        den    = Omega * redund + 1e-6
        return num / den

    def score_doc(self, did: str, citations: int, max_cit: int,
                  p99_O: float) -> Dict:
        Od_raw = self._Od_raw(did)
        Od     = min(Od_raw / p99_O, 1.0) if p99_O > 0 else 0.0
        Id     = (math.log(1.0 + citations) / math.log(1.0 + max_cit)
                  if max_cit > 0 else 0.0)
        dngs   = self.doc_ng.get(did, {})
        return {
            'Od_raw'   : Od_raw,
            'Od'       : Od,
            'Id'       : Id,
            'score_v2' : self.alpha * Od + (1.0 - self.alpha) * Id,
            'N_total_v2': len(dngs),
            'T_total_v2': sum(dngs.values()),
        }

    def analyze(self, documents: List[Dict]) -> pd.DataFrame:
        import time
        self.build(documents)
        max_cit = max(d['citations'] for d in documents)

        # Paso 1: calcular O_d_raw para todos -> obtener P99
        print('  Computing O_d_raw (1/2)...')
        raw = []
        n   = len(documents)
        t0  = time.time()
        for i, doc in enumerate(documents, 1):
            if i % 100 == 0 or i == n:
                _progress(i, n, 'O_d_raw', t_start=t0)
            raw.append(self._Od_raw(doc['id']))
        p99_O = float(np.percentile(raw, 99))
        print(f'    P99(O_d_raw) = {p99_O:.6f}  | time: {time.time()-t0:.1f}s')

        # Paso 2: scores finales
        print('  Computing SId v2 (2/2)...')
        rows = []
        t0   = time.time()
        for i, (doc, od_raw) in enumerate(zip(documents, raw), 1):
            if i % 100 == 0 or i == n:
                _progress(i, n, 'SId v2', t_start=t0)
            r = self.score_doc(doc['id'], doc['citations'], max_cit, p99_O)
            rows.append({
                'document_id' : doc['id'],
                'title'       : doc['title'],
                'author'      : doc.get('author', ''),
                'datepub'     : doc.get('datepub', ''),
                'citations'   : doc['citations'],
                'Od_raw'      : od_raw,
                'Od'          : r['Od'],
                'Id'          : r['Id'],
                'score_v2'    : r['score_v2'],
                'N_total_v2'  : r['N_total_v2'],
                'T_total_v2'  : r['T_total_v2'],
                'text_length' : len(doc['text']),
            })
        print(f'  SId v2 completed in {time.time()-t0:.1f}s')
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. CORPUS SINTETICO DE VALIDACION
# ---------------------------------------------------------------------------

SYNTHETIC_DOCS = [
    {'id':'A01','citations':12300,'category':'A',
     'title':'BERT: Pre-training deep bidirectional transformers',
     'text':"""Bidirectional Encoder Representations from Transformers introduces novel
language model pre-training. BERT pre-trains deep bidirectional representations from
unlabeled text jointly conditioning on left and right context across all layers.
The pre-trained model fine-tuned with one additional output layer creates state-of-the-art
models for question answering language inference and text classification without substantial
task-specific architecture modifications. We demonstrate bidirectional pre-training superiority
through ablation experiments. BERT achieves new state-of-the-art results on eleven natural
language processing tasks improving GLUE score to 80.5 percent MultiNLI accuracy to 86.7
percent SQuAD test F1 to 93.2 percent advancing numerous benchmarks substantially.
The masked language model objective enables unprecedented bidirectional pre-training
unlike all prior autoregressive language model approaches."""},

    {'id':'A02','citations':8500,'category':'A',
     'title':'Deep residual learning for image recognition',
     'text':"""Deeper neural networks are more difficult to train. We present residual
learning framework to ease training of substantially deeper networks. We explicitly reformulate
layers as learning residual functions with reference to layer inputs instead of learning
unreferenced functions. We provide comprehensive empirical evidence showing residual networks
are easier to optimize and gain accuracy from considerably increased depth. On ImageNet we
evaluate residual nets with depth up to 152 layers eight times deeper than VGG nets with
lower complexity. An ensemble achieves 3.57 percent error on ImageNet test set winning
first place on ILSVRC 2015 classification task. We also present analysis on CIFAR-10
with 100 and 1000 layers demonstrating depth importance for visual recognition tasks."""},

    {'id':'A03','citations':6700,'category':'A',
     'title':'Attention is all you need: the transformer architecture',
     'text':"""The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks including encoder and decoder. Best performing models connect
encoder and decoder through attention mechanism. We propose the Transformer architecture
based solely on attention mechanisms dispensing with recurrence and convolutions entirely.
Experiments on two machine translation tasks show these models superior in quality while
more parallelizable requiring significantly less training time. Our model achieves 28.4 BLEU
on WMT 2014 English-to-German translation improving over existing best results including
ensembles by over 2 BLEU. On WMT 2014 English-to-French our model establishes new
single-model state-of-the-art BLEU score of 41.0 after training 3.5 days on eight GPUs."""},

    {'id':'B01','citations':850,'category':'B',
     'title':'Bibliometric analysis of artificial intelligence in cardiology',
     'text':"""This systematic bibliometric analysis examines artificial intelligence
applications in cardiovascular medicine from 2010 to 2023 using Web of Science and Scopus.
We identified 8742 publications through controlled vocabulary search strategies combining
medical subject headings with computational terminology. Co-citation network analysis reveals
three primary research clusters: echocardiographic image segmentation, electrocardiogram
interpretation algorithms, and clinical outcome prediction models. Convolutional neural
networks dominate the methodological landscape representing 58 percent of machine learning
approaches followed by recurrent architectures at 19 percent. Geographic concentration
indicates United States China and United Kingdom account for 67 percent of total output.
Interdisciplinary collaboration networks demonstrate emerging partnerships between cardiology
departments and computer science faculties particularly in academic medical centers."""},

    {'id':'B02','citations':420,'category':'B',
     'title':'Document selection methodology for AI training corpora',
     'text':"""The construction of domain-specific corpora for artificial intelligence systems
requires methodological frameworks extending beyond traditional library acquisition criteria.
Established evaluation dimensions of authority accuracy currency coverage and objectivity
must be reinterpreted within computational learning contexts where document contribution
is measured by informational singularity rather than individual merit. We propose an integrated
selection framework incorporating textual originality metrics derived from n-gram frequency
distributions citation impact normalization thematic coverage stratification and temporal
weighting. Case studies from biomedical legal and engineering specialized libraries demonstrate
operational deployment with empirical validation. Our findings indicate diversity metrics
must complement quality indicators to prevent systematic topical bias in trained generative
models. Controlled vocabulary alignment using ontological structures improves thematic
representation considerably in specialized knowledge domain applications."""},

    {'id':'B03','citations':580,'category':'B',
     'title':'Echocardiography appropriateness in outpatient cardiology',
     'text':"""This prospective multicenter study evaluated 2110 echocardiogram prescriptions
across thirteen Italian regions to assess appropriateness utility and pertinence. Appropriateness
was graded according to Italian Federation of Cardiology guidelines as appropriate uncertain or
inappropriate class. Results demonstrated 54 percent appropriate 30 percent uncertain appropriateness
and 16 percent inappropriate. Echocardiograms prescribed by cardiologists showed significantly
superior appropriateness utility and pertinence compared to general practitioners who accounted
for 56 percent of all prescriptions. Hypertension was the most frequent indication at 22.9 percent
followed by asymptomatic screening at 16.8 percent showing particularly poor appropriateness
metrics. These findings highlight critical importance of cardiologist-led clinical management
for optimizing echocardiographic resource utilization and reducing inappropriate examinations."""},

    {'id':'C01','citations':12,'category':'C',
     'title':'Overview of machine learning methods',
     'text':"""Machine learning is a type of artificial intelligence. There are many machine
learning methods. Supervised learning uses labeled data. Unsupervised learning does not use
labels. Reinforcement learning uses rewards. Machine learning methods include decision trees
random forests neural networks and support vector machines. These methods can be applied to
many different problems. Neural networks are inspired by the brain. Deep learning is a type
of neural network with many layers. Machine learning has many applications in many fields.
Future research will develop better machine learning methods. More research is needed."""},

    {'id':'C02','citations':5,'category':'C',
     'title':'Introduction to deep learning and neural networks',
     'text':"""Deep learning is a form of machine learning. It uses neural networks with
many layers. Neural networks are made of nodes and connections. Each layer processes input
and passes it to the next layer. Deep learning has been used for image recognition and speech
recognition and natural language processing. These are important tasks. Deep learning requires
large amounts of data and powerful computers. There are many types of neural networks.
Convolutional networks are used for images. Recurrent networks are used for sequences.
Transformer networks are newer and very powerful. More applications are being developed."""},

    {'id':'C03','citations':8,'category':'C',
     'title':'Applications of artificial intelligence in healthcare',
     'text':"""Artificial intelligence can help in healthcare. It can help doctors make better
decisions. Artificial intelligence can analyze medical images. It can also analyze patient data.
Artificial intelligence has many applications in medicine including diagnosis and treatment
planning. Artificial intelligence can help with drug discovery and patient monitoring.
There are challenges in using artificial intelligence in healthcare. Data privacy is important.
So is data quality. Artificial intelligence systems need to be validated carefully to ensure
safety and effectiveness. Healthcare professionals must be trained to use artificial intelligence
tools effectively. The future of artificial intelligence in healthcare looks promising."""},

    {'id':'D01','citations':3200,'category':'D',
     'title':'Quantum error correction at fault-tolerant scale',
     'text':"""Fault-tolerant quantum computation demonstrated using surface codes achieving
logical error suppression exceeding hundredfold below physical threshold. Cryptographic
implications profound for post-quantum security infrastructure and optimization problems."""},

    {'id':'D02','citations':1800,'category':'D',
     'title':'mRNA vaccine efficacy against SARS-CoV-2 variants',
     'text':"""Phase III randomized trial demonstrates 94.1 percent efficacy against severe
COVID-19 with lipid nanoparticle mRNA delivery. Cross-variant neutralization maintained
against Omicron subvariants through heterologous booster administration protocol."""},

    {'id':'D03','citations':45,'category':'D',
     'title':'Chinchilla scaling laws for language model training',
     'text':"""Empirical analysis confirms power-law scaling between computational budget
and language model performance. Optimal training allocates equal resources to parameters
and training tokens, contradicting previous large-model paradigms."""},
]


def analyze_synthetic(n: int = 3, alpha: float = 0.7,
                       output_dir: str = '.') -> pd.DataFrame:
    print('\n' + '-'*60)
    print('VALIDATION -- SYNTHETIC CORPUS (known ground truth)')
    print('-'*60)

    docs    = [{'id': d['id'], 'title': d['title'],
                'text': d['text'], 'citations': d['citations'],
                'author': '', 'datepub': ''}
               for d in SYNTHETIC_DOCS]
    cat_map = {d['id']: d['category'] for d in SYNTHETIC_DOCS}

    cv1 = SingularityV1(n=n, beta=DEFAULT_BETA)
    df1 = cv1.analyze(docs)

    cv2 = SingularityV2(n=n, alpha=alpha)
    df2 = cv2.analyze(docs)

    df = df1.merge(df2[['document_id','Od','Id','score_v2',
                         'N_total_v2','T_total_v2']],
                   on='document_id')
    df['category'] = df['document_id'].map(cat_map)
    df = df.sort_values('score_v2', ascending=False).reset_index(drop=True)

    print('\n  Resultados:')
    print(f"  {'ID':<6} {'Cat':>3} {'Citations':>10} {'v1':>9} {'v2':>9} "
          f"{'O_d':>7} {'I_d':>7}  Title")
    print('  ' + '-'*80)
    for _, r in df.iterrows():
        t = r['title'][:40]
        print(f"  {r['document_id']:<6} {r['category']:>3} "
              f"{r['citations']:>7,}  {r['score_v1']:>8.4f}  {r['score_v2']:>8.4f} "
              f"  {r['Od']:>6.4f}  {r['Id']:>6.4f}  {t}")

    _plot_synthetic(df, os.path.join(output_dir, 'fig10_validacion_sintetica.png'))
    return df


# ---------------------------------------------------------------------------
# 8. ESTADISTICAS
# ---------------------------------------------------------------------------

def compute_stats(s: pd.Series, label: str = '') -> Dict:
    return {
        'label'   : label,
        'n_total' : int(len(s)),
        'n_zero'  : int((s == 0).sum()),
        'pct_zero': float((s == 0).mean() * 100),
        'n_nonzero': int((s > 0).sum()),
        'mean'    : float(s.mean()),
        'std'     : float(s.std()),
        'min'     : float(s.min()),
        'p10'     : float(s.quantile(0.10)),
        'q1'      : float(s.quantile(0.25)),
        'median'  : float(s.median()),
        'q3'      : float(s.quantile(0.75)),
        'p90'     : float(s.quantile(0.90)),
        'p99'     : float(s.quantile(0.99)),
        'max'     : float(s.max()),
        'iqr'     : float(s.quantile(0.75) - s.quantile(0.25)),
        'skewness': float(s.skew()),
        'kurtosis': float(s.kurt()),
    }


def compute_correlations(df: pd.DataFrame, score_col: str) -> Dict:
    s    = df[score_col]
    corr = {}
    mapping = {
        'log(1+citations)': np.log1p(df['citations'].clip(0)),
        'Raw citations': df['citations'],
    }
    if 'text_length' in df.columns:
        mapping['Text length']      = df['text_length']
    if 'N_total' in df.columns:
        mapping['Unique n-grams (v1)']  = df['N_total']
    if 'N_total_v2' in df.columns:
        mapping['Unique n-grams (v2)']  = df['N_total_v2']
    for label, x in mapping.items():
        try:
            corr[label] = round(float(s.corr(x)), 4)
        except Exception:
            pass
    return corr


def decile_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df2 = df[df[score_col] > 0].copy()
    if len(df2) < 10:
        return pd.DataFrame()
    try:
        df2['decile'] = pd.qcut(df2[score_col], q=10,
                                 labels=[f'D{i}' for i in range(1, 11)],
                                 duplicates='drop')
    except Exception:
        return pd.DataFrame()
    agg = df2.groupby('decile', observed=True).agg(
        n              = (score_col,    'count'),
        score_mean     = (score_col,    'mean'),
        score_min      = (score_col,    'min'),
        score_max      = (score_col,    'max'),
        citas_median   = ('citations',  'median'),
        citas_mean     = ('citations',  'mean'),
        text_len_mean  = ('text_length','mean'),
        ngrams_uniq    = ('N_total_v2' if 'N_total_v2' in df2.columns
                          else 'N_total', 'mean'),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# 9. FIGURAS
# ---------------------------------------------------------------------------

def _ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#f8f9fb')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['deep'], labelsize=8.5)
    ax.grid(axis='y', color=COLORS['grid'], lw=0.8, ls='--')
    if title:
        ax.set_title(title, fontsize=10.5, fontweight='bold',
                     color=COLORS['deep'], pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=COLORS['deep'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=COLORS['deep'])


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {os.path.basename(path)}')


def plot_histogramas(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 1 -- SId score distribution: v1 vs v2',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    bins = np.linspace(0, 1, 45)
    for ax, col, color, label in [
        (axes[0], 'score_v1', COLORS['v1'], 'SId v1 (original)'),
        (axes[1], 'score_v2', COLORS['v2'], 'SId v2 (improved)'),
    ]:
        s = df[col]
        ax.hist(s, bins=bins, color=color, alpha=0.85, edgecolor='white', lw=0.3)
        ax.axvline(s.median(), color=COLORS['deep'], lw=1.5, ls='--',
                   label=f'Median={s.median():.3f}')
        ax.axvline(s.mean(),   color=COLORS['accent'], lw=1.5, ls=':',
                   label=f'Mean={s.mean():.3f}')
        _ax(ax, label, 'SId Score', 'Frequency')
        ax.legend(fontsize=8)
        pct0 = (s == 0).mean() * 100
        ax.text(0.97, 0.95, f'{pct0:.1f}% = 0', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, fontweight='bold',
                color=COLORS['v2'])
    plt.tight_layout()
    _save(fig, out)


def plot_ceros_y_iqr(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 2 -- Discriminatory quality metrics: SId v1 vs SId v2',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    log_cit = np.log1p(df['citations'].clip(0))
    metrics = [
        ('% docs score=0',
         [(df['score_v1']==0).mean()*100, (df['score_v2']==0).mean()*100]),
        ('Score IQR',
         [df['score_v1'].quantile(.75)-df['score_v1'].quantile(.25),
          df['score_v2'].quantile(.75)-df['score_v2'].quantile(.25)]),
        ('r with log(1+citations)',
         [df['score_v1'].corr(log_cit), df['score_v2'].corr(log_cit)]),
    ]
    for ax, (titulo, vals) in zip(axes, metrics):
        bars = ax.bar(['SId v1','SId v2'], vals,
                      color=[COLORS['v1'], COLORS['v2']],
                      width=0.5, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + abs(max(vals))*0.04,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color=COLORS['deep'])
        _ax(ax, titulo, 'Version', titulo)
        lim = max(abs(v) for v in vals)
        ax.set_ylim(min(0, min(vals))*1.3, lim*1.4)
    plt.tight_layout()
    _save(fig, out)


def plot_correlaciones(df, out):
    variables = {'log(1+citations)': np.log1p(df['citations'].clip(0))}
    if 'text_length' in df.columns:
        variables['Text length'] = df['text_length']
    col_n = 'N_total_v2' if 'N_total_v2' in df.columns else 'N_total'
    if col_n in df.columns:
        variables['Unique n-grams'] = df[col_n]

    labels  = list(variables.keys())
    v1c     = [df['score_v1'].corr(x) for x in variables.values()]
    v2c     = [df['score_v2'].corr(x) for x in variables.values()]
    x       = np.arange(len(labels))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    b1 = ax.bar(x-w/2, v1c, w, label='SId v1',
                color=COLORS['v1'], alpha=0.9, edgecolor='white')
    b2 = ax.bar(x+w/2, v2c, w, label='SId v2',
                color=COLORS['v2'], alpha=0.9, edgecolor='white')
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2,
                    h + (0.01 if h >= 0 else -0.025),
                    f'{h:.3f}', ha='center',
                    va='bottom' if h >= 0 else 'top',
                    fontsize=9, color=COLORS['deep'])
    ax.axhline(0, color=COLORS['deep'], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _ax(ax, 'Figure 3 -- Pearson correlations (SId vs corpus variables)',
        ylabel='Pearson r')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out)


def plot_scatter_citas(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 4 -- SId v2 vs Bibliometric Impact',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    x = np.log1p(df['citations'].clip(0))
    y = df['score_v2']

    ax = axes[0]
    ax.hexbin(x, y, gridsize=35, cmap='YlOrRd', mincnt=1)
    r = float(y.corr(x))
    _ax(ax, f'Hexagonal density  [r = {r:.3f}]', 'log(1+citations)', 'SId v2')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    try:
        df2 = df.copy()
        df2['cq'] = pd.qcut(x, q=4,
                             labels=['Q1\n(least cited)','Q2','Q3','Q4\n(most cited)'],
                             duplicates='drop')
        groups = [df2.loc[df2['cq']==q,'score_v2'].dropna().values
                  for q in df2['cq'].cat.categories]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', lw=2))
        pal = [COLORS['light'], COLORS['v1'], COLORS['v2'], COLORS['deep']]
        for patch, c in zip(bp['boxes'], pal):
            patch.set_facecolor(c); patch.set_alpha(0.85)
        ax.set_xticklabels(df2['cq'].cat.categories, fontsize=8)
        _ax(ax, 'SId v2 by citation quartile', 'Citation quartile', 'SId v2')
    except Exception as e:
        ax.set_title(f'(error: {e})')
    plt.tight_layout()
    _save(fig, out)


def plot_boxplot_cuartiles(df, out):
    try:
        df2 = df.copy()
        df2['log_cit'] = np.log1p(df2['citations'].clip(0))
        df2['cq'] = pd.qcut(df2['log_cit'], q=4,
                             labels=['Q1','Q2','Q3','Q4'],
                             duplicates='drop')
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 5 -- Score distribution by citation quartile',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    for ax, col, color, label in [
        (axes[0], 'score_v1', COLORS['v1'], 'SId v1'),
        (axes[1], 'score_v2', COLORS['v2'], 'SId v2'),
    ]:
        groups = [df2.loc[df2['cq']==q, col].dropna().values
                  for q in df2['cq'].cat.categories]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', lw=2))
        for patch in bp['boxes']:
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticklabels(df2['cq'].cat.categories)
        _ax(ax, label, 'Citation quartile', 'Score')
    plt.tight_layout()
    _save(fig, out)


def plot_componentes(df, out):
    if 'Od' not in df.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 6 -- SId v2 decomposition: O_d (Originality) vs I_d (Impact)',
                 fontsize=10, fontweight='bold', color=COLORS['deep'])
    ax = axes[0]
    sc = ax.scatter(df['Id'], df['Od'], c=df['score_v2'],
                    cmap='RdYlGn', s=5, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label='SId v2')
    _ax(ax, 'O_d vs I_d', 'I_d (Impact)', 'O_d (Originality)')
    for ax2, col, color, tit in [
        (axes[1], 'Od', COLORS['v2'], 'O_d distribution'),
        (axes[2], 'Id', COLORS['v1'], 'I_d distribution'),
    ]:
        ax2.hist(df[col], bins=40, color=color, alpha=0.85,
                 edgecolor='white', lw=0.3)
        ax2.axvline(df[col].median(), color=COLORS['deep'], lw=1.5, ls='--')
        _ax(ax2, tit, col, 'Frequency')
    plt.tight_layout()
    _save(fig, out)


def plot_alpha_sensitivity(df, out):
    if 'Od' not in df.columns or 'Id' not in df.columns:
        return
    alphas  = np.linspace(0, 1, 21)
    records = []
    log_cit = np.log1p(df['citations'].clip(0))
    for a in alphas:
        sc = a * df['Od'] + (1-a) * df['Id']
        records.append({
            'alpha'     : a,
            'corr_cit'  : float(sc.corr(log_cit)),
            'corr_text' : float(sc.corr(df['text_length'])) if 'text_length' in df.columns else 0.0,
            'iqr'       : float(sc.quantile(.75) - sc.quantile(.25)),
        })
    da = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 7 -- Sensitivity to parameter alpha (originality weight)',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    for ax, col, ylabel, color in [
        (axes[0], 'corr_cit',  'r with log(citations)',   COLORS['v2']),
        (axes[1], 'iqr',       'Score IQR',      COLORS['accent']),
        (axes[2], 'corr_text', 'r with text length',  COLORS['v1']),
    ]:
        ax.plot(da['alpha'], da[col], color=color, lw=2, marker='o', ms=4)
        ax.axvline(DEFAULT_ALPHA, color=COLORS['deep'], ls='--', lw=1.5,
                   label=f'alpha={DEFAULT_ALPHA}')
        _ax(ax, ylabel, 'alpha', ylabel)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, out)


def plot_top_bottom(df, score_col, n, out):
    n   = int(n)
    top = df.nlargest(n, score_col).reset_index(drop=True)
    bot = df[df[score_col] > 0].nsmallest(n, score_col).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n*0.45)))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Figure 8 -- Top {n} and Bottom {n} documents by {score_col}',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    for ax, sub, tit, color in [
        (axes[0], top, f'TOP {n}',    COLORS['v2']),
        (axes[1], bot, f'BOTTOM {n}', COLORS['v1']),
    ]:
        lbls = [t[:50]+'...' if len(t)>50 else t for t in sub['title']]
        yp   = range(len(lbls)-1, -1, -1)
        bars = ax.barh(list(yp), sub[score_col].values,
                       color=color, alpha=0.82, edgecolor='white')
        ax.set_yticks(list(yp))
        ax.set_yticklabels(lbls, fontsize=7)
        ax.set_xlabel(score_col, fontsize=9)
        ax.set_title(tit, fontsize=9.5, fontweight='bold', color=COLORS['deep'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f8f9fb')
        ax.set_xlim(0, 1.1)
        for bar, cit in zip(bars, sub['citations'].values):
            ax.text(bar.get_width()+0.01,
                    bar.get_y()+bar.get_height()/2,
                    f'{int(cit):,}', va='center', fontsize=7,
                    color=COLORS['deep'])
    plt.tight_layout()
    _save(fig, out)


def plot_deciles(dec_df, out):
    if dec_df is None or dec_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 9 -- Feature profile by SId v2 decile',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    pal = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(dec_df)))
    decile_labels = dec_df['decile'].astype(str).tolist()
    for ax, col, label in [
        (axes[0], 'citas_median',  'Median citations'),
        (axes[1], 'text_len_mean', 'Mean text length (chars)'),
        (axes[2], 'ngrams_uniq',   'Mean unique n-grams'),
    ]:
        if col not in dec_df.columns:
            continue
        bars = ax.bar(decile_labels, dec_df[col].values,
                      color=pal, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()*1.01,
                    f'{bar.get_height():,.0f}',
                    ha='center', va='bottom', fontsize=7.5, color=COLORS['deep'])
        _ax(ax, label, 'SId v2 decile', label)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    _save(fig, out)


def _plot_synthetic(df, out):
    cats      = ['A','B','C','D']
    cat_lbls  = {'A':'A: Foundational','B':'B: Medium',
                 'C':'C: Low impact','D':'D: Short abstracts'}
    pal       = {'A':COLORS['green'],'B':COLORS['mid'],
                 'C':COLORS['v2'],   'D':COLORS['accent']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 10 -- Synthetic corpus validation (known ground truth)',
                 fontsize=11, fontweight='bold', color=COLORS['deep'])
    ax  = axes[0]
    x   = np.arange(len(cats))
    w   = 0.35
    v1m = [df[df['category']==c]['score_v1'].mean() for c in cats]
    v2m = [df[df['category']==c]['score_v2'].mean() for c in cats]
    ax.bar(x-w/2, v1m, w, label='SId v1', color=COLORS['v1'],
           alpha=0.85, edgecolor='white')
    ax.bar(x+w/2, v2m, w, label='SId v2', color=COLORS['v2'],
           alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.15)
    _ax(ax, 'Mean score by category', 'Category', 'SId Score')
    ax.legend(fontsize=9)
    ax = axes[1]
    for cat in cats:
        sub = df[df['category']==cat]
        ax.scatter(np.log1p(sub['citations']), sub['score_v2'],
                   c=pal[cat], s=130, label=cat_lbls[cat],
                   zorder=3, edgecolors='white', lw=1.5)
    _ax(ax, 'SId v2 vs log(1+citations)', 'log(1+citations)', 'SId v2')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# 10. REPORTE DE TEXTO
# ---------------------------------------------------------------------------

def write_report(df, stats_v1, stats_v2, corr_v1, corr_v2,
                 dec_df, alpha, n, path):
    lines = []
    SEP   = '=' * 70

    def sec(title):
        lines.extend(['', SEP, f'  {title}', SEP])

    lines += [SEP,
              '  SId v2 -- COMPLETE ANALYSIS REPORT',
              f'  Parameters: n-gram={n}  alpha={alpha}  beta={DEFAULT_BETA}',
              f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
              SEP]

    # Estadisticas comparativas
    sec('DESCRIPTIVE STATISTICS -- SId v1 vs SId v2')
    lines.append(f"  {'Metric':<28} {'SId v1':>12} {'SId v2':>12}  {'Change':>10}")
    lines.append('  ' + '-'*65)
    fields = [
        ('n_total',   'N documents',      False),
        ('n_zero',    'Docs score = 0',   True),
        ('pct_zero',  '% score = 0',      True),
        ('mean',      'Mean',             False),
        ('std',       'Std. deviation',   False),
        ('min',       'Minimum',          False),
        ('q1',        'Q1 (P25)',         False),
        ('median',    'Median (P50)',      False),
        ('q3',        'Q3 (P75)',         False),
        ('p90',       'P90',              False),
        ('max',       'Maximum',          False),
        ('iqr',       'IQR',              True),
        ('skewness',  'Skewness',         False),
        ('kurtosis',  'Kurtosis',         False),
    ]
    for key, label, lower_better in fields:
        v1 = stats_v1.get(key, float('nan'))
        v2 = stats_v2.get(key, float('nan'))
        if isinstance(v1, float):
            diff   = v2 - v1
            sign   = '+' if diff >= 0 else ''
            v1_s   = f'{v1:>12.4f}'
            v2_s   = f'{v2:>12.4f}'
            diff_s = f'{sign}{diff:.4f}'
        else:
            v1_s = v2_s = diff_s = ''
            v1_s   = f'{v1:>12}'
            v2_s   = f'{v2:>12}'
            diff_s = ''
        lines.append(f'  {label:<28}{v1_s}{v2_s}  {diff_s:>10}')

    # Correlaciones
    sec('PEARSON CORRELATIONS')
    lines.append(f"  {'Variable':<30} {'SId v1':>10} {'SId v2':>10}  {'Improvement':>12}")
    lines.append('  ' + '-'*62)
    all_keys = sorted(set(corr_v1) | set(corr_v2))
    for k in all_keys:
        r1 = corr_v1.get(k, float('nan'))
        r2 = corr_v2.get(k, float('nan'))
        m  = r2 - r1 if not (math.isnan(r1) or math.isnan(r2)) else float('nan')
        ms = f'{m:+.4f}' if not math.isnan(m) else ''
        lines.append(f'  {k:<30} {r1:>10.4f} {r2:>10.4f}  {ms:>8}')

    # Tabla por deciles
    if dec_df is not None and not dec_df.empty:
        sec('DECILE TABLE FOR SId v2 (documents with score > 0)')
        hdr = ['Decile','N','Mean score','Median cit.','Text length','Unique n-grams']
        lines.append('  ' + ' '.join(f'{h:>15}' for h in hdr))
        lines.append('  ' + '-'*95)
        for _, row in dec_df.iterrows():
            vals = [
                f'{str(row["decile"]):>15}',
                f'{int(row["n"]):>15,}',
                f'{row["score_mean"]:>15.4f}',
                f'{row["citas_median"]:>15.0f}',
                f'{row["text_len_mean"]:>15.0f}',
                f'{row["ngrams_uniq"]:>15.0f}',
            ]
            lines.append('  ' + ' '.join(vals))

    # Top 25
    sec('TOP 25 DOCUMENTS BY SId v2')
    lines.append(f"  {'#':<4} {'SId v2':>8} {'SId v1':>8} "
                 f"{'O_d':>7} {'I_d':>7} {'Citations':>10}  Title")
    lines.append('  ' + '-'*100)
    top25 = df.nlargest(25, 'score_v2')
    for rank, (_, row) in enumerate(top25.iterrows(), 1):
        t   = row['title'][:56] + '...' if len(row['title'])>56 else row['title']
        v1  = row.get('score_v1', float('nan'))
        od  = row.get('Od', float('nan'))
        iid = row.get('Id', float('nan'))
        lines.append(f"  {rank:<4} {row['score_v2']:>8.4f} {v1:>8.4f} "
                     f"{od:>7.4f} {iid:>7.4f} {int(row['citations']):>8,}  {t}")

    # Bottom 25
    sec('BOTTOM 25 DOCUMENTS BY SId v2 (excl. score=0)')
    lines.append(f"  {'#':<4} {'SId v2':>8} {'SId v1':>8} "
                 f"{'O_d':>7} {'I_d':>7} {'Citations':>10}  Title")
    lines.append('  ' + '-'*100)
    bot25 = df[df['score_v2'] > 0].nsmallest(25, 'score_v2')
    for rank, (_, row) in enumerate(bot25.iterrows(), 1):
        t   = row['title'][:56] + '...' if len(row['title'])>56 else row['title']
        v1  = row.get('score_v1', float('nan'))
        od  = row.get('Od', float('nan'))
        iid = row.get('Id', float('nan'))
        lines.append(f"  {rank:<4} {row['score_v2']:>8.4f} {v1:>8.4f} "
                     f"{od:>7.4f} {iid:>7.4f} {int(row['citations']):>8,}  {t}")

    lines += ['', SEP, '  END OF REPORT', SEP]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'    Reporte: {path}')


# ---------------------------------------------------------------------------
# 11. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def run_db_analysis(db_path, alpha, ngram, output_dir, limit=None):
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n{"="*60}')
    print(f'  SId v2  --  ANALYSIS ON calcia.db')
    print(f'  alpha={alpha}  n-gram={ngram}  output={output_dir}')
    print(f'{"="*60}')

    db_summary(db_path)
    documents = load_from_db(db_path, limit)
    if not documents:
        print('  ERROR: No documents found. Aborting.')
        return

    # IS_d v1
    print('\n  [1/2] SId v1...')
    t0  = datetime.now()
    cv1 = SingularityV1(n=ngram, beta=DEFAULT_BETA)
    df1 = cv1.analyze(documents)
    print(f'    Time: {(datetime.now()-t0).seconds}s')

    # IS_d v2
    print('\n  [2/2] SId v2...')
    t0  = datetime.now()
    cv2 = SingularityV2(n=ngram, alpha=alpha)
    df2 = cv2.analyze(documents)
    print(f'    Time: {(datetime.now()-t0).seconds}s')

    # Merge
    df = df1.merge(
        df2[['document_id','Od_raw','Od','Id','score_v2',
             'N_total_v2','T_total_v2']],
        on='document_id', how='left')
    df['log_cit'] = np.log1p(df['citations'].clip(0))

    # Estadisticas
    print('\n  Computing statistics...')
    stats_v1 = compute_stats(df['score_v1'], 'SId v1')
    stats_v2 = compute_stats(df['score_v2'], 'SId v2')
    corr_v1  = compute_correlations(df, 'score_v1')
    corr_v2  = compute_correlations(df, 'score_v2')
    dec_df   = decile_table(df, 'score_v2')

    # Resumen en consola
    print(f'\n  {"="*50}')
    print(f'  {"Metric":<28} {"SId v1":>10} {"SId v2":>10}')
    print(f'  {"-"*50}')
    for key, label in [
        ('n_total','N docs'), ('pct_zero','% score=0'),
        ('median','Median'), ('iqr','IQR'), ('skewness','Skewness')]:
        print(f'  {label:<28} {stats_v1[key]:>10.3f} {stats_v2[key]:>10.3f}')
    print(f'  {"-"*50}')
    print(f'  Correlation with log(citations):')
    print(f'    v1: {corr_v1.get("log(1+citations)",0):>7.4f}')
    print(f'    v2: {corr_v2.get("log(1+citations)",0):>7.4f}')
    print(f'  {"="*50}')

    # Exportar CSVs
    print('\n  Exporting CSVs...')
    p = lambda name: os.path.join(output_dir, name)

    df.to_csv(p('calcia_singularity_results.csv'), index=False)
    print(f'    calcia_singularity_results.csv  ({len(df):,} filas)')

    df.nlargest(REPORT_TOP_N, 'score_v2').to_csv(
        p('calcia_top50.csv'), index=False)
    df[df['score_v2']>0].nsmallest(REPORT_TOP_N,'score_v2').to_csv(
        p('calcia_bottom50.csv'), index=False)
    print(f'    calcia_top50.csv  /  calcia_bottom50.csv')

    if dec_df is not None and not dec_df.empty:
        dec_df.to_csv(p('calcia_deciles.csv'), index=False)
        print(f'    calcia_deciles.csv')

    # Tabla estadisticas
    stats_rows = []
    for key, label in [
        ('n_total','N documents'), ('n_zero','Docs score=0'),
        ('pct_zero','% score=0'), ('mean','Mean'), ('std','Std. deviation'),
        ('min','Minimum'), ('q1','Q1'), ('median','Median'),
        ('q3','Q3'), ('p90','P90'), ('max','Maximum'),
        ('iqr','IQR'), ('skewness','Skewness'), ('kurtosis','Kurtosis')]:
        stats_rows.append({'metrica': label,
                           'IS_d_v1': stats_v1.get(key,''),
                           'IS_d_v2': stats_v2.get(key,'')})
    pd.DataFrame(stats_rows).to_csv(p('calcia_stats_comparison.csv'), index=False)

    corr_rows = [{'variable': v,
                  'r_v1': corr_v1.get(v,''),
                  'r_v2': corr_v2.get(v,'')}
                 for v in sorted(set(corr_v1)|set(corr_v2))]
    pd.DataFrame(corr_rows).to_csv(p('calcia_correlaciones.csv'), index=False)
    print(f'    calcia_stats_comparison.csv  /  calcia_correlaciones.csv')

    # Figuras
    print('\n  Generating figures...')
    plot_histogramas      (df, p('fig1_distribucion.png'))
    plot_ceros_y_iqr      (df, p('fig2_ceros.png'))
    plot_correlaciones    (df, p('fig3_correlaciones.png'))
    plot_scatter_citas    (df, p('fig4_scatter_citas.png'))
    plot_boxplot_cuartiles(df, p('fig5_boxplot_cuartiles.png'))
    plot_componentes      (df, p('fig6_componentes.png'))
    plot_alpha_sensitivity(df, p('fig7_alpha_sensibilidad.png'))
    plot_top_bottom       (df, 'score_v2', 20, p('fig8_top_bottom.png'))
    plot_deciles          (dec_df, p('fig9_deciles_perfil.png'))

    # Validacion sintetica
    print('\n  Synthetic validation...')
    analyze_synthetic(n=ngram, alpha=alpha, output_dir=output_dir)

    # Reporte
    print('\n  Writing report...')
    write_report(df, stats_v1, stats_v2, corr_v1, corr_v2,
                 dec_df, alpha, ngram, p('singularity_report.txt'))

    print(f'\n{"="*60}')
    print(f'  ANALYSIS COMPLETE. Files in: {output_dir}/')
    print(f'{"="*60}\n')


def run_csv_analysis(csv_path, alpha, ngram, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n  Loading CSV: {csv_path}')
    df = pd.read_csv(csv_path)

    # Renombrar columna si viene de v1 antiguo
    for old, new in [('singularity_score','score_v1'),
                     ('N_qualifying_ngrams','N_qual_v1')]:
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if 'score_v1' not in df.columns:
        print('  ERROR: Column score_v1 not found.')
        return
    if 'text_length' not in df.columns:
        df['text_length'] = 1000
    if 'N_total' not in df.columns:
        df['N_total'] = df.get('unique_ngrams', pd.Series([100]*len(df)))

    # Aproximar O_d y I_d desde columnas existentes
    n_qual = df.get('N_qual_v1', df['N_total'])
    raw_O  = df['N_total'] * np.log1p(n_qual.clip(0))
    raw_O  = raw_O / df['text_length'].clip(lower=1)
    p99    = raw_O.quantile(0.99)
    df['Od']       = (raw_O / p99).clip(0, 1) if p99 > 0 else 0.0
    df['Id']       = np.log1p(df['citations'].clip(0)) / np.log1p(df['citations'].max())
    df['score_v2'] = alpha * df['Od'] + (1-alpha) * df['Id']
    df['N_total_v2'] = df['N_total']

    stats_v1 = compute_stats(df['score_v1'], 'SId v1')
    stats_v2 = compute_stats(df['score_v2'], 'SId v2')
    corr_v1  = compute_correlations(df, 'score_v1')
    corr_v2  = compute_correlations(df, 'score_v2')
    dec_df   = decile_table(df, 'score_v2')

    p = lambda name: os.path.join(output_dir, name)
    plot_histogramas      (df, p('fig1_distribucion.png'))
    plot_ceros_y_iqr      (df, p('fig2_ceros.png'))
    plot_correlaciones    (df, p('fig3_correlaciones.png'))
    plot_scatter_citas    (df, p('fig4_scatter_citas.png'))
    plot_componentes      (df, p('fig6_componentes.png'))
    plot_alpha_sensitivity(df, p('fig7_alpha_sensibilidad.png'))
    if 'title' in df.columns:
        plot_top_bottom(df, 'score_v2', 15, p('fig8_top_bottom.png'))
    plot_deciles(dec_df, p('fig9_deciles_perfil.png'))

    df.to_csv(p('calcia_singularity_results_v2.csv'), index=False)
    write_report(df, stats_v1, stats_v2, corr_v1, corr_v2,
                 dec_df, alpha, ngram, p('singularity_report.txt'))
    print(f'\n  CSV ANALYSIS complete. Files in: {output_dir}/')


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SId v2 -- Documentary Singularity Indicator for AI',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python singularitycalcia_v2.py --db calcia.db
  python singularitycalcia_v2.py --db calcia.db --alpha 0.7 --ngram 3 --output ./results
  python singularitycalcia_v2.py --csv results_v1.csv
  python singularitycalcia_v2.py --synthetic
""")
    parser.add_argument('--db',        type=str,
                        help='Path to calcia.db')
    parser.add_argument('--csv',       type=str,
                        help='Path to previous v1 results CSV')
    parser.add_argument('--synthetic', action='store_true',
                        help='Run only synthetic corpus validation')
    parser.add_argument('--alpha',     type=float, default=DEFAULT_ALPHA,
                        help=f'Originality weight alpha [0-1]  (default: {DEFAULT_ALPHA})')
    parser.add_argument('--ngram',     type=int,   default=DEFAULT_NGRAM,
                        help=f'N-gram size  (default: {DEFAULT_NGRAM})')
    parser.add_argument('--limit',     type=int,   default=None,
                        help='Limit number of documents (for quick tests)')
    parser.add_argument('--output',    type=str,   default='./resultados_SId_v2',
                        help='Output directory  (default: ./resultados_SId_v2)')

    args = parser.parse_args()

    if args.synthetic and not args.db and not args.csv:
        os.makedirs(args.output, exist_ok=True)
        analyze_synthetic(n=args.ngram, alpha=args.alpha, output_dir=args.output)
        return
    if args.csv:
        run_csv_analysis(args.csv, args.alpha, args.ngram, args.output)
        return
    if args.db:
        run_db_analysis(args.db, args.alpha, args.ngram, args.output, args.limit)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
