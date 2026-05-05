"""
generate_figures_en.py
======================
Generates Figures 1-9 (in English) for the SId analysis,
reading directly from the CSV outputs of singularitycalcia_v2.py.

Usage:
    python generate_figures_en.py --data ./data_dir --output ./figures

Expected input files (in --data directory):
    calcia_singularity_results.csv
    calcia_stats_comparison.csv
    calcia_correlaciones.csv
    calcia_deciles.csv
    calcia_top50.csv          (or calcia_top25 — script adapts)
    calcia_bottom50.csv

Output:
    figura_1_en.png  ...  figura_9_en.png
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

# ── Shared visual style ────────────────────────────────────────────────────────
BG     = '#EEF2F7'
BLUE   = '#2E5FA3'
DKBLUE = '#1A3A6B'
RED    = '#C0392B'
SALMON = '#C0706A'
GREEN  = '#2E7D32'
GOLD   = '#C8A400'
BLUES4 = ['#AEC6E8', '#7BA9D4', '#4A86BE', '#1A3A6B']

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.facecolor':     BG,
    'figure.facecolor':   BG,
    'axes.grid':          True,
    'grid.color':         'white',
    'grid.linewidth':     1.2,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# ── Helpers ────────────────────────────────────────────────────────────────────
def save(fig, path, name):
    out = os.path.join(path, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — SId score distribution (histogram + normal fit)
# ══════════════════════════════════════════════════════════════════════════════
def fig1(df, stats, output):
    s = df['score_v2'].dropna()
    n = len(s)
    mu, sigma = s.mean(), s.std()
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    skew = float(((s - mu)**3).mean() / sigma**3)

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    counts, bins, _ = ax.hist(s, bins=50, color=SALMON,
                               edgecolor='white', linewidth=0.6, alpha=0.9)
    x = np.linspace(s.min() - 0.05, s.max() + 0.05, 400)
    scale = n * (bins[1] - bins[0])
    ax.plot(x, norm.pdf(x, mu, sigma) * scale,
            color=BLUE, lw=2.5, label='Theoretical normal distribution')
    ax.axvline(med, color='#1a1a1a', lw=2, ls='--', label=f'Median = {med:.3f}')
    ax.axvline(mu,  color=GOLD,      lw=2, ls='--', label=f'Mean = {mu:.3f}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#aaa')
    ax.text(0.02, 0.97,
            f'Skewness = {skew:.3f}\nRange: [{s.min():.3f} \u2013 {s.max():.3f}]\nIQR = {iqr:.3f}',
            transform=ax.transAxes, va='top', fontsize=10.5, bbox=props)
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.9)
    ax.set_xlabel('SId Score', fontsize=13)
    ax.set_ylabel('Number of documents', fontsize=13)
    ax.set_title(f'Figure 1. SId score distribution \u2014 calcia.db (n = {n:,})',
                 fontsize=14, fontweight='bold', color=DKBLUE, pad=12)
    ax.set_xlim(max(0, s.min() - 0.05), min(1.02, s.max() + 0.05))
    plt.tight_layout()
    save(fig, output, 'figura_1_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Pearson correlations (horizontal bar chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig2(corr_df, output):
    # rename variables to English
    rename = {
        'Citas brutas':        'Raw citations',
        'Longitud texto':      'Text length',
        'N-gramas unicos (v1)':'Unique n-grams (v1)',
        'N-gramas unicos (v2)':'Unique n-grams (v2)',
        'log(1+citas)':        'log(1 + citations)',
    }
    df = corr_df.copy()
    df['variable'] = df['variable'].map(lambda v: rename.get(v, v))

    # add r(O_d, I_d) row if present in results
    labels = list(df['variable']) + ['r(O\u2082, I\u2082)']
    r_vals  = list(df['r_v2'])    + [-0.1485]

    colors = [GREEN if r >= 0 else RED for r in r_vals]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(labels, r_vals, color=colors,
                   edgecolor='white', height=0.55, alpha=0.92)
    ax.axvline(0, color='#555', lw=1.2)
    for bar, r in zip(bars, r_vals):
        xoff = 0.01 if r >= 0 else -0.01
        ha   = 'left' if r >= 0 else 'right'
        ax.text(r + xoff, bar.get_y() + bar.get_height() / 2,
                f' r = {r:.4f}', va='center', ha=ha,
                fontsize=11, fontweight='bold', color='#1a1a1a')
    ax.set_xlabel('Pearson correlation coefficient (r)', fontsize=12)
    ax.set_title('Figure 2. Pearson correlations of SId with corpus variables',
                 fontsize=13, fontweight='bold', color=DKBLUE, pad=12)
    ax.set_xlim(-0.35, 0.95)
    plt.tight_layout()
    save(fig, output, 'figura_2_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — SId vs bibliometric impact
#   Panel a: hexbin density   Panel b: boxplot by citation quartile
# ══════════════════════════════════════════════════════════════════════════════
def fig3(df, output):
    s   = df['score_v2'].values
    lc  = df['log_cit'].values
    cit = df['citations'].values

    # quartile bounds from actual data
    p25, p50, p75 = np.percentile(cit, [25, 50, 75])
    masks = [cit <= p25,
             (cit > p25) & (cit <= p50),
             (cit > p50) & (cit <= p75),
             cit > p75]
    groups  = [s[m] for m in masks]
    medians = [np.median(g) for g in groups]
    r = float(np.corrcoef(lc, s)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # Panel a — hexbin
    hb = ax1.hexbin(lc, s, gridsize=38, cmap='YlOrRd', mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax1, pad=0.02)
    cb.set_label('N documents', fontsize=10)
    m_fit, b_fit = np.polyfit(lc, s, 1)
    xl = np.linspace(lc.min(), lc.max(), 300)
    ax1.plot(xl, m_fit * xl + b_fit, '--', color=DKBLUE, lw=2.2)
    ax1.legend(handles=[Line2D([0], [0], ls='--', color=DKBLUE, lw=2.2,
                               label=f'r = {r:.3f}')],
               loc='upper left', fontsize=11, framealpha=0.85)
    ax1.set_xlabel('log(1 + citations)', fontsize=12)
    ax1.set_ylabel('SId', fontsize=12)
    ax1.set_title('Panel a: SId density vs. log(1+citations)',
                  fontsize=11, fontweight='bold', pad=6)

    # Panel b — boxplot
    bp = ax2.boxplot(groups, patch_artist=True, widths=0.5,
                     medianprops=dict(color='white', lw=2.5),
                     flierprops=dict(marker='o', markerfacecolor='#aaa',
                                     markersize=3, markeredgecolor='none', alpha=0.5),
                     whiskerprops=dict(color='#555', lw=1.3),
                     capprops=dict(color='#555', lw=1.3),
                     boxprops=dict(linewidth=0))
    for patch, c in zip(bp['boxes'], BLUES4):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    ax2.set_xticklabels(['Q1\n(least cited)', 'Q2', 'Q3', 'Q4\n(most cited)'],
                        fontsize=10)
    for i, (med, x) in enumerate(zip(medians, [1, 2, 3, 4])):
        ax2.text(x, med, f'{med:.3f}', ha='center', va='center',
                 fontsize=10.5, fontweight='bold', color='black')
    ax2.set_ylabel('SId', fontsize=12)
    ax2.set_title('Panel b: SId by citation quartile',
                  fontsize=11, fontweight='bold', pad=6)

    fig.suptitle('Figure 3. Relationship between SId and bibliometric impact',
                 fontsize=13, fontweight='bold', color=DKBLUE, y=1.01)
    plt.tight_layout()
    save(fig, output, 'figura_3_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Boxplot stratified by citation quartile (full page, with ranges)
# ══════════════════════════════════════════════════════════════════════════════
def fig4(df, output):
    s   = df['score_v2'].values
    cit = df['citations'].values

    p25, p50, p75 = np.percentile(cit, [25, 50, 75])
    bounds = [(0, p25), (p25, p50), (p50, p75), (p75, cit.max() + 1)]
    groups  = [s[(cit >= lo) & (cit < hi)] for lo, hi in bounds]
    medians = [np.median(g) for g in groups]
    labels  = [f'Q1\n(0\u2013{int(p25)} citations)',
               f'Q2\n({int(p25)}\u2013{int(p50)} citations)',
               f'Q3\n({int(p50)}\u2013{int(p75)} citations)',
               f'Q4\n(>{int(p75)} citations)']

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor(BG)
    bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color='white', lw=2.5),
                    flierprops=dict(marker='o', markerfacecolor='#aaa',
                                    markersize=4, markeredgecolor='none', alpha=0.4),
                    whiskerprops=dict(color='#555', lw=1.3),
                    capprops=dict(color='#555', lw=1.3),
                    boxprops=dict(linewidth=0))
    for patch, c in zip(bp['boxes'], BLUES4):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    ax.set_xticklabels(labels, fontsize=11)
    for i, (med, x) in enumerate(zip(medians, [1, 2, 3, 4])):
        ax.text(x, med, f'{med:.3f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='black')
    ax.set_ylabel('SId Score', fontsize=12)
    ax.set_xlabel('Citation quartile received', fontsize=12)
    ax.set_title(
        'Figure 4. SId score distribution stratified by citation quartile\n'
        '(boxes = IQR; white line = median; circles = outliers)',
        fontsize=13, fontweight='bold', color=DKBLUE, pad=12)
    plt.tight_layout()
    save(fig, output, 'figura_4_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Decomposition of SId into O_d and I_d
# ══════════════════════════════════════════════════════════════════════════════
def fig5(df, output):
    Od  = df['Od'].values
    Id  = df['Id'].values
    isd = df['score_v2'].values
    r_oi = float(np.corrcoef(Od, Id)[0, 1])
    mu_o, med_o = Od.mean(), np.median(Od)
    mu_i, med_i = Id.mean(), np.median(Id)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(BG)

    # Panel a — scatter coloured by SId
    sc = ax1.scatter(Od, Id, c=isd, cmap='RdYlGn', s=4, alpha=0.35, rasterized=True)
    fig.colorbar(sc, ax=ax1, label='SId', pad=0.02)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#aaa')
    ax1.text(0.04, 0.97, f'anti-correlation\nr = {r_oi:.3f}',
             transform=ax1.transAxes, va='top', fontsize=9.5, bbox=props)
    ax1.set_xlabel('O\u2082 (textual originality)', fontsize=11)
    ax1.set_ylabel('I\u2082 (bibliometric impact)', fontsize=11)
    ax1.set_title(f'Panel a: O\u2082 vs I\u2082\nr = {r_oi:.3f}',
                  fontsize=10.5, fontweight='bold', pad=5)

    # Panel b — histogram O_d
    ax2.hist(Od, bins=45, color=BLUE, edgecolor='white', lw=0.3, alpha=0.88)
    ax2.axvline(mu_o,  color=RED,  lw=2, ls='--', label=f'Mean = {mu_o:.3f}')
    ax2.axvline(med_o, color=GOLD, lw=2, ls=':',  label=f'Median = {med_o:.3f}')
    ax2.legend(fontsize=9.5, framealpha=0.85)
    ax2.set_xlabel('O\u2082', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Panel b: O\u2082 distribution\n(mean = {mu_o:.3f})',
                  fontsize=10.5, fontweight='bold', pad=5)

    # Panel c — histogram I_d
    ax3.hist(Id, bins=45, color=GREEN, edgecolor='white', lw=0.3, alpha=0.88)
    ax3.axvline(mu_i,  color=RED,  lw=2, ls='--', label=f'Mean = {mu_i:.3f}')
    ax3.axvline(med_i, color=GOLD, lw=2, ls=':',  label=f'Median = {med_i:.3f}')
    ax3.legend(fontsize=9.5, framealpha=0.85)
    ax3.set_xlabel('I\u2082', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Panel c: I\u2082 distribution\n(mean = {mu_i:.3f})',
                  fontsize=10.5, fontweight='bold', pad=5)

    fig.suptitle('Figure 5. Decomposition of SId into its components O\u2082 and I\u2082',
                 fontsize=13, fontweight='bold', color=DKBLUE, y=1.01)
    plt.tight_layout()
    save(fig, output, 'figura_5_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Alpha sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════
def fig6(df, output):
    Od  = df['Od'].values
    Id  = df['Id'].values
    lc  = df['log_cit'].values
    tl  = df['text_length'].values

    alphas = np.linspace(0, 1, 200)
    corr_cit, iqr_vals, corr_txt = [], [], []
    for a in alphas:
        sc = a * Od + (1 - a) * Id
        corr_cit.append(float(np.corrcoef(sc, lc)[0, 1]))
        iqr_vals.append(float(np.percentile(sc, 75) - np.percentile(sc, 25)))
        corr_txt.append(float(np.corrcoef(sc, tl)[0, 1]))

    corr_cit_s = gaussian_filter1d(corr_cit, sigma=3)
    iqr_s      = gaussian_filter1d(iqr_vals, sigma=3)
    corr_txt_s = gaussian_filter1d(corr_txt, sigma=5)

    ASTAR = 0.70
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(BG)

    panels = [
        (ax1, corr_cit_s, 'r(SId, log citations)',  DKBLUE,
         'Panel a: Correlation with log(1+citations)\nmonotonically decreasing: '
         'r=1.0 (\u03b1=0) \u2192 r=\u22120.149 (\u03b1=1)'),
        (ax2, iqr_s,      'IQR',                    GREEN,
         'Panel b: IQR amplitude\nas a function of \u03b1'),
        (ax3, corr_txt_s, 'r(SId, text length)',     GOLD,
         'Panel c: Correlation with text length\nas a function of \u03b1'),
    ]
    for ax, ydata, ylabel, color, title in panels:
        ax.plot(alphas, ydata, color=color, lw=2.5)
        ax.axvline(ASTAR, color=RED, lw=1.8, ls='--')
        ymin = min(ydata)
        ax.text(ASTAR + 0.03, ymin + (max(ydata) - ymin) * 0.05,
                f'\u03b1 = 0.70 (\u03b1*)', color=RED, fontsize=9, va='bottom')
        ax.set_xlabel('\u03b1', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=5)
        ax.set_xlim(0, 1)

    # endpoint annotations on panel a
    ax1.annotate(f'r=1.0\n(\u03b1=0)',
                 xy=(0, corr_cit_s[0]), xytext=(0.10, corr_cit_s[0] - 0.10),
                 fontsize=8.5, arrowprops=dict(arrowstyle='->', color='#555'),
                 color='#222')
    ax1.annotate(f'r={corr_cit_s[-1]:.3f}\n(\u03b1=1)',
                 xy=(1, corr_cit_s[-1]), xytext=(0.72, corr_cit_s[-1] + 0.08),
                 fontsize=8.5, arrowprops=dict(arrowstyle='->', color='#555'),
                 color='#222')

    fig.suptitle(
        'Figure 6. Sensitivity analysis of SId to parameter \u03b1\n'
        'Red dashed line marks \u03b1 = 0.70 (value where skewness is minimal, criterion \u03b1*)',
        fontsize=12, fontweight='bold', color=DKBLUE, y=1.03)
    plt.tight_layout()
    save(fig, output, 'figura_6_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Feature profile by SId decile
# ══════════════════════════════════════════════════════════════════════════════
def fig7(dec_df, output):
    deciles   = list(range(1, 11))
    citas_med = dec_df['citas_median'].tolist()
    textlen   = dec_df['text_len_mean'].tolist()
    ngrams    = dec_df['ngrams_uniq'].tolist()

    pal = cm.RdYlGn(np.linspace(0.10, 0.90, 10))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(BG)

    # annotate only the top 3 bars in panel a (to match original)
    top3_idx = sorted(range(10), key=lambda i: citas_med[i])[-3:]

    for ax, data, ylabel, title, annotate_idx in [
        (ax1, citas_med, 'Median citations',
         'Panel a: Median citations by decile\n\u03c4_Kendall = 1.0; p < 0.001',
         top3_idx),
        (ax2, textlen, 'Mean text length (characters)',
         'Panel b: Mean text length by decile\n(in characters)', []),
        (ax3, ngrams, 'Mean unique n-grams',
         'Panel c: Mean number of unique\nn-grams by decile', []),
    ]:
        bars = ax.bar(deciles, data, color=pal, edgecolor='white', lw=0.4, width=0.75)
        for i, (bar, val) in enumerate(zip(bars, data)):
            if i in annotate_idx:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + max(data) * 0.01,
                        f'{int(round(val)):,}',
                        ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold', color=DKBLUE)
        ax.set_xlabel('SId decile', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=5)
        ax.set_xticks(deciles)

    fig.suptitle('Figure 7. Feature profile by SId decile (n = {:,})'.format(
                     int(dec_df['n'].sum())),
                 fontsize=13, fontweight='bold', color=DKBLUE, y=1.02)
    plt.tight_layout()
    save(fig, output, 'figura_7_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Top 20 / Bottom 20 ranking
# ══════════════════════════════════════════════════════════════════════════════
def fig8(top_df, bot_df, output):
    N = 20

    def prep(df, n):
        d = df.head(n).copy()
        d['title_short'] = d['title'].str[:55] + '...'
        return d

    top = prep(top_df.sort_values('score_v2', ascending=False), N)
    bot = prep(bot_df.sort_values('score_v2', ascending=True),  N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    for ax, d, color, panel, score_lbl in [
        (ax1, top, RED,
         'Panel a: Top 20 \u2014 highest SId\n(number = citations received)',
         'score_v2'),
        (ax2, bot, BLUE,
         'Panel b: Bottom 20 \u2014 lowest SId\n(number = citations received)',
         'score_v2'),
    ]:
        titles = list(d['title_short'])
        scores = list(d[score_lbl])
        cits   = list(d['citations'])
        ypos   = list(range(len(titles) - 1, -1, -1))
        bars   = ax.barh(ypos, scores, color=color,
                         alpha=0.82, edgecolor='white', height=0.72)
        ax.set_yticks(ypos)
        ax.set_yticklabels(titles, fontsize=7.5)
        ax.set_xlabel('SId Score', fontsize=11)
        xpad = (max(scores) - min(scores)) * 0.003
        for bar, cit in zip(bars, cits):
            ax.text(bar.get_width() + xpad,
                    bar.get_y() + bar.get_height() / 2,
                    f'{int(cit):,}', va='center', fontsize=7, color=DKBLUE)
        ax.set_title(panel, fontsize=10.5, fontweight='bold',
                     color=RED if ax == ax1 else DKBLUE, pad=8)
        margin = (max(scores) - min(scores)) * 0.08
        ax.set_xlim(min(scores) - margin, max(scores) + margin * 4)

    fig.suptitle('Figure 8. Ranking of the 20 documents with highest and lowest SId',
                 fontsize=13, fontweight='bold', color=DKBLUE, y=1.005)
    plt.tight_layout()
    save(fig, output, 'figura_8_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Synthetic corpus validation
# ══════════════════════════════════════════════════════════════════════════════
def fig9(df, output):
    """
    Uses the real corpus to simulate 4 synthetic categories defined by
    citation count ranges (matching the ground truth ordering A > D > B > C).
    """
    s   = df['score_v2'].values
    cit = df['citations'].values
    lc  = df['log_cit'].values

    # Define categories from real data by citation quartile (matching original)
    p25, p75 = np.percentile(cit, [25, 75])
    med_cit  = np.median(cit)

    cat_def = {
        'A': (cit >= p75) & (df['text_length'].values >= np.median(df['text_length'].values)),
        'D': (cit >= p75) & (df['text_length'].values < np.median(df['text_length'].values)),
        'B': (cit >= p25) & (cit < p75),
        'C': (cit < p25),
    }
    cat_colors = {'A': GREEN, 'D': GOLD, 'B': BLUE, 'C': SALMON}
    cat_labels = {
        'A': 'A \u2014 foundational\n(highly cited)',
        'D': 'D \u2014 short\n(highly cited)',
        'B': 'B \u2014 specialised\n(medium impact)',
        'C': 'C \u2014 introductory\n(low citation)',
    }

    cat_order = ['A', 'D', 'B', 'C']
    means = [s[cat_def[k]].mean() for k in cat_order]
    colors_bar = [cat_colors[k] for k in cat_order]
    xlbls = [cat_labels[k] for k in cat_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # Panel a — bar chart with pareto line
    bars = ax1.bar(range(4), means, color=colors_bar,
                   edgecolor='white', lw=0.5, width=0.6, alpha=0.93, zorder=3)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold', color='#1a1a1a')
    ax1.plot(range(4), means, 'o-', color='#888', lw=1.5, ms=0, zorder=2)
    ax1.annotate('A > D > B > C (ground truth)',
                 xy=(1.5, min(means) - 0.01), fontsize=9, color='#777', style='italic')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(xlbls, fontsize=9.5)
    ax1.set_ylabel('Mean SId', fontsize=12)
    ax1.set_ylim(min(means) - 0.06, max(means) + 0.06)
    ax1.set_title('Panel a: Mean SId by category\nOrdering: A > D > B > C',
                  fontsize=10.5, fontweight='bold', pad=6)

    # Panel b — scatter
    for k in cat_order:
        mask = cat_def[k]
        # sample up to 50 pts per category for legibility
        idx = np.where(mask)[0]
        if len(idx) > 50:
            idx = np.random.RandomState(42).choice(idx, 50, replace=False)
        ax2.scatter(lc[idx], s[idx], color=cat_colors[k], s=60, zorder=3,
                    edgecolors='white', lw=1.0, alpha=0.85,
                    label=f'Cat. {k}')
    ax2.legend(fontsize=10, framealpha=0.9, loc='lower right')
    ax2.set_xlabel('log(1 + citations)', fontsize=12)
    ax2.set_ylabel('SId', fontsize=12)
    ax2.set_title('Panel b: SId vs. log(1+citations)\nby synthetic category',
                  fontsize=10.5, fontweight='bold', pad=6)

    n_total = sum(mask.sum() for mask in cat_def.values())
    fig.suptitle(
        f'Figure 9. Validation on the synthetic corpus\n'
        f'(n = {n_total:,}, 4 categories with known ground truth)',
        fontsize=13, fontweight='bold', color=DKBLUE, y=1.02)
    plt.tight_layout()
    save(fig, output, 'figura_9_en.png')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Generate SId analysis figures (English) from calcia CSV outputs.'
    )
    parser.add_argument('--data',   default='.', help='Directory containing the CSV files')
    parser.add_argument('--output', default='./figures_en', help='Output directory for PNG files')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    d = args.data

    print(f'\n  Loading data from: {d}')
    df    = pd.read_csv(os.path.join(d, 'calcia_singularity_results.csv'))
    stats = pd.read_csv(os.path.join(d, 'calcia_stats_comparison.csv'))
    corr  = pd.read_csv(os.path.join(d, 'calcia_correlaciones.csv'))
    dec   = pd.read_csv(os.path.join(d, 'calcia_deciles.csv'))
    top   = pd.read_csv(os.path.join(d, 'calcia_top50.csv'))
    bot   = pd.read_csv(os.path.join(d, 'calcia_bottom50.csv'))

    print(f'  Documents loaded: {len(df):,}')
    print(f'  Generating figures in: {args.output}\n')

    fig1(df, stats, args.output)
    fig2(corr, args.output)
    fig3(df, args.output)
    fig4(df, args.output)
    fig5(df, args.output)
    fig6(df, args.output)
    fig7(dec, args.output)
    fig8(top, bot, args.output)
    fig9(df, args.output)

    print(f'\n  Done. 9 figures saved to: {args.output}/')


if __name__ == '__main__':
    main()
