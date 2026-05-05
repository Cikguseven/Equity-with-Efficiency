import numpy as np
import pandas as pd
from scipy import stats

# ── Data ──────────────────────────────────────────────────────────────────
data = {
    'fil_Latn':  {'script': 'Latin',   'MYTE': 1.28, 'PBPE': 1.26, 'BLBPE': 1.82, 'BLT': 3.22},
    'ind_Latn':  {'script': 'Latin',   'MYTE': 1.06, 'PBPE': 1.25, 'BLBPE': 1.03, 'BLT': 2.03},
    'vie_Latn':  {'script': 'Latin',   'MYTE': 1.34, 'PBPE': 1.26, 'BLBPE': 1.28, 'BLT': 2.12},
    'zsm_Latn':  {'script': 'Latin',   'MYTE': 1.07, 'PBPE': 1.25, 'BLBPE': 1.05, 'BLT': 2.06},
    'tha_Thai':  {'script': 'Abugida', 'MYTE': 1.49, 'PBPE': 1.26, 'BLBPE': 1.32, 'BLT': 2.13},
    'lao_Laoo':  {'script': 'Abugida', 'MYTE': 0.95, 'PBPE': 1.25, 'BLBPE': 3.05, 'BLT': 4.81},
    'khm_Khmr':  {'script': 'Abugida', 'MYTE': 1.36, 'PBPE': 1.10, 'BLBPE': 2.55, 'BLT': 3.86},
    'mya_Mymr':  {'script': 'Abugida', 'MYTE': 1.47, 'PBPE': 1.25, 'BLBPE': 2.53, 'BLT': 3.97},
    'tam_Taml':  {'script': 'Abugida', 'MYTE': 1.31, 'PBPE': 1.25, 'BLBPE': 1.46, 'BLT': 2.15},
}
# English excluded from parity analysis (parity = 1.0 by definition)

TOKENIZERS = ['MYTE', 'PBPE', 'BLBPE', 'BLT']

# ── Build subject matrix Y: shape (N, b) ──────────────────────────────────
langs   = list(data.keys())
scripts = [data[l]['script'] for l in langs]
Y       = np.array([[data[l][t] for t in TOKENIZERS] for l in langs])
# Y[i, j] = parity of language i under tokenizer j

groups    = sorted(set(scripts))          # ['Abugida', 'Latin']
group_idx = {g: [i for i, s in enumerate(scripts) if s == g] for g in groups}
N, b      = Y.shape     # N=9 languages, b=4 tokenizers
a         = len(groups)  # 2 script groups
n         = [len(group_idx[g]) for g in groups]  # subjects per group

# ── Compute means ─────────────────────────────────────────────────────────
grand_mean   = Y.mean()
subj_means   = Y.mean(axis=1)                                         # (N,)
within_means = Y.mean(axis=0)                                         # (b,)
group_means  = np.array([Y[group_idx[g]].mean() for g in groups])    # (a,)
cell_means   = np.array([Y[group_idx[g]].mean(axis=0) for g in groups])  # (a, b)

# ── Sum of Squares ────────────────────────────────────────────────────────
# Between-subjects: effect of Script
SS_B  = sum(b * ni * (gm - grand_mean)**2 for ni, gm in zip(n, group_means))
df_B  = a - 1

# Error (between): subject variability within groups
SS_Sw = sum(b * np.sum((subj_means[group_idx[g]] - group_means[i])**2)
            for i, g in enumerate(groups))
df_Sw = N - a

# Within-subjects: effect of Tokenizer
SS_W  = N * np.sum((within_means - grand_mean)**2)
df_W  = b - 1

# Interaction: Script × Tokenizer
SS_BW = sum(n[i] * np.sum((cell_means[i] - group_means[i] - within_means + grand_mean)**2)
            for i, g in enumerate(groups))
df_BW = (a - 1) * (b - 1)

# Error (within): residual after removing all effects
SS_err = sum(
    np.sum((Y[si] - cell_means[i] - subj_means[si] + group_means[i])**2)
    for i, g in enumerate(groups)
    for si in group_idx[g]
)
df_err = (N - a) * (b - 1)

# ── F-statistics and p-values ─────────────────────────────────────────────
def f_stat(SS1, df1, SS2, df2):
    MS1 = SS1 / df1
    MS2 = SS2 / df2
    F   = MS1 / MS2
    p   = 1 - stats.f.cdf(F, df1, df2)
    eta2 = SS1 / (SS1 + SS2)   # partial eta-squared
    return MS1, MS2, F, p, eta2

MS_B,  MS_Sw,  F_B,  p_B,  eta2_B  = f_stat(SS_B,  df_B,  SS_Sw,  df_Sw)
MS_W,  MS_err_w, F_W, p_W,  eta2_W  = f_stat(SS_W,  df_W,  SS_err, df_err)
MS_BW, MS_err_bw,F_BW,p_BW, eta2_BW = f_stat(SS_BW, df_BW, SS_err, df_err)

# ── Print ANOVA Table ─────────────────────────────────────────────────────
anova_table = pd.DataFrame([
    ('Script (between)',    SS_B,   df_B,  MS_B,    F_B,  p_B,  eta2_B),
    ('Tokenizer (within)', SS_W,   df_W,  MS_W,    F_W,  p_W,  eta2_W),
    ('Script × Tokenizer', SS_BW,  df_BW, MS_BW,   F_BW, p_BW, eta2_BW),
    ('Error (between)',    SS_Sw,  df_Sw, MS_Sw,   None, None, None),
    ('Error (within)',     SS_err, df_err,SS_err/df_err, None, None, None),
], columns=['Source', 'SS', 'df', 'MS', 'F', 'p-value', 'partial η²'])

print("=" * 72)
print("TWO-WAY MIXED ANOVA — TOKENIZER PARITY (English excluded, N=9)")
print("Between: Script (Latin vs Abugida) | Within: Tokenizer (4 levels)")
print("=" * 72)
print(anova_table.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x else ""))

