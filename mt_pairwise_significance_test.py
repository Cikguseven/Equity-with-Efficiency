"""
Batch paired bootstrap significance testing for MT translation scores.

Files are expected under  {model_dir}/{xx}/*_{src}-{tgt}_scores.csv
where {xx} is the non-English ISO-639-1 code (the language subdirectory).
Each CSV must have  sentence_bleu  and  sentence_chrf  columns.

Significance tests are run per language-pair per metric.
Macro-averages are computed per direction (en->xx or xx->en) per metric.

Usage:
    python batch_significance_test.py <model1> <model2> <model1_dir> <model2_dir> \
        [--n_bootstrap N] [--seed S] [--output OUTPUT]             \
        [--languages id,th,vi,...] [--directions en->xx,xx->en]

Example:
    python batch_significance_test.py modelA modelB \
        /path/to/modelA/scores /path/to/modelB/scores \
        --output results.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

SEA_LANGUAGES = ['id', 'km', 'lo', 'ms', 'my', 'ta', 'th', 'vi', 'zh', 'tl']
METRICS = ['sentence_bleu', 'sentence_chrf']
DIRECTION_EN_XX = 'en->xx'
DIRECTION_XX_EN = 'xx->en'


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_csv_scores(filepath: Path) -> Dict[str, np.ndarray]:
    """
    Load sentence_bleu and sentence_chrf columns from a scores CSV.

    Returns a dict mapping metric name -> numpy array of floats.
    """
    columns: Dict[str, List[float]] = {m: [] for m in METRICS}

    with open(filepath, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {filepath}")
        missing = [m for m in METRICS if m not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV {filepath.name} is missing columns: {missing}. "
                f"Available: {reader.fieldnames}"
            )
        for row in reader:
            for metric in METRICS:
                columns[metric].append(float(row[metric]))

    return {m: np.array(v) for m, v in columns.items()}


# ── CHANGED: find_score_file replaced with find_translation_file ───────────────

def find_translation_file(base_dir: Path, src: str, tgt: str) -> Path:
    """
    Locate a translation score CSV file.

    Looks in {base_dir}/{lang}/ for files matching *_{src}-{tgt}_scores.csv,
    where lang is the non-English side of the pair (always the subfolder name).
    """
    lang = tgt if src == 'en' else src
    lang_dir = base_dir / lang

    if not lang_dir.exists():
        raise FileNotFoundError(f"Language directory not found: {lang_dir}")

    direction = f"{src}-{tgt}"
    pattern = f"*_{direction}_scores.csv"
    matches = list(lang_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {lang_dir}"
        )

    if len(matches) > 1:
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"  Warning: Multiple files for {direction}, "
              f"using most recent: {matches[0].name}")

    return matches[0]


# ── Statistical tests ─────────────────────────────────────────────────────────

def paired_bootstrap_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap test for H1: mean(scores1) > mean(scores2).

    Returns:
        p_value       – fraction of bootstrap samples where diff <= 0
        observed_diff – mean(scores1) - mean(scores2)
        std_err       – bootstrap standard deviation of the difference
    """
    if len(scores1) != len(scores2):
        raise ValueError(
            f"Score arrays must have equal length. Got {len(scores1)} vs {len(scores2)}."
        )
    n = len(scores1)
    observed_diff = np.mean(scores1) - np.mean(scores2)

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap)
    idx = rng.integers(0, n, size=(n_bootstrap, n))          # shape (B, n)
    boot_diffs = scores1[idx].mean(axis=1) - scores2[idx].mean(axis=1)

    return float(np.mean(boot_diffs <= 0)), observed_diff, float(np.std(boot_diffs))


def macro_average_bootstrap_test(
    scores1_list: List[np.ndarray],
    scores2_list: List[np.ndarray],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float, float, float]:
    """
    Macro-average paired bootstrap test across multiple language pairs.

    For each bootstrap replicate the macro-mean is computed as the arithmetic
    mean of per-language-pair bootstrap means, giving equal weight to every
    language regardless of corpus size.

    Returns:
        p_value, observed_diff, std_err, macro_mean1, macro_mean2
    """
    if len(scores1_list) != len(scores2_list):
        raise ValueError("scores1_list and scores2_list must have the same length.")

    lang_means1 = [float(np.mean(s)) for s in scores1_list]
    lang_means2 = [float(np.mean(s)) for s in scores2_list]
    macro_mean1 = float(np.mean(lang_means1))
    macro_mean2 = float(np.mean(lang_means2))
    observed_diff = macro_mean1 - macro_mean2

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        bm1, bm2 = [], []
        for s1, s2 in zip(scores1_list, scores2_list):
            n = len(s1)
            idx = rng.integers(0, n, size=n)
            bm1.append(np.mean(s1[idx]))
            bm2.append(np.mean(s2[idx]))
        boot_diffs[i] = np.mean(bm1) - np.mean(bm2)

    return (
        float(np.mean(boot_diffs <= 0)),
        observed_diff,
        float(np.std(boot_diffs)),
        macro_mean1,
        macro_mean2,
    )


def significance_level(p_value: float) -> str:
    """Star/caret notation for significance level."""
    if p_value < 0.001: return "***"
    if p_value < 0.01:  return "**"
    if p_value < 0.05:  return "*"
    if p_value > 0.999: return "^^^"
    if p_value > 0.99:  return "^^"
    if p_value > 0.95:  return "^"
    return "ns"


# ── Core per-direction logic ───────────────────────────────────────────────────

def run_direction(
    direction: str,
    languages: List[str],
    model1_dir: Path,
    model2_dir: Path,
    model1_name: str,
    model2_name: str,
    n_bootstrap: int,
    seed: int,
    results: list,
) -> None:
    """
    Run significance tests for every language pair in *direction*.

    direction is either DIRECTION_EN_XX ('en->xx') or DIRECTION_XX_EN ('xx->en').
    Results rows are appended to the *results* list in-place.
    Macro-averages per metric are appended at the end.
    """
    print(f"\n\n{'='*80}")
    print(f"DIRECTION: {direction}")
    print(f"{'='*80}")

    # Accumulators for macro-average (keyed by metric)
    ok_scores1: Dict[str, List[np.ndarray]] = {m: [] for m in METRICS}
    ok_scores2: Dict[str, List[np.ndarray]] = {m: [] for m in METRICS}

    for lang in languages:
        # ── CHANGED: derive src/tgt explicitly for find_translation_file ──────
        src, tgt = ('en', lang) if direction == DIRECTION_EN_XX else (lang, 'en')
        lang_pair = f"{src}-{tgt}"
        # ─────────────────────────────────────────────────────────────────────

        print(f"\n{'-'*60}")
        print(f"Language pair: {lang_pair}")
        print(f"{'-'*60}")

        try:
            # ── CHANGED: call find_translation_file instead of find_score_file
            file1 = find_translation_file(model1_dir, src, tgt)
            file2 = find_translation_file(model2_dir, src, tgt)
            # ─────────────────────────────────────────────────────────────────
            print(f"  Model 1: {file1.name}")
            print(f"  Model 2: {file2.name}")

            scores1_map = load_csv_scores(file1)
            scores2_map = load_csv_scores(file2)

        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            for metric in METRICS:
                results.append(_error_row(
                    direction, lang_pair, metric, model1_name, model2_name, 'NOT_FOUND'
                ))
            continue
        except Exception as exc:
            print(f"  ERROR: {exc}")
            for metric in METRICS:
                results.append(_error_row(
                    direction, lang_pair, metric, model1_name, model2_name, str(exc)[:60]
                ))
            continue

        for metric in METRICS:
            s1 = scores1_map[metric]
            s2 = scores2_map[metric]

            if len(s1) != len(s2):
                print(f"  [{metric}] ERROR: sample count mismatch ({len(s1)} vs {len(s2)})")
                results.append(_error_row(
                    direction, lang_pair, metric, model1_name, model2_name,
                    f"MISMATCH {len(s1)} vs {len(s2)}"
                ))
                continue

            p, diff, se = paired_bootstrap_test(s1, s2, n_bootstrap, seed)
            sig = significance_level(p)
            mean1, mean2 = float(np.mean(s1)), float(np.mean(s2))

            results.append({
                'direction':              direction,
                'lang_pair':              lang_pair,
                'metric':                 metric,
                f'{model1_name}_score':   mean1,
                f'{model2_name}_score':   mean2,
                'n_samples':              len(s1),
                'diff':                   diff,
                'std_err':                se,
                'p_value':                p,
                'significance':           sig,
                'status':                 'OK',
            })
            ok_scores1[metric].append(s1)
            ok_scores2[metric].append(s2)

            if p < 0.05:
                winner = f"Model 1 ({model1_name})"
            elif p > 0.95:
                winner = f"Model 2 ({model2_name})"
            else:
                winner = "No significant winner"

    # ── Macro-average per metric ───────────────────────────────────────────────
    for metric in METRICS:
        s1_list = ok_scores1[metric]
        s2_list = ok_scores2[metric]
        n_langs = len(s1_list)

        if n_langs < 2:
            print(f"\n  Skipping macro-average for [{metric}]: "
                  f"only {n_langs} valid language pair(s).")
            continue

        print(f"\n{'='*60}")
        print(f"MACRO-AVERAGE  direction={direction}  metric={metric}")
        print(f"{'='*60}")

        p, diff, se, m1, m2 = macro_average_bootstrap_test(
            s1_list, s2_list, n_bootstrap, seed
        )
        sig = significance_level(p)

        results.append({
            'direction':              direction,
            'lang_pair':              f'MACRO-{direction}',
            'metric':                 metric,
            f'{model1_name}_score':   m1,
            f'{model2_name}_score':   m2,
            'n_samples':              n_langs,
            'diff':                   diff,
            'std_err':                se,
            'p_value':                p,
            'significance':           sig,
            'status':                 'OK',
        })

        if p < 0.05:
            winner = f"Model 1 ({model1_name})"
        elif p > 0.95:
            winner = f"Model 2 ({model2_name})"
        else:
            winner = "No significant winner"

        print(
            f"  {model1_name}: {m1:.4f}  {model2_name}: {m2:.4f}  "
            f"diff: {diff:+.4f}  SE: {se:.4f}  p: {p:.4f} {sig}  → {winner}"
        )
        print(f"  ({n_langs} language pairs included)")


def _error_row(
    direction: str, lang_pair: str, metric: str,
    model1_name: str, model2_name: str, status: str,
) -> dict:
    return {
        'direction':              direction,
        'lang_pair':              lang_pair,
        'metric':                 metric,
        f'{model1_name}_score':   0.0,
        f'{model2_name}_score':   0.0,
        'n_samples':              0,
        'diff':                   0.0,
        'std_err':                0.0,
        'p_value':                1.0,
        'significance':           'ERROR',
        'status':                 status,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Batch paired bootstrap significance testing for MT translation scores'
    )
    parser.add_argument('model1',     type=str, help='Display name for model 1')
    parser.add_argument('model2',     type=str, help='Display name for model 2')
    parser.add_argument('model1_dir', type=str, help='Directory containing model 1 score CSVs')
    parser.add_argument('model2_dir', type=str, help='Directory containing model 2 score CSVs')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap replicates (default: 1000)')
    parser.add_argument('--seed',        type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output',      type=str, default=None,
                        help='Path to write results CSV')
    parser.add_argument(
        '--languages', type=str, default=None,
        help=(
            'Comma-separated ISO-639-1 language codes to test '
            f'(default: all {len(SEA_LANGUAGES)} SEA languages: {",".join(SEA_LANGUAGES)})'
        ),
    )
    parser.add_argument(
        '--directions', type=str, default=f'{DIRECTION_EN_XX},{DIRECTION_XX_EN}',
        help=(
            f'Comma-separated translation directions to test. '
            f'Choices: "{DIRECTION_EN_XX}", "{DIRECTION_XX_EN}" '
            f'(default: both)'
        ),
    )
    args = parser.parse_args()

    languages  = args.languages.split(',') if args.languages else SEA_LANGUAGES
    directions = [d.strip() for d in args.directions.split(',')]

    model1_dir = Path(args.model1_dir)
    model2_dir = Path(args.model2_dir)

    for label, d in [(args.model1, model1_dir), (args.model2, model2_dir)]:
        if not d.exists():
            print(f"Error: directory for {label} not found: {d}")
            return 1

    print(f"{'='*80}")
    print(f"BATCH MT TRANSLATION SIGNIFICANCE TESTING")
    print(f"{'='*80}")
    print(f"Model 1 ({args.model1}): {model1_dir}")
    print(f"Model 2 ({args.model2}): {model2_dir}")
    print(f"Languages:   {', '.join(languages)}")
    print(f"Directions:  {', '.join(directions)}")
    print(f"Metrics:     {', '.join(METRICS)}")
    print(f"Bootstrap:   {args.n_bootstrap} replicates  (seed={args.seed})")
    print(f"{'='*80}")

    results: list = []

    for direction in directions:
        if direction not in (DIRECTION_EN_XX, DIRECTION_XX_EN):
            print(f"Warning: unknown direction '{direction}', skipping.")
            continue
        run_direction(
            direction, languages,
            model1_dir, model2_dir,
            args.model1, args.model2,
            args.n_bootstrap, args.seed,
            results,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 100
    print(f"\n\n{'='*W}")
    print(f"SUMMARY")
    print(f"{'='*W}\n")
    col_m1 = f"{args.model1[:9]:<10}"
    col_m2 = f"{args.model2[:9]:<10}"
    print(
        f"{'Direction':<10} {'Lang pair':<20} {'Metric':<16} "
        f"{col_m1:>10} {col_m2:>10} {'Diff':>9} {'P-value':>9}  Sig"
    )
    print(f"{'-'*W}")

    prev_direction = None
    for r in results:
        if r['direction'] != prev_direction:
            if prev_direction is not None:
                print(f"{'-'*W}")
            prev_direction = r['direction']

        is_macro = r['lang_pair'].startswith('MACRO')
        if is_macro:
            print(f"{'-'*W}")

        if r['status'] == 'OK':
            print(
                f"{r['direction']:<10} {r['lang_pair']:<20} {r['metric']:<16} "
                f"{r[args.model1+'_score']:>10.4f} {r[args.model2+'_score']:>10.4f} "
                f"{r['diff']:>+9.4f} {r['p_value']:>9.4f}  {r['significance']}"
            )
        else:
            print(
                f"{r['direction']:<10} {r['lang_pair']:<20} {r['metric']:<16} "
                f"{'ERROR':>10} {'ERROR':>10} {'N/A':>9} {'N/A':>9}  {r['status']}"
            )

        if is_macro:
            print(f"{'-'*W}")

    print(f"\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
    print(f"^ codes indicate model 2 is significantly better (p>0.95/0.99/0.999)")

    # ── CSV output ────────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        fieldnames = [
            'direction', 'metric',
            f'{args.model1}_score', f'{args.model2}_score',  'significance',
            'diff', 'std_err', 'p_value',
            'n_samples', 'status', 'lang_pair',
        ]
        # ── CHANGED: only write macro-average rows ────────────────────────────
        macro_results = [r for r in results if r['lang_pair'].startswith('MACRO')]
        # ─────────────────────────────────────────────────────────────────────
        with open(out, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(macro_results)  # ← was: writer.writerows(results)
        print(f"\nResults saved to: {out}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())