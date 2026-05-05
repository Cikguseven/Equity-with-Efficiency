"""
Batch paired bootstrap significance testing for multiple tasks and languages where
we want to check whether model1 is better than model2

Usage:
    python batch_significance_test.py <model1_dir> <model2_dir> [--n_bootstrap N] [--output OUTPUT]

Example:
    python batch_significance_test.py \
        /path/to/model1/checkpoint/samples \
        /path/to/model2/checkpoint/samples \
        --output results.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

NO_LANG = "__nolang__"

LANG_CANONICAL = {
    'eng_Latn': 'en',
    'ind_Latn': 'id',
    'khm_Khmr': 'km',
    'lao_Laoo': 'lo',
    'zsm_Latn': 'ms',
    'mya_Mymr': 'my',
    'tam_Taml': 'ta',
    'tha_Thai': 'th',
    'vie_Latn': 'vi',
    'zho_Hans': 'zh',
    'tgl_Latn': 'tl',
}


ENGLISH_ONLY_TASKS = {
    'piqa', 'hellaswag', 'arc_challenge'
}

DEFAULT_TASK_CONFIGS = {
    # Multilingual tasks
    'xnli': ['en', 'th', 'vi', 'zh'],
    'xcopa': ['en', 'id', 'ta', 'th', 'vi', 'zh'],
    'xstorycloze': ['en', 'id', 'my', 'zh'],
    # English-only tasks use NO_LANG sentinel
    'piqa': [NO_LANG],
    'hellaswag': [NO_LANG],
    'arc_challenge': [NO_LANG],
}


def display_lang(lang: str) -> str:
    """Convert internal language sentinel to display string."""
    return 'en' if lang == NO_LANG else lang


def canonical_lang(lang: str) -> str:
    """Normalise language code to a short canonical form for cross-task grouping."""
    if lang == NO_LANG:
        return 'en'
    return LANG_CANONICAL.get(lang, lang)


def parse_tasks_string(tasks_str: str) -> Dict[str, List[str]]:
    """
    Parse a comma-separated tasks string (lm_eval format) into task configs.

    Handles:
      - Multilingual:      xnli_en, xcopa_id
      - English-only:      piqa, hellaswag, arc_challenge

    Example input:  "xnli_en,xnli_th,xcopa_id,piqa,arc_challenge"
    Example output: {'xnli': ['en', 'th'], 'xcopa': ['id'],
                     'piqa': ['__nolang__'], 'arc_challenge': ['__nolang__']}
    """
    task_configs: Dict[str, List[str]] = {}

    for task_spec in tasks_str.split(','):
        task_spec = task_spec.strip()
        if not task_spec:
            continue

        if task_spec in ENGLISH_ONLY_TASKS:
            task = task_spec
            lang = NO_LANG

        else:
            parts = task_spec.split('_', 1)
            if len(parts) == 2:
                task, lang = parts
            else:
                print(f"  Warning: '{task_spec}' not in ENGLISH_ONLY_TASKS but has no language suffix. "
                      f"Treating as English-only. Add to ENGLISH_ONLY_TASKS if intentional.")
                task = task_spec
                lang = NO_LANG

        if task not in task_configs:
            task_configs[task] = []
        if lang not in task_configs[task]:
            task_configs[task].append(lang)

    return task_configs


def load_jsonl(filepath: str) -> List[dict]:
    """Load a JSONL file and return list of records."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_accuracies_keyed(data: List[dict]) -> Dict[int, float]:
    """Extract {doc_id: accuracy} mapping."""
    result = {}
    for record in data:
        doc_id = record.get('doc_id')
        if doc_id is None:
            raise ValueError(f"Record missing 'doc_id': {record}")
        if 'acc_norm' in record:
            result[doc_id] = float(record['acc_norm'])
        elif 'acc' in record:
            result[doc_id] = float(record['acc'])
        else:
            raise ValueError(f"Record missing 'acc' or 'acc_norm': {record}")
    return result


def align_accuracies(
    data1: List[dict], data2: List[dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two sets of records by doc_id.
    Only keeps doc_ids present in BOTH files (intersection).
    Raises a warning if the sets differ.
    """
    keyed1 = extract_accuracies_keyed(data1)
    keyed2 = extract_accuracies_keyed(data2)

    common_ids = sorted(set(keyed1.keys()) & set(keyed2.keys()))

    only_in_1 = set(keyed1) - set(keyed2)
    only_in_2 = set(keyed2) - set(keyed1)
    if only_in_1 or only_in_2:
        print(f"  Warning: doc_id mismatch — "
              f"{len(only_in_1)} only in model1, {len(only_in_2)} only in model2. "
              f"Using {len(common_ids)} common docs.")

    if not common_ids:
        raise ValueError("No common doc_ids between the two files!")

    acc1 = np.array([keyed1[i] for i in common_ids])
    acc2 = np.array([keyed2[i] for i in common_ids])
    return acc1, acc2


def paired_bootstrap_test(
    acc1: np.ndarray,
    acc2: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Perform paired bootstrap significance test.

    Returns:
        Tuple of (p_value, observed_diff, std_err)
    """
    if len(acc1) != len(acc2):
        raise ValueError(f"Arrays must have same length. Got {len(acc1)} and {len(acc2)}")

    n_samples = len(acc1)
    observed_diff = np.mean(acc1) - np.mean(acc2)

    np.random.seed(seed)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_diff = np.mean(acc1[indices]) - np.mean(acc2[indices])
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)
    std_err = np.std(bootstrap_diffs)
    p_value = np.mean(bootstrap_diffs <= 0)

    return p_value, observed_diff, std_err


def macro_average_bootstrap_test(
    acc1_arrays: List[np.ndarray],
    acc2_arrays: List[np.ndarray],
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float, float, float]:
    """
    Perform macro-average paired bootstrap significance test.

    Returns:
        Tuple of (p_value, observed_diff, std_err, macro_mean1, macro_mean2)
    """
    if len(acc1_arrays) != len(acc2_arrays):
        raise ValueError(f"Must have same number of languages. Got {len(acc1_arrays)} and {len(acc2_arrays)}")

    lang_means1 = [np.mean(acc) for acc in acc1_arrays]
    lang_means2 = [np.mean(acc) for acc in acc2_arrays]
    macro_mean1 = np.mean(lang_means1)
    macro_mean2 = np.mean(lang_means2)
    observed_diff = macro_mean1 - macro_mean2

    np.random.seed(seed)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        boot_lang_means1, boot_lang_means2 = [], []

        for acc1, acc2 in zip(acc1_arrays, acc2_arrays):
            n = len(acc1)
            indices = np.random.choice(n, size=n, replace=True)
            boot_lang_means1.append(np.mean(acc1[indices]))
            boot_lang_means2.append(np.mean(acc2[indices]))

        bootstrap_diffs.append(np.mean(boot_lang_means1) - np.mean(boot_lang_means2))

    bootstrap_diffs = np.array(bootstrap_diffs)
    std_err = np.std(bootstrap_diffs)
    p_value = np.mean(bootstrap_diffs <= 0)

    return p_value, observed_diff, std_err, macro_mean1, macro_mean2


def harmonic_mean(values: List[float]) -> float:
    """
    Compute harmonic mean of a list of values.
    Returns 0.0 if any value is zero to avoid division by zero.
    """
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def lang_harmonic_bootstrap_test(
    acc1_arrays: List[np.ndarray],
    acc2_arrays: List[np.ndarray],
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float, float, float]:
    """
    Perform per-language harmonic mean bootstrap significance test across tasks.

    For each bootstrap iteration:
      1. Resample each task's accuracy array independently (with replacement).
      2. Compute per-task mean for model1 and model2.
      3. Compute harmonic mean of those per-task means for each model.
      4. Record the difference in harmonic means.

    Using harmonic mean penalises weak performance on any single task more
    strongly than arithmetic mean, preventing a high-resource task from
    masking poor performance on a low-resource one.

    Returns:
        Tuple of (p_value, observed_diff, std_err, hmean1, hmean2)
    """
    if len(acc1_arrays) != len(acc2_arrays):
        raise ValueError(
            f"Must have same number of tasks. Got {len(acc1_arrays)} and {len(acc2_arrays)}"
        )

    task_means1 = [np.mean(acc) for acc in acc1_arrays]
    task_means2 = [np.mean(acc) for acc in acc2_arrays]
    hmean1 = harmonic_mean(task_means1)
    hmean2 = harmonic_mean(task_means2)
    observed_diff = hmean1 - hmean2

    np.random.seed(seed)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        boot_task_means1, boot_task_means2 = [], []

        for acc1, acc2 in zip(acc1_arrays, acc2_arrays):
            n = len(acc1)
            indices = np.random.choice(n, size=n, replace=True)
            boot_task_means1.append(np.mean(acc1[indices]))
            boot_task_means2.append(np.mean(acc2[indices]))

        bootstrap_diffs.append(harmonic_mean(boot_task_means1) - harmonic_mean(boot_task_means2))

    bootstrap_diffs = np.array(bootstrap_diffs)
    std_err = np.std(bootstrap_diffs)
    p_value = np.mean(bootstrap_diffs <= 0)

    return p_value, observed_diff, std_err, hmean1, hmean2


def find_sample_file(directory: Path, task: str, lang: str) -> Optional[Path]:
    """
    Find sample file matching task and language.

    Actual filename format: {task}_{lang}_samples.jsonl  (multilingual)
                            {task}_samples.jsonl          (English-only)
    """
    if lang == NO_LANG:
        pattern = f"{task}_samples.jsonl"
    else:
        pattern = f"{task}_{lang}_samples.jsonl"

    matches = list(directory.glob(pattern))

    if not matches:
        matches = list(directory.rglob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No file found matching pattern '{pattern}' in {directory}"
        )

    if len(matches) > 1:
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"  Warning: Multiple files found for {task}_{display_lang(lang)}, "
              f"using most recent: {matches[0].name}")

    return matches[0]


def significance_level(p_value: float) -> str:
    """Return significance level indicator."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value > 0.999:
        return "^^^"
    elif p_value > 0.99:
        return "^^"
    elif p_value > 0.95:
        return "^"
    else:
        return "ns"


def main():
    parser = argparse.ArgumentParser(
        description='Batch paired bootstrap significance testing'
    )
    parser.add_argument('model1', type=str)
    parser.add_argument('model2', type=str)
    parser.add_argument('model1_dir', type=str)
    parser.add_argument('model2_dir', type=str)
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument(
        '--tasks',
        type=str,
        default=None,
        help=(
            'Comma-separated task list in lm_eval format. '
            'Multilingual: xnli_en,xcopa_id. '
            'English-only: piqa,hellaswag,arc_challenge. '
            'If not provided, uses DEFAULT_TASK_CONFIGS.'
        )
    )

    args = parser.parse_args()

    task_configs = parse_tasks_string(args.tasks) if args.tasks else DEFAULT_TASK_CONFIGS
    if not task_configs:
        print(f"Error: Could not parse any tasks from: {args.tasks}")
        return 1

    model1_dir = Path(args.model1_dir)
    model2_dir = Path(args.model2_dir)

    if not model1_dir.exists():
        print(f"Error: Directory not found: {args.model1_dir}")
        return 1
    if not model2_dir.exists():
        print(f"Error: Directory not found: {args.model2_dir}")
        return 1

    print(f"{'='*80}")
    print(f"BATCH PAIRED BOOTSTRAP SIGNIFICANCE TESTING")
    print(f"{'='*80}")
    print(f"Model 1 directory: {model1_dir}")
    print(f"Model 2 directory: {model2_dir}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")

    results = []
    all_acc1_arrays, all_acc2_arrays = [], []
    task_acc1_arrays: Dict[str, List[np.ndarray]] = {}
    task_acc2_arrays: Dict[str, List[np.ndarray]] = {}

    lang_acc1_arrays: Dict[str, List[np.ndarray]] = {}
    lang_acc2_arrays: Dict[str, List[np.ndarray]] = {}
    lang_task_names:  Dict[str, List[str]]         = {}

    for task, languages in task_configs.items():
        print(f"\n{'='*80}")
        print(f"Task: {task.upper()}")
        print(f"{'='*80}")

        task_acc1_arrays[task] = []
        task_acc2_arrays[task] = []

        for lang in languages:
            lang_display = display_lang(lang)
            print(f"\n{'-'*80}")
            print(f"Language: {lang_display}")
            print(f"{'-'*80}")

            try:
                file1 = find_sample_file(model1_dir, task, lang)
                file2 = find_sample_file(model2_dir, task, lang)

                print(f"Model 1 file: {file1.name}")
                print(f"Model 2 file: {file2.name}")

                acc1, acc2 = align_accuracies(load_jsonl(file1), load_jsonl(file2))

                if len(acc1) != len(acc2):
                    print(f"ERROR: Sample count mismatch ({len(acc1)} vs {len(acc2)})")
                    results.append({
                        'task': task,
                        'language': lang_display,
                        args.model1 + '_acc': np.mean(acc1),
                        args.model2 + '_acc': np.mean(acc2),
                        'n_samples': f"{len(acc1)} vs {len(acc2)}",
                        'diff': 0.0,
                        'std_err': 0.0,
                        'p_value': 1.0,
                        'significance': 'ERROR',
                        'status': 'MISMATCH'
                    })
                    continue

                mean1, mean2 = np.mean(acc1), np.mean(acc2)
                p_value, observed_diff, std_err = paired_bootstrap_test(
                    acc1, acc2, n_bootstrap=args.n_bootstrap, seed=args.seed
                )
                sig_level = significance_level(p_value)

                results.append({
                    'task': task,
                    'language': lang_display,
                    args.model1 + '_acc': mean1,
                    args.model2 + '_acc': mean2,
                    'n_samples': len(acc1),
                    'diff': observed_diff,
                    'std_err': std_err,
                    'p_value': p_value,
                    'significance': sig_level,
                    'status': 'OK'
                })

                all_acc1_arrays.append(acc1)
                all_acc2_arrays.append(acc2)
                task_acc1_arrays[task].append(acc1)
                task_acc2_arrays[task].append(acc2)

                if lang != NO_LANG:
                    clang = canonical_lang(lang)
                    if clang not in lang_acc1_arrays:
                        lang_acc1_arrays[clang] = []
                        lang_acc2_arrays[clang] = []
                        lang_task_names[clang]  = []
                    lang_acc1_arrays[clang].append(acc1)
                    lang_acc2_arrays[clang].append(acc2)
                    lang_task_names[clang].append(task)

                print(f"\nModel 1 accuracy: {mean1:.4f} ({mean1*100:.2f}%)")
                print(f"Model 2 accuracy: {mean2:.4f} ({mean2*100:.2f}%)")
                print(f"Samples:          {len(acc1)}")
                print(f"Difference:       {observed_diff:+.4f} ({observed_diff*100:+.2f}%)")
                print(f"Std error:        {std_err:.4f}")
                print(f"P-value:          {p_value:.4f} {sig_level}")
                if p_value < 0.05:
                    print(f"Winner:           {'Model 1' if observed_diff > 0 else 'Model 2'}")
                else:
                    print(f"Winner:           No significant winner")

            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                results.append({
                    'task': task,
                    'language': lang_display,
                    args.model1 + '_acc': 0.0,
                    args.model2 + '_acc': 0.0,
                    'n_samples': 0,
                    'diff': 0.0,
                    'std_err': 0.0,
                    'p_value': 1.0,
                    'significance': 'ERROR',
                    'status': 'NOT_FOUND'
                })
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    'task': task,
                    'language': lang_display,
                    args.model1 + '_acc': 0.0,
                    args.model2 + '_acc': 0.0,
                    'n_samples': 0,
                    'diff': 0.0,
                    'std_err': 0.0,
                    'p_value': 1.0,
                    'significance': 'ERROR',
                    'status': str(e)[:50]
                })

    # -------------------------------------------------------------------------
    # Per-task macro-average (only meaningful for multilingual tasks with >= 2 languages)
    # -------------------------------------------------------------------------
    for task in task_configs:
        acc1_list = task_acc1_arrays.get(task, [])
        acc2_list = task_acc2_arrays.get(task, [])

        if len(acc1_list) >= 2:
            print(f"\n\n{'='*80}")
            print(f"MACRO-AVERAGE: {task.upper()}")
            print(f"{'='*80}")

            task_p, task_diff, task_se, task_m1, task_m2 = macro_average_bootstrap_test(
                acc1_list, acc2_list, n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            task_sig = significance_level(task_p)

            results.append({
                'task': f'MACRO-{task}',
                'language': 'ALL',
                args.model1 + '_acc': task_m1,
                args.model2 + '_acc': task_m2,
                'n_samples': f"{len(acc1_list)} langs",
                'diff': task_diff,
                'std_err': task_se,
                'p_value': task_p,
                'significance': task_sig,
                'status': 'OK'
            })

            print(f"\nModel 1 macro accuracy: {task_m1:.4f} ({task_m1*100:.2f}%)")
            print(f"Model 2 macro accuracy: {task_m2:.4f} ({task_m2*100:.2f}%)")
            print(f"Number of languages:    {len(acc1_list)}")
            print(f"Difference:             {task_diff:+.4f} ({task_diff*100:+.2f}%)")
            print(f"Std error:              {task_se:.4f}")
            print(f"P-value:                {task_p:.4f} {task_sig}")
            if task_p < 0.05:
                print(f"Winner:                 {'Model 1' if task_diff > 0 else 'Model 2'}")
            else:
                print(f"Winner:                 No significant winner")

    # -------------------------------------------------------------------------
    # Per-language harmonic mean across tasks
    # Only computed when a language appears in >= 2 tasks (otherwise meaningless).
    # English-only tasks (NO_LANG) are excluded from this grouping since they
    # don't carry a real language identity.
    # -------------------------------------------------------------------------
    lang_entries_for_summary = []

    langs_with_multi_task = {
        lang for lang, tasks in lang_task_names.items() if len(tasks) >= 2
    }

    if langs_with_multi_task:
        print(f"\n\n{'='*80}")
        print(f"PER-LANGUAGE HARMONIC MEAN ACROSS TASKS")
        print(f"{'='*80}")

        for lang in sorted(langs_with_multi_task):
            acc1_list = lang_acc1_arrays[lang]
            acc2_list = lang_acc2_arrays[lang]
            tasks_for_lang = lang_task_names[lang]

            print(f"\n{'-'*80}")
            print(f"Language: {lang}  |  Tasks: {', '.join(tasks_for_lang)}")
            print(f"{'-'*80}")

            lang_p, lang_diff, lang_se, lang_hm1, lang_hm2 = lang_harmonic_bootstrap_test(
                acc1_list, acc2_list, n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            lang_sig = significance_level(lang_p)

            row = {
                'task': f'LANG-HM-{lang}',
                'language': lang,
                args.model1 + '_acc': lang_hm1,
                args.model2 + '_acc': lang_hm2,
                'n_samples': f"{len(tasks_for_lang)} tasks",
                'diff': lang_diff,
                'std_err': lang_se,
                'p_value': lang_p,
                'significance': lang_sig,
                'status': 'OK'
            }
            results.append(row)
            lang_entries_for_summary.append(row)

            print(f"\nModel 1 harmonic mean: {lang_hm1:.4f} ({lang_hm1*100:.2f}%)")
            print(f"Model 2 harmonic mean: {lang_hm2:.4f} ({lang_hm2*100:.2f}%)")
            print(f"Number of tasks:       {len(tasks_for_lang)}")
            print(f"Difference:            {lang_diff:+.4f} ({lang_diff*100:+.2f}%)")
            print(f"Std error:             {lang_se:.4f}")
            print(f"P-value:               {lang_p:.4f} {lang_sig}")
            if lang_p < 0.05:
                print(f"Winner:                {'Model 1' if lang_diff > 0 else 'Model 2'}")
            else:
                print(f"Winner:                No significant winner")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Task':<22} {'Lang':<8} {'Model1':<10} {'Model2':<10} {'Diff':<10} {'P-value':<10} {'Sig':<5}")
    print(f"{'-'*80}")

    for r in results:
        if r['status'] == 'OK':
            tag = r['task']
            is_aggregate = tag.startswith('MACRO') or tag.startswith('LANG-HM')
            line = (
                f"{tag:<22} {r['language']:<8} "
                f"{r[args.model1 + '_acc']:>7.4f}    {r[args.model2 + '_acc']:>7.4f}    "
                f"{r['diff']:>+7.4f}    {r['p_value']:>7.4f}    {r['significance']:<5}"
            )
            if is_aggregate:
                print(f"{'-'*80}")
                print(line)
                print(f"{'-'*80}")
            else:
                print(line)
        else:
            print(f"{r['task']:<22} {r['language']:<8} {'ERROR':<10} {'ERROR':<10} "
                  f"{'N/A':<10} {'N/A':<10} {r['status']:<5}")

    print(f"\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10, ns not significant")
    print(f"^ codes indicate model2 is significantly better (p > 0.95/0.99/0.999)")
    print(f"LANG-HM-* rows use harmonic mean across tasks for that language\n")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['task', args.model1 + '_acc', args.model2 + '_acc',
                          'significance', 'p_value', 'diff', 'std_err',
                          'language', 'n_samples', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {output_path}\n")

    return 0


if __name__ == '__main__':
    exit(main())
