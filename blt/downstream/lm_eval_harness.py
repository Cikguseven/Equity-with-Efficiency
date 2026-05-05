#!/usr/bin/env python3
"""
lm-eval-harness evaluation pipeline for OLMo tokenizer comparison.
Benchmarks: XNLI, XCOPA, XStoryCloze,
            PIQA, HellaSwag, ARC-C

Usage:
    python pipeline_lm_eval.py \
        --model_path /data/models/olmo-local-myte \
        --tokenizer_name myte \
        --output_dir ./lmeval_results/myte
    done
"""

import argparse
import json
import csv
import logging
from pathlib import Path
from typing import Optional

import lm_eval

from lm_eval.models.huggingface import HFLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  TASK DEFINITIONS
# ---------------------------------------------------------------------------
# SEA-relevant language codes supported by each multilingual benchmark.
# Restrict to what lm-eval harness actually has task configs for.

# XNLI language codes
XNLI_LANGS = ["en", "th", "vi", "zh"]

# XCOPA SEA languages
XCOPA_LANGS = ["en", "id", "ta", "th", "vi", "zh"]

# XStoryCloze SEA languages
XSTORYCLOZE_LANGS = ["en", "id", "my", "zh"]

# English-only benchmarks
ENGLISH_TASKS = ["piqa", "hellaswag", "arc_challenge"]


def build_task_list(
    xnli_langs=XNLI_LANGS,
    xcopa_langs=XCOPA_LANGS,
    xstorycloze_langs=XSTORYCLOZE_LANGS,
    english_tasks=ENGLISH_TASKS,
) -> list[str]:
    tasks = []
    tasks += [f"xnli_{l}"           for l in xnli_langs]
    tasks += [f"xcopa_{l}"          for l in xcopa_langs]
    tasks += [f"xstorycloze_{l}"    for l in xstorycloze_langs]
    tasks += english_tasks
    return tasks


# ---------------------------------------------------------------------------
# 2.  FEW-SHOT CONFIG
# ---------------------------------------------------------------------------
# Standard few-shot settings matching published baselines.
FEWSHOT_CONFIG = {
    # --- Multilingual ---
    "xnli":           0,
    "xcopa":          0,
    "xstorycloze":    0,
    # --- English ---
    "piqa":           0,
    "hellaswag":      0,
    "arc_challenge":  0,
}


def fewshot_for(task_name: str) -> int:
    """Look up few-shot count by matching task prefix."""
    for prefix, n in FEWSHOT_CONFIG.items():
        if task_name.startswith(prefix):
            return n
    return 0   # safe default


# ---------------------------------------------------------------------------
# 3.  EVALUATION RUNNER
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str,
    output_dir: str,
    batch_size: int = 16,
    max_length: int = 2048,
    limit: Optional[float] = None,
    tasks: Optional[list[str]] = None,
    apply_chat_template: bool = False,
    trust_remote_code: bool = False,
) -> dict:
    """Run lm-eval harness for one model/tokenizer variant."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = build_task_list()

    from collections import defaultdict
    fewshot_groups: dict[int, list[str]] = defaultdict(list)
    for t in tasks:
        fewshot_groups[fewshot_for(t)].append(t)

    # Instantiate the model ONCE here to prevent reloading
    log.info(f"Loading model {model_path} into memory...")
    lm_obj = HFLM(
        pretrained=model_path,
        dtype="bfloat16",
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_length=max_length,
        tokenizer_kwargs={"trust_remote_code": trust_remote_code},
    )

    all_results: dict[str, dict] = {}

    for num_fewshot, task_group in sorted(fewshot_groups.items()):
        log.info(
            f"Evaluating {len(task_group)} tasks at {num_fewshot}-shot: "
            f"{task_group}"
        )
        try:
            # Pass the lm_obj directly instead of model="hf" and model_args
            result = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=task_group,
                num_fewshot=num_fewshot,
                limit=limit,
                apply_chat_template=apply_chat_template,
                log_samples=False,
            )
            all_results.update(result["results"])
        except Exception as exc:
            log.error(f"Error evaluating {task_group}: {exc}", exc_info=True)

    return all_results


# ---------------------------------------------------------------------------
# 4.  RESULT AGGREGATION & SAVING
# ---------------------------------------------------------------------------

def _primary_metric(task_name: str, result: dict) -> tuple[str, float]:
    """Return (metric_name, value) using priority order for each task family."""
    # Prefer acc_norm > acc > f1 > exact_match
    for base_key in ("acc_norm", "acc", "f1", "exact_match"):
        for key in (f"{base_key},none", base_key):
            if key in result:
                return base_key, round(result[key] * 100, 2)
    # Fallback: first numeric value
    for k, v in result.items():
        if isinstance(v, (int, float)) and not (k.endswith("_stderr") or "stderr" in k):
            return k, round(float(v) * 100, 2)
    return "unknown", 0.0


def save_results(
    all_results: dict,
    tokenizer_name: str,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-task CSV
    csv_rows = []
    for task, result in all_results.items():
        metric, value = _primary_metric(task, result)
        csv_rows.append({
            "tokenizer": tokenizer_name,
            "task": task,
            "metric": metric,
            "value": value,
        })

    csv_path = out / f"{tokenizer_name}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tokenizer", "task", "metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Aggregate by benchmark family
    family_map = {
        "xnli": [], "xcopa": [], "xstorycloze": [],
        "piqa": [], "hellaswag": [], "arc_challenge": [],
    }
    for row in csv_rows:
        for family in family_map:
            if row["task"].startswith(family):
                family_map[family].append(row["value"])

    averages = {
        fam: round(sum(vals) / len(vals), 2) if vals else None
        for fam, vals in family_map.items()
    }

    summary = {
        "tokenizer": tokenizer_name,
        "per_task": {r["task"]: r["value"] for r in csv_rows},
        "averages_by_benchmark": averages,
    }
    json_path = out / f"{tokenizer_name}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console print
    print(f"\n{'='*65}")
    print(f"RESULTS — {tokenizer_name}")
    print(f"{'Benchmark':<22} {'Avg (%)':>10}  Tasks")
    print("-" * 65)
    for fam, avg in averages.items():
        count = len(family_map[fam])
        avg_str = f"{avg:>10.2f}" if avg is not None else f"{'N/A':>10}"
        print(f"  {fam:<20} {avg_str}  ({count} tasks)")
    print(f"\nOutputs: {csv_path}\n         {json_path}")


# ---------------------------------------------------------------------------
# 5.  MULTI-TOKENIZER COMPARISON TABLE
# ---------------------------------------------------------------------------

def merge_results(result_dir: str, tokenizers: list[str]) -> None:
    """
    After running all three tokenizers, merge summaries into one CSV
    for easy comparison: rows = benchmarks, cols = tokenizer averages.
    """
    merged = {}
    for tok in tokenizers:
        p = Path(result_dir) / tok / f"{tok}_summary.json"
        if not p.exists():
            log.warning(f"Missing summary: {p}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for bench, avg in data["averages_by_benchmark"].items():
            merged.setdefault(bench, {})[tok] = avg

    if not merged:
        return

    csv_path = Path(result_dir) / "comparison_table.csv"
    fieldnames = ["benchmark"] + tokenizers
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bench, tok_avgs in merged.items():
            row = {"benchmark": bench, **{t: tok_avgs.get(t, "") for t in tokenizers}}
            writer.writerow(row)

    print(f"\nComparison table saved to: {csv_path}")
    print(f"\n{'Benchmark':<22}", end="")
    for tok in tokenizers:
        print(f"  {tok:>10}", end="")
    print()
    print("-" * (22 + 13 * len(tokenizers)))
    for bench, tok_avgs in merged.items():
        print(f"  {bench:<20}", end="")
        for tok in tokenizers:
            val = tok_avgs.get(tok, "")
            print(f"  {str(val):>10}", end="")
        print()


# ---------------------------------------------------------------------------
# 6.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="lm-eval harness pipeline for OLMo tokenizer comparison"
    )
    parser.add_argument("--model_path",      required=True, help="HF model dir or hub id")
    parser.add_argument("--tokenizer_name",  required=True,
                        choices=["parity-aware-bpe", "myte", "byte-level-bpe"])
    parser.add_argument("--output_dir",      default="./lmeval_results")
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--max_length",      type=int, default=2048)
    parser.add_argument("--limit",           type=float, default=None,
                        help="Fraction or N samples per task (for debugging)")
    parser.add_argument("--tasks",           nargs="+", default=None,
                        help="Override task list (default: all benchmarks)")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument(
        "--merge_only", nargs="+", default=None,
        metavar="TOK",
        help="Skip evaluation; just merge existing result JSONs for these tokenizers."
             " E.g.: --merge_only parity-aware-bpe myte byte-level-bpe"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.merge_only:
        merge_results(args.output_dir, args.merge_only)
    else:
        out = Path(args.output_dir) / args.tokenizer_name
        all_results = run_evaluation(
            model_path=args.model_path,
            output_dir=str(out),
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
            tasks=args.tasks,
            apply_chat_template=args.apply_chat_template,
            trust_remote_code=True,
        )
        save_results(all_results, args.tokenizer_name, str(out))
