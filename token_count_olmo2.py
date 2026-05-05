"""
Token count estimation for parallel MT data using the allenai/OLMo-2-0425-1B tokenizer.

Reads aligned en-xx sentence pairs, builds MT prompts, tokenizes in batches,
and reports per-language and total token counts. No files are written.

Speed-up: per-language tokenization is parallelised via a
reader-thread → N tokenizer-subprocesses → accumulator pipeline.
Since only a total count is needed, no ordering is required.
"""

import concurrent.futures
import json
import logging
import multiprocessing as mp
import threading
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler

# ── Rich ──────────────────────────────────────────────────────────────────────
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.theme import Theme

# ── HuggingFace Tokenizer ─────────────────────────────────────────────────────
from transformers import AutoTokenizer

# ── Console & Logging ─────────────────────────────────────────────────────────

console = Console(theme=Theme({"logging.level.info": "cyan"}))


def _setup_logging(use_rich: bool = True) -> None:
    handlers = (
        [RichHandler(console=console, rich_tracebacks=True, markup=True)]
        if use_rich
        else [logging.StreamHandler()]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )


_setup_logging()
log = logging.getLogger(__name__)


def get_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class Config:
    hf_model_id: str = "allenai/OLMo-2-0425-1B"

    data_dir: str = "/scratch/Projects/1/1/data/parallel"

    languages: Dict = field(
        default_factory=lambda: {
            "Indonesian": {"iso": "id"},
        }
    )

    min_chars: int = 10
    max_chars: int = 4000
    max_sentences_per_lang: int = 4_300_000
    tokenize_batch_size: int = 10_000

    # Language-level parallelism (one process per language)
    max_workers: int = 1
    # Tokenizer-level parallelism within each language
    num_tokenizer_workers: int = 16

    output_dir: str = "/scratch/Projects/1/1/user"


CFG = Config()


# ── Prompt Formatting ─────────────────────────────────────────────────────────


def build_mt_prompt(src: str, tgt: str, src_lang: str, tgt_lang: str) -> str:
    return f"{src_lang}: {src.strip()}\n{tgt_lang}: {tgt.strip()}"


# ── Tokenizer Worker (top-level for pickling) ─────────────────────────────────


def _tokenizer_worker_fn(
    in_q: "Queue[Optional[List[str]]]",
    out_q: "Queue[Optional[int]]",
    cfg: Config,
) -> None:
    """
    Subprocess: pulls a batch of prompts from in_q, tokenizes with the OLMo-2
    tokenizer, and pushes the integer token count (including +1 EOS per seq)
    to out_q. Terminates on None sentinel; emits its own None when done.
    """
    _setup_logging(use_rich=False)
    tok = AutoTokenizer.from_pretrained(cfg.hf_model_id)

    while True:
        batch = in_q.get()
        if batch is None:           # poison pill
            out_q.put(None)
            return

        encodings = tok(batch, add_special_tokens=False)
        count = sum(len(ids) + 1 for ids in encodings["input_ids"])  # +1 EOS
        out_q.put(count)


# ── Parallel Token Counter ────────────────────────────────────────────────────


def count_tokens_parallel(
    lang_name: str,
    lang_cfg: Dict,
    cfg: Config,
    num_workers: int,
) -> Dict:
    """
    Pipeline:
        [reader thread] → in_q → [N tokenizer procs] → out_q → [accumulator]

    - Reader: filters lines, caps at max_sentences_per_lang, pushes prompt
      batches (List[str]) to in_q.
    - Workers: tokenize each batch and push the integer count to out_q.
    - Accumulator: drains out_q and sums counts. No ordering needed.
    """
    folder   = Path(cfg.data_dir) / f"en-{lang_cfg['iso']}"
    src_path = folder / "train.en"
    tgt_path = folder / f"train.{lang_cfg['iso']}"

    if not src_path.exists():
        raise FileNotFoundError(f"[{lang_name}] Missing source file: {src_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"[{lang_name}] Missing target file: {tgt_path}")

    in_q:  "Queue[Optional[List[str]]]" = Queue(maxsize=num_workers * 2)
    out_q: "Queue[Optional[int]]"       = Queue(maxsize=num_workers * 4)

    procs = [
        Process(target=_tokenizer_worker_fn, args=(in_q, out_q, cfg), daemon=True)
        for _ in range(num_workers)
    ]
    for p in procs:
        p.start()

    log.info(f"[{lang_name}] Started {num_workers} tokenizer worker(s)")

    # Shared state written by reader thread, read after join
    reader_stats = {"n_seen": 0, "n_dropped": 0, "n_valid": 0}
    reader_exc: List[Optional[Exception]] = [None]

    # ── Reader thread ──────────────────────────────────────────────────────
    def _reader() -> None:
        n_seen = n_dropped = n_valid = 0
        batch: List[str] = []
        cap = cfg.max_sentences_per_lang

        try:
            with src_path.open("r", encoding="utf-8") as fs, \
                 tgt_path.open("r", encoding="utf-8") as ft:

                for src_line, tgt_line in zip(fs, ft):
                    n_seen += 1

                    if n_seen % 200_000 == 0:
                        log.info(
                            f"[{lang_name}] {n_seen:,} seen | {n_valid:,} valid | "
                            f"{n_dropped:,} dropped ({100 * n_dropped / n_seen:.1f}%)"
                        )

                    src = src_line.strip()
                    tgt = tgt_line.strip()

                    if not (src and tgt):
                        n_dropped += 1
                        continue
                    if not (cfg.min_chars <= len(src) <= cfg.max_chars):
                        n_dropped += 1
                        continue
                    if not (cfg.min_chars <= len(tgt) <= cfg.max_chars):
                        n_dropped += 1
                        continue

                    batch.append(build_mt_prompt(src, tgt, "English", lang_name))
                    n_valid += 1

                    if len(batch) >= cfg.tokenize_batch_size:
                        in_q.put(batch)
                        batch = []

                    if cap and n_valid >= cap:
                        log.info(
                            f"[{lang_name}] Sentence cap ({cap:,}) reached "
                            f"after {n_seen:,} seen."
                        )
                        break

            if batch:
                in_q.put(batch)

            if not (cap and n_valid >= cap):
                log.warning(
                    f"[{lang_name}] File exhausted: {n_valid:,} valid sentences "
                    f"(cap={cap:,}). Consider lowering min_chars/max_chars or "
                    "adding supplementary data."
                )

        except Exception as exc:
            reader_exc[0] = exc
        finally:
            reader_stats.update(n_seen=n_seen, n_dropped=n_dropped, n_valid=n_valid)
            for _ in range(num_workers):
                in_q.put(None)  # one sentinel per worker

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # ── Accumulator ────────────────────────────────────────────────────────
    # Order is irrelevant: just sum counts as they arrive.
    total_tokens = 0
    active_workers = num_workers

    while active_workers > 0:
        item = out_q.get()
        if item is None:
            active_workers -= 1
        else:
            total_tokens += item

    reader_thread.join()
    for p in procs:
        p.join()

    if reader_exc[0] is not None:
        raise reader_exc[0]

    n_valid   = reader_stats["n_valid"]
    n_dropped = reader_stats["n_dropped"]
    n_seen    = reader_stats["n_seen"]

    log.info(
        f"[{lang_name}] Done — {n_valid:,} sentences | "
        f"{total_tokens:,} tokens | "
        f"{n_dropped:,} dropped ({100 * n_dropped / n_seen:.1f}%)"
    )

    return {
        "sentences":               n_valid,
        "tokens":                  total_tokens,
        "dropped":                 n_dropped,
        "seen":                    n_seen,
        "avg_tokens_per_sentence": round(total_tokens / n_valid, 2) if n_valid else 0,
    }


# ── Per-language Entry Point (runs in ProcessPoolExecutor subprocess) ─────────


def count_tokens_for_language(
    lang_name: str,
    lang_cfg: Dict,
    cfg: Config,
) -> Dict:
    _setup_logging(use_rich=False)
    return count_tokens_parallel(
        lang_name   = lang_name,
        lang_cfg    = lang_cfg,
        cfg         = cfg,
        num_workers = cfg.num_tokenizer_workers,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    log.info(
        f"Token counting using tokenizer: [bold]{CFG.hf_model_id}[/bold]\n"
        f"Languages : {list(CFG.languages.keys())}\n"
        f"Workers   : {CFG.max_workers} language(s) × "
        f"{CFG.num_tokenizer_workers} tokenizer worker(s) each"
    )

    results: Dict[str, Dict] = {}

    with get_progress() as progress:
        task_id = progress.add_task(
            "Counting tokens...", total=len(CFG.languages)
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CFG.max_workers
        ) as executor:
            future_to_lang = {
                executor.submit(
                    count_tokens_for_language,
                    lang_name,
                    lang_cfg,
                    CFG,
                ): lang_name
                for lang_name, lang_cfg in CFG.languages.items()
            }

            for future in concurrent.futures.as_completed(future_to_lang):
                lang_name = future_to_lang[future]
                try:
                    results[lang_name] = future.result()
                    log.info(f"[{lang_name}] ✓ finished")
                except Exception as exc:
                    log.error(f"[{lang_name}] ✗ failed: {exc}", exc_info=True)
                    results[lang_name] = {
                        "sentences": 0, "tokens": 0,
                        "dropped": 0, "seen": 0,
                        "avg_tokens_per_sentence": 0,
                    }
                finally:
                    progress.advance(task_id)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'═' * 72}")
    log.info(f"Token count summary  |  tokenizer: {CFG.hf_model_id}")
    log.info(f"{'─' * 72}")
    log.info(
        f"  {'Language':12s} | {'Sentences':>12s} | {'Tokens':>15s} "
        f"| {'Avg tok/sent':>12s} | {'Dropped':>8s}"
    )
    log.info(f"  {'─' * 68}")

    total_sentences = total_tokens = 0
    for lang, info in sorted(results.items()):
        log.info(
            f"  {lang:12s} | {info['sentences']:>12,} | {info['tokens']:>15,} "
            f"| {info['avg_tokens_per_sentence']:>12.1f} | {info['dropped']:>8,}"
        )
        total_sentences += info["sentences"]
        total_tokens    += info["tokens"]

    log.info(f"  {'─' * 68}")
    log.info(
        f"  {'TOTAL':12s} | {total_sentences:>12,} | {total_tokens:>15,} "
        f"| {'':>12s} | "
    )
    log.info(f"  Grand total: {total_tokens / 1e9:.3f}B tokens")
    log.info(f"{'═' * 72}")

    # ── Optional JSON dump ────────────────────────────────────────────────────
    if CFG.output_dir:
        out_dir = Path(CFG.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "token_count_olmo2.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "tokenizer": CFG.hf_model_id,
                    "total_sentences": total_sentences,
                    "total_tokens": total_tokens,
                    "total_tokens_billions": round(total_tokens / 1e9, 4),
                    "per_language": results,
                },
                f,
                indent=2,
            )
        log.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()