import time
from pathlib import Path
from typing import Dict, List, Any

import torch

# Set to True to enable testing for each tokenizer
TOKENIZERS_TO_TEST = {
    "myte": False,
    "parity_aware_bpe": False,
    "byte_level_bpe": False,
    "blt": True,
}

# Tokenizer file paths
MYTE_DECOMPOSE_MAP_PATH = "/localhome/user/8192_myte_SEA_1m/decompose.json"
MYTE_MERGE_MAP_PATH = "/localhome/user/8192_myte_SEA_1m/morf_map_mc4.json"
PARITY_AWARE_BPE_PATH = "/localhome/user/90k_parity-aware_SEA_1m/tokenizer.json"
BYTE_LEVEL_BPE_PATH = "/localhome/user/90k_byte-level_SEA_1m/tokenizer.json"
BLT_ENTROPY_MODEL_DIR = "/localhome/user/blt/entropy/"
BLT_CHECKPOINT_PATH = "/localhome/user/blt/model"

LINES = 1012
EVAL_DIR = Path("/localhome/user/flores-plus_dev_devtest")

SEA_11 = [
    "eng_Latn",
    "ind_Latn",
    "fil_Latn",
    "vie_Latn",
    "zsm_Latn",
    "khm_Khmr",
    "lao_Laoo",
    "mya_Mymr",
    "tha_Thai",
    "tam_Taml",
    "cmn_Hans",
]

VALID_LANGS = SEA_11


def read_lines(fp: Path, max_lines: int) -> List[str]:
    lines: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def token_counts_per_sentence(tokenizer: Any, lines: List[str]) -> List[int]:
    if not lines:
        return []
    out = tokenizer(lines, padding=False, add_special_tokens=False)
    return [len(mask) for mask in out["attention_mask"]]


def parity_token_counts_per_sentence(tokenizer: Any, lines: List[str]) -> List[int]:
    if not lines:
        return []
    encs = tokenizer.encode_batch(lines)
    return [len(enc.ids) for enc in encs]


def blt_patch_counts_per_sentence(tokenizer: Any, patcher: Any, lines: List[str]) -> List[int]:
    counts: List[int] = []
    for prompt in lines:
        token_ids = tokenizer.encode(prompt)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not token_ids:
            counts.append(0)
            continue

        tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
        patch_lengths, _ = patcher.patch(tokens_tensor, include_next_token=False)
        counts.append(len(patch_lengths.squeeze(0).tolist()))
    return counts


def gini(values: List[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values."""
    vals = [v for v in values]
    n = len(vals)
    if n < 2:
        return 0.0
    vals.sort()
    total = sum(vals)
    if total == 0:
        return 0.0
    weighted_sum = 0.0
    for i, v in enumerate(vals, start=1):
        weighted_sum += i * v
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def main() -> None:
    enabled = {k for k, v in TOKENIZERS_TO_TEST.items() if v}
    if not enabled:
        raise ValueError("No tokenizers enabled. Set at least one entry in TOKENIZERS_TO_TEST to True.")

    print(f"Enabled tokenizers: {', '.join(sorted(enabled))}")

    myte_tokenizer = None
    parity_aware_bpe_tokenizer = None
    byte_level_bpe_tokenizer = None
    blt_tokenizer = None
    blt_patcher = None

    if "myte" in enabled:
        import importlib.util
        spec = importlib.util.spec_from_file_location("myt5_tokenizer", "/localhome/user/8192_myte_SEA_1m/myt5_tokenizer.py")
        myt5_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(myt5_module)
        MyT5Tokenizer = myt5_module.MyT5Tokenizer
        myte_tokenizer = MyT5Tokenizer(
            decompose_map=MYTE_DECOMPOSE_MAP_PATH,
            merge_map=MYTE_MERGE_MAP_PATH
        )

    if "parity_aware_bpe" in enabled:
        from tokenizers import Tokenizer
        parity_aware_bpe_tokenizer = Tokenizer.from_file(PARITY_AWARE_BPE_PATH)

    if "byte_level_bpe" in enabled:
        from tokenizers import Tokenizer
        byte_level_bpe_tokenizer = Tokenizer.from_file(BYTE_LEVEL_BPE_PATH)

    if "blt" in enabled:
        from bytelatent.generate import load_consolidated_model_and_tokenizer
        from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum

        entropy_model_dir = Path(BLT_ENTROPY_MODEL_DIR)
        checkpoint_path = Path(BLT_CHECKPOINT_PATH)

        print("Loading BLT model...")
        _, blt_tokenizer, _ = load_consolidated_model_and_tokenizer(checkpoint_path)

        print("Initializing BLT Patcher...")
        patcher_args = PatcherArgs(
            patching_mode=PatchingModeEnum.entropy,
            realtime_patching=True,
            entropy_model_checkpoint_dir=str(entropy_model_dir),
            patching_device="cuda:0",
            device="cuda:0",
        )
        blt_patcher = patcher_args.build()

    # Metrics dicts
    tokens_per_lang_myte: Dict[str, int] = {}
    avg_tokens_myte: Dict[str, float] = {}
    avg_parity_myte: Dict[str, float] = {}
    cr_lang_myte: Dict[str, float] = {}

    tokens_per_lang_pa_bpe: Dict[str, int] = {}
    avg_tokens_pa_bpe: Dict[str, float] = {}
    avg_parity_pa_bpe: Dict[str, float] = {}
    cr_lang_pa_bpe: Dict[str, float] = {}

    tokens_per_lang_bl_bpe: Dict[str, int] = {}
    avg_tokens_bl_bpe: Dict[str, float] = {}
    avg_parity_bl_bpe: Dict[str, float] = {}
    cr_lang_bl_bpe: Dict[str, float] = {}

    tokens_per_lang_blt: Dict[str, int] = {}
    avg_tokens_blt: Dict[str, float] = {}
    avg_parity_blt: Dict[str, float] = {}
    cr_lang_blt: Dict[str, float] = {}

    if not EVAL_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {EVAL_DIR}")

    output_lines: List[str] = []
    output_lines.append(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    output_lines.append(f"Enabled tokenizers: {', '.join(sorted(enabled))}")

    files_to_process = [
        entry for entry in sorted(EVAL_DIR.iterdir())
        if entry.is_file() and entry.name.rsplit(".", 1)[0] in VALID_LANGS and entry.suffix == ".devtest"
    ]

    for idx, entry in enumerate(files_to_process, 1):
        code = entry.name.rsplit(".", 1)[0]
        print(f"[{idx}/{len(files_to_process)}] Processing language: {code}")

        lines = read_lines(entry, LINES)
        if not lines:
            continue

        output_lines.append(f"\n--- Language: {code} ---")
        per_lang_parts = []

        if "myte" in enabled:
            counts = token_counts_per_sentence(myte_tokenizer, lines)
            counts = [c for c in counts if c > 0]
            tokens_per_lang_myte[code] = sum(counts)
            avg_tokens_myte[code] = (sum(counts) / len(counts)) if counts else 0.0
            cr_lang_myte[code] = (sum(1.0 / c for c in counts) / len(counts)) if counts else 0.0
            per_lang_parts.append(f"MYTE tokens={tokens_per_lang_myte[code]}")

        if "parity_aware_bpe" in enabled:
            counts = parity_token_counts_per_sentence(parity_aware_bpe_tokenizer, lines)
            counts = [c for c in counts if c > 0]
            tokens_per_lang_pa_bpe[code] = sum(counts)
            avg_tokens_pa_bpe[code] = (sum(counts) / len(counts)) if counts else 0.0
            cr_lang_pa_bpe[code] = (sum(1.0 / c for c in counts) / len(counts)) if counts else 0.0
            per_lang_parts.append(f"Parity-Aware BPE tokens={tokens_per_lang_pa_bpe[code]}")

        if "byte_level_bpe" in enabled:
            counts = parity_token_counts_per_sentence(byte_level_bpe_tokenizer, lines)
            counts = [c for c in counts if c > 0]
            tokens_per_lang_bl_bpe[code] = sum(counts)
            avg_tokens_bl_bpe[code] = (sum(counts) / len(counts)) if counts else 0.0
            cr_lang_bl_bpe[code] = (sum(1.0 / c for c in counts) / len(counts)) if counts else 0.0
            per_lang_parts.append(f"Byte-Level BPE tokens={tokens_per_lang_bl_bpe[code]}")

        if "blt" in enabled:
            counts = blt_patch_counts_per_sentence(blt_tokenizer, blt_patcher, lines)
            counts = [c for c in counts if c > 0]
            tokens_per_lang_blt[code] = sum(counts)
            avg_tokens_blt[code] = (sum(counts) / len(counts)) if counts else 0.0
            cr_lang_blt[code] = (sum(1.0 / c for c in counts) / len(counts)) if counts else 0.0
            per_lang_parts.append(f"BLT patches={tokens_per_lang_blt[code]}")

        output_lines.append(", ".join(per_lang_parts))

    # English baselines (only for enabled tokenizers)
    eng_avg_myte = avg_tokens_myte.get("eng_Latn") if "myte" in enabled else None
    eng_avg_pa_bpe = avg_tokens_pa_bpe.get("eng_Latn") if "parity_aware_bpe" in enabled else None
    eng_avg_bl_bpe = avg_tokens_bl_bpe.get("eng_Latn") if "byte_level_bpe" in enabled else None
    eng_avg_blt = avg_tokens_blt.get("eng_Latn") if "blt" in enabled else None

    # Per-language parity + compression stats
    all_langs = sorted(set().union(
        avg_tokens_myte.keys(),
        avg_tokens_pa_bpe.keys(),
        avg_tokens_bl_bpe.keys(),
        avg_tokens_blt.keys(),
    ))

    for lang in all_langs:
        output_lines.append(f"\n--- Language: {lang} ---")

        if "myte" in enabled and lang in avg_tokens_myte and eng_avg_myte:
            parity = avg_tokens_myte[lang] / eng_avg_myte
            avg_parity_myte[lang] = parity
            lines_per_token = 1 / avg_tokens_myte[lang] if avg_tokens_myte[lang] > 0 else 0.0
            parts = []
            parts.append(f"MYTE parity: {parity:.2f}")
            parts.append(f"MYTE average tokens per sentence: {avg_tokens_myte[lang]:.2f}")
            parts.append(f"MYTE compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))

        if "parity_aware_bpe" in enabled and lang in avg_tokens_pa_bpe and eng_avg_pa_bpe:
            parity = avg_tokens_pa_bpe[lang] / eng_avg_pa_bpe
            avg_parity_pa_bpe[lang] = parity
            lines_per_token = 1 / avg_tokens_pa_bpe[lang] if avg_tokens_pa_bpe[lang] > 0 else 0.0
            parts = []
            parts.append(f"Parity-Aware BPE parity: {parity:.2f}")
            parts.append(f"Parity-Aware BPE average tokens per sentence: {avg_tokens_pa_bpe[lang]:.2f}")
            parts.append(f"Parity-Aware BPE compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))

        if "byte_level_bpe" in enabled and lang in avg_tokens_bl_bpe and eng_avg_bl_bpe:
            parity = avg_tokens_bl_bpe[lang] / eng_avg_bl_bpe
            avg_parity_bl_bpe[lang] = parity
            lines_per_token = 1 / avg_tokens_bl_bpe[lang] if avg_tokens_bl_bpe[lang] > 0 else 0.0
            parts = []
            parts.append(f"Byte-Level BPE parity: {parity:.2f}")
            parts.append(f"Byte-Level BPE average tokens per sentence: {avg_tokens_bl_bpe[lang]:.2f}")
            parts.append(f"Byte-Level BPE compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))

        if "blt" in enabled and lang in avg_tokens_blt and eng_avg_blt:
            parity = avg_tokens_blt[lang] / eng_avg_blt
            avg_parity_blt[lang] = parity
            lines_per_token = 1 / avg_tokens_blt[lang] if avg_tokens_blt[lang] > 0 else 0.0
            parts = []
            parts.append(f"BLT parity: {parity:.2f}")
            parts.append(f"BLT average patches per sentence: {avg_tokens_blt[lang]:.2f}")
            parts.append(f"BLT compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))

    # Gini coefficients
    output_lines.append("\n--- Gini Coefficient (lower is better) ---")
    gini_parts: List[str] = []
    if "myte" in enabled:
        gini_parts.append(f"MYTE: {gini(list(avg_tokens_myte.values())):.3f}")
    if "parity_aware_bpe" in enabled:
        gini_parts.append(f"Parity-Aware BPE: {gini(list(avg_tokens_pa_bpe.values())):.3f}")
    if "byte_level_bpe" in enabled:
        gini_parts.append(f"Byte-Level BPE: {gini(list(avg_tokens_bl_bpe.values())):.3f}")
    if "blt" in enabled:
        gini_parts.append(f"BLT: {gini(list(avg_tokens_blt.values())):.3f}")
    output_lines.append(", ".join(gini_parts) if gini_parts else "No enabled tokenizers produced data.")

    def _combined_avg(d: Dict[str, float], exclude_key: str = None) -> float:
        if not d:
            return 0.0
        if exclude_key:
            vals = [v for k, v in d.items() if k != exclude_key]
            return (sum(vals) / len(vals)) if vals else 0.0
        return (sum(d.values()) / len(d))

    output_lines.append("\n--- Average tokens per sentence (lower is better) ---")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {_combined_avg(avg_tokens_myte):.2f}")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {_combined_avg(avg_tokens_pa_bpe):.2f}")
    if "byte_level_bpe" in enabled:
        output_lines.append(f"Byte-Level BPE: {_combined_avg(avg_tokens_bl_bpe):.2f}")
    if "blt" in enabled:
        output_lines.append(f"BLT: {_combined_avg(avg_tokens_blt):.2f}")

    output_lines.append("\n--- Compression Rate (macro-averaged; higher is better) ---")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {_combined_avg(cr_lang_myte):.4f}")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {_combined_avg(cr_lang_pa_bpe):.4f}")
    if "byte_level_bpe" in enabled:
        output_lines.append(f"Byte-Level BPE: {_combined_avg(cr_lang_bl_bpe):.4f}")
    if "blt" in enabled:
        output_lines.append(f"BLT: {_combined_avg(cr_lang_blt):.4f}")

    output_lines.append("\n--- Average Tokenizer Parity vs English (macro-averaged; lower is better) ---")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {_combined_avg(avg_parity_myte, exclude_key='eng_Latn'):.2f}" if eng_avg_myte else "MYTE: English missing")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {_combined_avg(avg_parity_pa_bpe, exclude_key='eng_Latn'):.2f}" if eng_avg_pa_bpe else "Parity-Aware BPE: English missing")
    if "byte_level_bpe" in enabled:
        output_lines.append(f"Byte-Level BPE: {_combined_avg(avg_parity_bl_bpe, exclude_key='eng_Latn'):.2f}" if eng_avg_bl_bpe else "Byte-Level BPE: English missing")
    if "blt" in enabled:
        output_lines.append(f"BLT: {_combined_avg(avg_parity_blt, exclude_key='eng_Latn'):.2f}" if eng_avg_blt else "BLT: English missing")

    output_lines.append("\n--- Worst-case Tokenizer Parity vs English (lower is better) ---")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {max(avg_parity_myte.values()):.2f}" if eng_avg_myte and avg_parity_myte else "MYTE: English missing")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {max(avg_parity_pa_bpe.values()):.2f}" if eng_avg_pa_bpe and avg_parity_pa_bpe else "Parity-Aware BPE: English missing")
    if "byte_level_bpe" in enabled:
        output_lines.append(f"Byte-Level BPE: {max(avg_parity_bl_bpe.values()):.2f}" if eng_avg_bl_bpe and avg_parity_bl_bpe else "Byte-Level BPE: English missing")
    if "blt" in enabled:
        output_lines.append(f"BLT: {max(avg_parity_blt.values()):.2f}" if eng_avg_blt and avg_parity_blt else "BLT: English missing")

    timestamp = time.strftime("%b%d-%H%M", time.localtime())
    output_path = Path(f"/localhome/user/tokenizer_eval_{timestamp}.log")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()