import sys
sys.path.insert(0, "/scratch/Projects/1/1/user")

import time
import json
from pathlib import Path
from typing import List, Any

import torch

from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum

JSONL_DIR = Path("/scratch/Projects/1/1/user/data/fineweb2_SEA_100M_sentences_blt")

BLT_ENTROPY_MODEL_DIR = "/scratch/Projects/1/1/user/blt/blt-entropy"
BLT_CHECKPOINT_PATH = "/scratch/Projects/1/1/user/blt/hf-weights/blt_1b"

def blt_patch_counts_per_sentence(tokenizer: Any, patcher: Any, lines: List[str]) -> List[int]:
    """Encodes a list of text prompts and returns the number of BLT patches for each."""
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


def read_jsonl_texts(fp: Path) -> List[str]:
    """Extracts the 'text' field from each line in a JSONL file."""
    lines: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        lines.append(obj["text"])
                except json.JSONDecodeError:
                    continue
    return lines


def main() -> None:
    if not JSONL_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {JSONL_DIR}")

    print("Loading BLT model...")
    checkpoint_path = Path(BLT_CHECKPOINT_PATH)
    _, blt_tokenizer, _ = load_consolidated_model_and_tokenizer(checkpoint_path)

    print("Initializing BLT Patcher...")
    entropy_model_dir = Path(BLT_ENTROPY_MODEL_DIR)
    patcher_args = PatcherArgs(
        patching_mode=PatchingModeEnum.entropy,
        realtime_patching=True,
        entropy_model_checkpoint_dir=str(entropy_model_dir),
        patching_device="cuda:0",
        device="cuda:0",
    )
    blt_patcher = patcher_args.build()

    # Find all .jsonl files
    files_to_process = sorted([
        entry for entry in JSONL_DIR.iterdir()
        if entry.is_file() and entry.suffix == ".jsonl"
    ])

    if not files_to_process:
        print(f"No .jsonl files found in {JSONL_DIR}")
        return

    print(f"Found {len(files_to_process)} JSONL files to process.\n")

    total_patches_all_files = 0
    total_sentences_all_files = 0

    output_lines: List[str] = []
    output_lines.append(f"BLT Patch Counting started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    output_lines.append(f"Target Directory: {JSONL_DIR}\n")

    # Process each file
    for idx, entry in enumerate(files_to_process, 1):
        texts = read_jsonl_texts(entry)
        if not texts:
            continue

        counts = blt_patch_counts_per_sentence(blt_tokenizer, blt_patcher, texts)
        counts = [c for c in counts if c > 0]

        file_total_patches = sum(counts)
        file_total_sentences = len(counts)

        total_patches_all_files += file_total_patches
        total_sentences_all_files += file_total_sentences

        log_msg = f"[{idx:03d}/{len(files_to_process):03d}] {entry.name}: {file_total_patches} patches across {file_total_sentences} sentences."
        print(log_msg)
        output_lines.append(log_msg)

    # Summary Stats
    summary = (
        f"\n--- Final Summary ---\n"
        f"Total JSONL files processed: {len(files_to_process)}\n"
        f"Total sentences patched: {total_sentences_all_files}\n"
        f"Total BLT patches across all files: {total_patches_all_files}"
    )
    print(summary)
    output_lines.append(summary)

    # Write output log
    timestamp = time.strftime("%b%d-%H%M", time.localtime())
    output_path = Path(f"/scratch/Projects/1/1/user/blt_patch_counts_{timestamp}.log")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"\nLog saved to {output_path}")

if __name__ == "__main__":
    main()