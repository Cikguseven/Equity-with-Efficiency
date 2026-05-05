import json
import os
from pathlib import Path

from tqdm import tqdm


def count_lines_in_file(file_path):
    """Helper to get line count for a single file"""
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)


def process_single_file(input_file, output_dir, chunk_size=100_000):
    """Process a single txt file into chunked JSONL files"""
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    # Use the input filename (without extension) as the stem for output files
    stem = input_file.stem
    suffix = ".jsonl"

    total_lines = count_lines_in_file(input_file)

    current_chunk_idx = 0
    lines_in_current_chunk = 0
    f_out = None

    try:
        with tqdm(total=total_lines, unit='lines', desc=f"Converting {input_file.name}") as pbar:
            with input_file.open("r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    pbar.update(1)

                    line = line.strip()
                    if not line:
                        continue

                    # Open a new chunk file if we don't have one open
                    if f_out is None:
                        chunk_filename = f"{stem}.chunk.{current_chunk_idx:02d}{suffix}"
                        chunk_path = output_dir / chunk_filename
                        f_out = chunk_path.open("w", encoding="utf-8")

                    # Write the JSON object
                    obj = {"text": line}
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    lines_in_current_chunk += 1

                    # Rotate file if chunk size limit is reached
                    if lines_in_current_chunk >= chunk_size:
                        f_out.close()
                        f_out = None
                        lines_in_current_chunk = 0
                        current_chunk_idx += 1

    finally:
        if f_out:
            f_out.close()


def txt_dir_to_jsonl_split(input_dir, output_dir, chunk_size=100_000, pattern="*.txt"):
    """Process all matching files in input_dir and split them into JSONL chunks"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Find all files matching the pattern
    input_files = sorted(input_dir.glob(pattern))

    if not input_files:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return

    print(f"Found {len(input_files)} file(s) to process")

    # Process each file
    for input_file in input_files:
        print(f"\nProcessing: {input_file.name}")
        process_single_file(input_file, output_dir, chunk_size)

    print(f"\nAll done! Output written to {output_dir}")


if __name__ == "__main__":
    txt_dir_to_jsonl_split(
        input_dir="/scratch/Projects/1/1/user/data/mc4_SEA_1M_sentences",
        output_dir="/scratch/Projects/1/1/user/data/mc4_SEA_1M_sentences_blt",
        chunk_size=1000,
        pattern="*.txt"  # Change to "*" for all files, or "combined*.txt" for specific pattern
    )
