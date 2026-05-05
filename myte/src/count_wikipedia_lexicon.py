import argparse
import multiprocessing
from functools import partial
from datasets import load_dataset
from tqdm import tqdm

def count_bytes_in_corpus(language: str, trust_remote_code: bool = False) -> int:
    max_dataset_size = 2500000
    batch_size = 1000
    total_bytes = 0

    try:
        # Load in streaming mode
        dataset = load_dataset(
            'wikimedia/wikipedia',
            f"20231101.{language}",
            split='train',
            streaming=True,
            trust_remote_code=trust_remote_code
        )

        dataset = dataset.take(max_dataset_size)

        # We leave the inner tqdm enabled so you can see progress on large files (like 'en')
        # We use leave=False so finished languages disappear from the console to reduce clutter
        for batch in dataset.iter(batch_size=batch_size):
            # Calculate byte size of the UTF-8 encoded text
            batch_bytes = sum(len(text.encode("utf-8")) for text in batch["text"])
            total_bytes += batch_bytes

    except Exception as e:
        print(f"Error processing '{language}': {e}")
        return 0

    # Print summary for this language upon completion
    print(f"Finished {language}: {total_bytes / (1024**3):.2f} GB")
    return total_bytes

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--languages", nargs='+', help="list of language codes to process", required=True)
    argparser.add_argument("--trust_remote_code", help="allow remote code execution for HF datasets", action="store_true", default=False)
    # Default to CPU count - 1 to keep system responsive, or 4 as a safe default
    default_workers = multiprocessing.cpu_count() // 8
    argparser.add_argument("--workers", type=int, help="number of parallel processes", default=default_workers)

    args = argparser.parse_args()

    print(f"Starting parallel processing with {args.workers} workers...")

    # Create a partial function with the fixed 'trust_remote_code' argument
    worker_func = partial(count_bytes_in_corpus, trust_remote_code=args.trust_remote_code)

    grand_total_bytes = 0

    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=args.workers) as pool:
        # imap_unordered allows us to process results as soon as they finish
        # We don't need a progress bar for the languages themselves since the inner functions have bars
        for result in pool.imap_unordered(worker_func, args.languages):
            grand_total_bytes += result

    print("=" * 40)
    print(f"Grand Total Bytes for {args.languages}: {grand_total_bytes}")
    print(f"Grand Total Size for {args.languages}:  {grand_total_bytes / (1024**3):.2f} GB")
    print("=" * 40)
