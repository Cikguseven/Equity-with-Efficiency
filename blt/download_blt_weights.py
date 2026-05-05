import os

import typer
from huggingface_hub import snapshot_download


def main():
    if not os.path.exists("hf-weights"):
        os.makedirs("hf-weights")
    snapshot_download(f"facebook/blt", local_dir=f"hf-weights", allow_patterns="blt_7b/**")


if __name__ == "__main__":
    typer.run(main)
