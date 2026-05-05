import argparse
import json
import os

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from transformers import PreTrainedTokenizerFast


def build_vocab_from_merges(merges):
    if merges[0].startswith("#version:"):
        merges = merges[1:]
    vocab = {char: idx for idx, char in enumerate(ByteLevel.alphabet())}
    index = len(vocab)

    def ensure(token):
        nonlocal index
        if token not in vocab:
            vocab[token] = index
            index += 1

    for line in merges:
        token1, token2 = line.split()
        token1, token2 = token1.strip(), token2.strip()
        ensure(token1)
        ensure(token2)
        ensure(token1 + token2)
    return vocab

def load_custom_tokenizer(tokenizer_path: str):
    merge_file = os.path.join(tokenizer_path, "merges.txt")
    vocab_file = os.path.join(tokenizer_path, "vocab.json")
    tokenizer = Tokenizer(BPE.from_file(vocab_file, merge_file))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False)])
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token= "</s>",
    )
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    wrapped_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return wrapped_tokenizer


def create_huggingface_tokenizer(merges_file_path, tokenizer_path):
    """ Create a HuggingFace tokenizer from a merges file.
    Args:
        merges_file_path (str): Path to the merges file.
        tokenizer_path (str): Path to save the tokenizer files.
    Returns:
        PreTrainedTokenizerFast: A HuggingFace tokenizer.
    """

    merges = []
    with open(merges_file_path, "r", encoding="utf-8") as f:
        merges = f.readlines()

    vocab = build_vocab_from_merges(merges)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    vocab_file_path = os.path.join(tokenizer_path, "vocab.json")
    with open(vocab_file_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    merges_file_path = os.path.join(tokenizer_path, "merges.txt")
    with open(merges_file_path, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(merge)
    tokenizer = load_custom_tokenizer(tokenizer_path)
    return tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a HuggingFace tokenizer from a merges file.")
    parser.add_argument("--merges_file_path", type=str, required=True, help="Path to the merges file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to save the tokenizer files.")
    args = parser.parse_args()

    tokenizer = create_huggingface_tokenizer(args.merges_file_path, args.tokenizer_path)
    os.makedirs(args.tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_path)

    print(f"Tokenizer created and saved to {args.tokenizer_path}")
    print("You can now use this tokenizer with HuggingFace models.")