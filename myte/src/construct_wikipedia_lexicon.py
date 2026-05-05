import argparse
import codecs
from collections import defaultdict
from datasets import load_dataset, Dataset
import nltk

from rewrite_bytes import ByteRewriter
from utils import  str_to_hex
from tqdm import tqdm

import multiprocessing
from functools import partial


def save_lexicon(lex_counted, language, lex_directory):
    # if lex_directory is not None:
    with codecs.open(f"{lex_directory}/{language}_lex.txt", "w") as lexicon_file:
        for b_lex, count in lex_counted.items():
            lexicon_file.write(f"{count}\t{b_lex}\n")
    print(f"Lexicon saved to {lex_directory}/{language}_lex.txt")


def process_wikipedia_example(batch, rewriter, lexeme_rewriter, reverse_lexeme_codes):
    partial_lexeme_count = defaultdict(int)
    example = "\n".join(batch["text"])
    bytes_processed = len(example.encode("utf-8"))
    tokenized_txt = nltk.word_tokenize(example.replace("\n"," "))
    bytes_normalized = rewriter.rewrite_bytes(str_to_hex(" ".join(tokenized_txt)).split(' '))
    bytes_lexemized = lexeme_rewriter.rewrite_bytes(bytes_normalized)
    lexem_ids = [tok for tok in bytes_lexemized if tok.startswith("lex_")]

    for lid in lexem_ids:
        partial_lexeme_count[reverse_lexeme_codes[lid]] += 1

    lexemes = list(partial_lexeme_count.keys())
    counts = list(partial_lexeme_count.values())
    return {'lexemes': [lexemes], 'counts': [counts], 'bytes_processed': [bytes_processed]}

def process_wikipedia_example_no_lexicon(batch, rewriter):
    partial_lexeme_count = defaultdict(int)
    example = "\n".join(batch["text"])
    bytes_processed = len(example.encode("utf-8"))
    tokenized_txt = nltk.word_tokenize(example.replace("\n", " "))

    for token in tokenized_txt:
        token_normalized = rewriter.rewrite_bytes(str_to_hex(token).split(' '))
        partial_lexeme_count[" ".join(token_normalized)] += 1

    lexemes = list(partial_lexeme_count.keys())
    counts = list(partial_lexeme_count.values())
    return {'lexemes': [lexemes], 'counts': [counts], 'bytes_processed': [bytes_processed]}


def count_in_corpus(language: str, lexeme_count: dict[str, int], rewriter: ByteRewriter, no_lexicon: bool, corpus: str, trust_remote_code: bool = False) -> dict[str, int]:

    lexeme_codes = {lex: f"lex_{lid}" for lid, lex in enumerate(lexeme_count.keys())}
    reverse_lexeme_codes = {v:k for k,v in lexeme_codes.items()}
    lexeme_rewriter = ByteRewriter(lexeme_codes)

    batch_size = 1000
    max_dataset_size = 2500000

    if corpus == 'wikipedia':
        dataset = load_dataset('wikimedia/wikipedia', f"20231101.{language}", split='train', streaming=True, trust_remote_code=trust_remote_code)
        dataset = dataset.take(max_dataset_size)

        print("Downloading and materializing dataset for parallel processing...")
        dataset = Dataset.from_list(list(tqdm(dataset, total=max_dataset_size, desc="Downloading")))
    else:
        raise ValueError(f"Only Wikipedia supported")

    if no_lexicon:
        process_func = partial(
            process_wikipedia_example_no_lexicon,
            rewriter=rewriter
        )
    else:
        process_func = partial(
            process_wikipedia_example,
            rewriter=rewriter,
            lexeme_rewriter=lexeme_rewriter,
            reverse_lexeme_codes=reverse_lexeme_codes
        )

    num_cores = multiprocessing.cpu_count()
    if num_cores > 4:
        num_cores = num_cores // 2  # leave some cores free for system/IO
    else:
        num_cores = 1 # Fallback for small machines

    dataset = dataset.map(
        process_func,
        batched=True,
        batch_size=batch_size,
        num_proc=num_cores,
        remove_columns=list(dataset.features.keys()) # clear text columns to save RAM
    )

    total_bytes = 0
    # Aggregate results
    # Note: dataset is now a list of {'lexemes': [...], 'counts': [...], 'bytes_processed': ...}
    for batch in tqdm(dataset, desc="Aggregating counts"):
        for lexeme, count in zip(batch['lexemes'], batch['counts']):
            lexeme_count[lexeme] += count
        total_bytes += batch['bytes_processed']

    print(f"Total bytes processed for {language}: {total_bytes}")

    return lexeme_count


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--language", help="language code to process", default="ro")
    argparser.add_argument("--lexicon_directory", help="directory with lexicon files", default="../lexicons")
    argparser.add_argument("--output_directory", help="directory for output *_lex.txt files", default="../lexicons")
    argparser.add_argument("--pre_processing_file", help="file with processing parameters", default="../byte_maps/decompose.json")
    argparser.add_argument("--do_capitalize", help="if capitalize lexemes", action='store_true', default=False)
    argparser.add_argument("--no_lexicon", help="do not save lexicon", action="store_true", default=False)
    argparser.add_argument("--min_occurrences", help="minimum occurrences for a lexeme", default=0, type=int)
    argparser.add_argument("--lexicon_size", help="Pre-set lexicon size", default=30000, type=int)
    argparser.add_argument("--filter_en", help="filter english words", action="store_true", default=False)
    argparser.add_argument("--trust_remote_code", help="allow HuggingFace datasets to execute remote dataset loading code (required for wikipedia)", action="store_true", default=False)

    args = argparser.parse_args()

    # word counts and save as lexicon
    lexeme_counts = defaultdict(int)
    pp_rewriter = ByteRewriter(args.pre_processing_file)

    no_lexicon = args.no_lexicon
    if not args.no_lexicon and args.lexicon_directory is not None:
        try:
            with open(f"{args.lexicon_directory}/{args.language}_dict.txt", "r") as dictionary_file:
                dictionary_lines = dictionary_file.readlines()
                for line in dictionary_lines:
                    en_lexeme, lexeme = line.split()
                    if args.filter_en and en_lexeme == lexeme and args.language != 'en':
                        # filter words that are the same in english
                        continue
                    lexeme_counts[lexeme] = 0
                    if args.do_capitalize:
                        lexeme_counts[lexeme.capitalize()] = 0
        except FileNotFoundError:
            print(f"Warning: {args.language}_dict.txt not found, creating new lexicon")
            no_lexicon = True

    # hexify and normalize lexemes
    if not no_lexicon:
        lexeme_counts = {" ".join(pp_rewriter.rewrite_bytes(str_to_hex(lexeme).split(' '))): count for lexeme, count in tqdm(lexeme_counts.items(), desc="Normalizing lexemes")}

    lexeme_counts = count_in_corpus(args.language, lexeme_counts, pp_rewriter, no_lexicon=no_lexicon, corpus='wikipedia', trust_remote_code=args.trust_remote_code)

    # sort and filter lexemes
    lexeme_counts = {lexeme: count for lexeme, count in lexeme_counts.items() if count >= args.min_occurrences}

    if args.lexicon_size is not None:
        lexeme_counts = {lexeme: count for lexeme, count in sorted(lexeme_counts.items(), key=lambda x: x[1], reverse=True)[:args.lexicon_size]}
    else:
        lexeme_counts = {lexeme: count for lexeme, count in sorted(lexeme_counts.items(), key=lambda x: x[1], reverse=True)}

    save_lexicon(lexeme_counts, args.language, args.output_directory)