import argparse
import codecs
from collections import defaultdict
import nltk
import os

from rewrite_bytes import ByteRewriter
from utils import  str_to_hex
from tqdm import tqdm


def save_lexicon(lex_counted, language, lex_directory):

    # if lex_directory is not None:
    with codecs.open(f"{lex_directory}/{language}_lex.txt", "w") as lexicon_file:
        for b_lex, count in lex_counted.items():
            lexicon_file.write(f"{count}\t{b_lex}\n")
    print(f"Lexicon saved to {lex_directory}/{language}_lex.txt")


def count_in_corpus(language: str, lexeme_count: dict[str, int], rewriter: ByteRewriter, no_lexicon: bool, corpus_path: str) -> dict[str, int]:

    lexeme_codes = {lex: f"lex_{lid}" for lid, lex in enumerate(lexeme_count.keys())}
    reverse_lexeme_codes = {v:k for k,v in lexeme_codes.items()}
    lexeme_rewriter = ByteRewriter(lexeme_codes)

    batch_size = 10000
    max_dataset_size = 1000000

    if corpus_path is None:
        raise ValueError("corpus_path must be provided for C4 corpus")

    c4_file = os.path.join(corpus_path, f"{language}.txt")
    if not os.path.exists(c4_file):
        raise FileNotFoundError(f"C4 file not found: {c4_file}")

    print(f"Loading C4 data from {c4_file}")

    # Read file and create batches
    def read_c4_batches():
        with open(c4_file, 'r', encoding='utf-8') as f:
            batch = []
            line_count = 0
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                    line_count += 1
                    if len(batch) >= batch_size:
                        yield {'text': batch}
                        batch = []
                    if line_count >= max_dataset_size:
                        break
            if batch:
                yield {'text': batch}

    dataset = read_c4_batches()

    def process_example(batch):
        partial_lexeme_count = defaultdict(int)
        #for example in batch["text"]:
        example = "\n".join(batch["text"])
        tokenized_txt = nltk.word_tokenize(example.replace("\n"," "))
        bytes_normalized = rewriter.rewrite_bytes(str_to_hex(" ".join(tokenized_txt)).split(' '))
        bytes_lexemized = lexeme_rewriter.rewrite_bytes(bytes_normalized)
        # find lexem ids in the text
        lexem_ids = [tok for tok in bytes_lexemized if tok.startswith("lex_")]

        for lid in lexem_ids:
            partial_lexeme_count[reverse_lexeme_codes[lid]] += 1
        return {'lexeme_count': [[(lex, count) for lex, count in partial_lexeme_count.items()]]}

    def process_example_no_lexicon(batch):
        partial_lexeme_count = defaultdict(int)

        example = "\n".join(batch["text"])
        tokenized_txt = nltk.word_tokenize(example.replace("\n", " "))
        for token in tokenized_txt:
            token_normalized = rewriter.rewrite_bytes(str_to_hex(token).split(' '))
            partial_lexeme_count[" ".join(token_normalized)] += 1
        return {'lexeme_count': [[(lex, count) for lex, count in partial_lexeme_count.items()]]}


    for batch in tqdm(dataset, desc="Processing C4 lexemes"):
        if no_lexicon:
            result = process_example_no_lexicon(batch)
        else:
            result = process_example(batch)

        for lexeme, count in result['lexeme_count'][0]:
            lexeme_count[lexeme] += count

    return lexeme_count


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--language", help="language code to process", default="ro")
    argparser.add_argument("--lexicon_directory", help="directory with lexicon files", default="../lexicons")
    argparser.add_argument("--output_directory", help="directory for output *_lex.txt files", default="../lexicons")
    argparser.add_argument("--corpus_path", help="path to multilingual C4 files", default=None)
    argparser.add_argument("--pre_processing_file", help="file with processing parameters", default="../byte_maps/decompose.json")
    argparser.add_argument("--do_capitalize", help="if capitalize lexemes", action='store_true', default=False)
    argparser.add_argument("--no_lexicon", help="do not save lexicon", action="store_true", default=False)
    argparser.add_argument("--min_occurrences", help="minimum occurrences for a lexeme", default=0, type=int)
    argparser.add_argument("--lexicon_size", help="Pre-set lexicon size", default=30000, type=int)
    argparser.add_argument("--filter_en", help="filter english words", action="store_true", default=False)


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
    lexeme_counts = count_in_corpus(args.language, lexeme_counts, pp_rewriter, no_lexicon=no_lexicon,corpus_path=args.corpus_path)
    # sort and filter lexemes
    lexeme_counts = {lexeme: count for lexeme, count in lexeme_counts.items() if count >= args.min_occurrences}
    if args.lexicon_size is not None:
        lexeme_counts = {lexeme: count for lexeme, count in sorted(lexeme_counts.items(), key=lambda x: x[1], reverse=True)[:args.lexicon_size]}
    else:
        lexeme_counts = {lexeme: count for lexeme, count in sorted(lexeme_counts.items(), key=lambda x: x[1], reverse=True)}

    save_lexicon(lexeme_counts, args.language, args.output_directory)