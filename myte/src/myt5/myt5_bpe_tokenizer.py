# coding=utf-8
# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model MyT5 + BPE."""


import warnings
from typing import Dict, List, Optional, Tuple, Union

from tokenizers import Tokenizer as HFTokenizer
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging
import json
import os

from collections import defaultdict

logger = logging.get_logger(__name__)


class ByteRewriter:

    LEAF ='[LEAF]'

    def __init__(self, rewriting_rules: Union[str, Dict[str, str]]):

        if type(rewriting_rules) == str:
            with open(rewriting_rules, "r") as f:
                rewriting_rules = json.load(f)
        elif not type(rewriting_rules) == dict:
            raise ValueError(f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}")

        self.hash_tree = self.construct_hash_tree(rewriting_rules)
        reverse_rewriting_rules = {v:k for k,v in rewriting_rules.items()}
        self.reverse_hash_tree = self.construct_hash_tree(reverse_rewriting_rules)

    def add_leaf(self,hash_tree, byte_in_sequence, byte_out_sequence):

        byte_in_list = byte_in_sequence.split(' ')
        byte_out_list = byte_out_sequence.split(' ')

        tree_pointer = hash_tree
        for b in byte_in_list:
            if b not in tree_pointer:
                tree_pointer[b] = {}
            tree_pointer = tree_pointer[b]

        tree_pointer[self.LEAF] = byte_out_list

    def construct_hash_tree(self, rewriting_rules):

        hash_tree = defaultdict(dict)
        for b in (f"{x:02x}" for x in range(256)):
             hash_tree[b][self.LEAF] = [b]

        for in_sequence, out_sequence in rewriting_rules.items():
            self.add_leaf(hash_tree, in_sequence, out_sequence)

        return hash_tree

    def rewrite_bytes(self, in_bytes, reverse=False):

        out_bytes = []
        b_start = 0

        while b_start < len(in_bytes):
            tree_pointer = self.hash_tree if not reverse else self.reverse_hash_tree
            cur_leaf = None
            match_end = b_start -1 # Track end of the longest match

            # Find the longest match starting from b_start
            for j in range(b_start, len(in_bytes)):
                b = in_bytes[j]
                if b in tree_pointer:
                    tree_pointer = tree_pointer[b]
                    # If this node is a valid end-point, record it
                    if self.LEAF in tree_pointer:
                        cur_leaf = tree_pointer[self.LEAF]
                        match_end = j
                else:
                    # Byte not in tree, stop searching for this starting position
                    break

            # If no match was found (not even single byte), handle error/skip
            if cur_leaf is None:
                 # Default to the single byte if no rule applied
                 if b_start < len(in_bytes):
                     cur_leaf = [in_bytes[b_start]]
                     match_end = b_start
                 else:
                     break # Should not happen if len(in_bytes) > 0

            out_bytes.extend(cur_leaf)
            b_start = match_end + 1

        return out_bytes


class MyT5BPETokenizer(PreTrainedTokenizer):
    """
    Construct a MyT5 tokenizer with optional BPE on morphological sequences.

    Args:
        decompose_map (`str` or `dict`): Path or dict for decomposition rules.
        merge_map (`str` or `dict`): Path or dict for merging rules.
        bpe_tokenizer_path (`str`, *optional*): Path to the pre-trained BPE tokenizer JSON file.
        eos_token (`str`, *optional*, defaults to `"</s>"`): End of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`): Unknown token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`): Padding token.
        extra_ids (`int`, *optional*, defaults to 125): Number of extra sentinel ids.
        additional_special_tokens (`List[str]`, *optional*): Additional special tokens.
    """

    model_input_names = ["input_ids", "attention_mask"]

    MERGE_MAP_REL_PATH = "byte_maps/merge_map.json"
    DECOMPOSE_MAP_REL_PATH = "byte_maps/decompose_map.json"

    def __init__(
            self,
            decompose_map=None,
            merge_map=None,
            bpe_tokenizer_path="./100000_16000.json",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=125,
            additional_special_tokens=None,
            **kwargs,
    ) -> None:

        # --- Resolve map paths ---
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if decompose_map is None:
            decompose_map = os.path.join(dir_path, self.DECOMPOSE_MAP_REL_PATH)
        if merge_map is None:
            merge_map = os.path.join(dir_path, self.MERGE_MAP_REL_PATH)
        bpe_tokenizer_path = os.path.join(dir_path, bpe_tokenizer_path)

        # --- Handle extra_ids and special tokens ---
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                 raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided. Ensure additional_special_tokens includes the extra_ids tokens."
                )

        # --- Load BPE Tokenizer (if provided) ---
        self.bpe_tokenizer = None
        if bpe_tokenizer_path:
            if os.path.exists(bpe_tokenizer_path):
                self.bpe_tokenizer = HFTokenizer.from_file(bpe_tokenizer_path)
                logger.info(f"Loaded BPE tokenizer from {bpe_tokenizer_path}")
            else:
                logger.warning(f"BPE tokenizer path specified but not found: {bpe_tokenizer_path}")

        # --- Initialize ByteRewriters ---
        self.decompose_rewriter = ByteRewriter(decompose_map)
        self.merge_rewriter = ByteRewriter(merge_map)

        # --- Initialize PreTrainedTokenizer ---
        # If using BPE, let BPE handle vocab size and special tokens internally for conversion
        # PreTrainedTokenizer still manages the overall special token behavior (adding eos, etc.)
        if not self.bpe_tokenizer:
             # Original setup for byte-level without BPE
             pad_token = AddedToken(pad_token, lstrip=True, rstrip=True) if isinstance(pad_token, str) else pad_token
             eos_token = AddedToken(eos_token, lstrip=True, rstrip=True) if isinstance(eos_token, str) else eos_token
             unk_token = AddedToken(unk_token, lstrip=True, rstrip=True) if isinstance(unk_token, str) else unk_token
             # Store special tokens defined by T5, offset accordingly
             self._added_tokens_decoder = {0: pad_token, 1: eos_token, 2: unk_token}
             self.offset = len(self._added_tokens_decoder)
             self._utf_vocab_size = 2**8
        else:
             # When using BPE, offset and byte vocab size are not directly used for main vocab
             self.offset = 0 # BPE handles its own vocab mapping
             self._utf_vocab_size = 0 # Not relevant when BPE defines vocab size

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=0, # extra_ids are handled by adding them to additional_special_tokens
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        if self.bpe_tokenizer:
            return self.bpe_tokenizer.get_vocab_size()
        else:
            # Vocab size is 256 bytes + number of special tokens (pad, eos, unk)
            return self._utf_vocab_size + self.offset

    def get_vocab(self):
        if self.bpe_tokenizer:
            # Return BPE vocab + added special tokens by PreTrainedTokenizer
            vocab = self.bpe_tokenizer.get_vocab()
            vocab.update(self.added_tokens_encoder)
            return vocab
        else:
            # Original byte vocab + added special tokens
            vocab = {self._convert_id_to_token(i): i for i in range(self.offset, self.vocab_size)}
            vocab.update({str(t): i for i, t in self._added_tokens_decoder.items()}) # Add pad, eos, unk
            vocab.update(self.added_tokens_encoder) # Add extra_ids etc.
            return vocab

    # --- Methods using BPE ---

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """ Tokenize: text -> bytes -> hex -> morphological -> [BPE] -> final tokens (strings) """
        tokens = [f"{i:02x}" for i in text.encode("utf-8")]
        tokens = self.morphological_encode(tokens)

        if self.bpe_tokenizer:
            # BPE expects a single string, join morpho tokens with space
            morpho_string = " ".join(tokens)
            bpe_output = self.bpe_tokenizer.encode(morpho_string)
            tokens = bpe_output.tokens # Return BPE tokens (strings)
        # else: tokens remain the morphologically encoded hex strings

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """
        if self.bpe_tokenizer:
            # Use BPE vocab mapping
            return self.bpe_tokenizer.token_to_id(token) or self.unk_token_id # Return unk_token_id if not found
        else:
            # Original byte logic
            if token in self._added_tokens_decoder.values():
                 # Find the ID for pad, eos, unk
                 for i, t in self._added_tokens_decoder.items():
                     if t == token:
                         return i
                 return self.unk_token_id # Should not happen if setup is correct
            elif len(token) == 2:
                 try:
                     # Convert hex byte token to ID with offset
                     return int(token, 16) + self.offset
                 except ValueError:
                     return self.unk_token_id
            else:
                 # Handle added special tokens (like <extra_id_...>)
                 return self.added_tokens_encoder.get(token, self.unk_token_id)


    def _convert_id_to_token(self, index: int) -> str:
        """ Converts an index (integer) in a token (str) using the vocab. """
        if self.bpe_tokenizer:
            # Use BPE vocab mapping
            return self.bpe_tokenizer.id_to_token(index) or self.unk_token # Return unk_token if not found
        else:
            # Original byte logic
            if index in self._added_tokens_decoder:
                 # Handle pad, eos, unk
                 return str(self._added_tokens_decoder[index])
            elif self.offset <= index < self.vocab_size:
                 # Convert ID back to hex byte token
                 return f"{index - self.offset:02x}"
            else:
                 # Handle added special tokens (like <extra_id_...>)
                 return self.added_tokens_decoder.get(index, self.unk_token)

    # --- Core Decoding Logic ---

    def _decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True, # Changed default
        **kwargs,
    ) -> str:
        # This is the main function called by decode()
        # It receives IDs potentially including special tokens

        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i not in self.all_special_ids]

        if not token_ids:
            return ""

        if self.bpe_tokenizer:
            # Decode BPE IDs to the space-separated morpho-hex string
            # Note: BPE decode often handles cleanup of its own internal spaces.
            # We might need to adjust clean_up_tokenization_spaces behavior.
            decoded_morpho_hex_string = self.bpe_tokenizer.decode(token_ids)
            # Split back into individual morpho-hex tokens
            morpho_hex_tokens = decoded_morpho_hex_string.split(' ')
        else:
            # Convert non-BPE IDs back to morpho-hex tokens (strings)
            morpho_hex_tokens = [self._convert_id_to_token(idx) for idx in token_ids]

        # Perform morphological decoding (common step)
        hex_tokens = self.morphological_decode(morpho_hex_tokens)

        # Convert final hex tokens to bytes
        bstring = b""
        for token in hex_tokens:
            if len(token) == 2: # Basic check for hex byte format
                try:
                    bstring += bytes.fromhex(token)
                except ValueError:
                    # Handle cases where a token after decoding isn't a valid hex byte
                    # This might happen with malformed rules or unexpected BPE output
                    logger.warning(f"Could not convert token '{token}' to byte, skipping.")
                    # Optionally append a replacement character or handle differently
            else:
                # This case should ideally not happen if morphological_decode is correct
                # and BPE decode produces expected space-separated hex.
                # If it does, it might be parts of special tokens or errors.
                logger.warning(f"Unexpected token format after morphological decode: '{token}', attempting utf-8.")
                # Try encoding directly, might be part of a special token string
                try:
                    bstring += token.encode('utf-8')
                except UnicodeEncodeError:
                     logger.warning(f"Could not encode token '{token}' as utf-8, skipping.")


        # Decode final byte string to text
        string = bstring.decode("utf-8", errors="ignore") # Or use 'replace'

        # Optional: Further cleanup (though BPE decode might handle some)
        if clean_up_tokenization_spaces:
             # Example: Replace multiple spaces with one, strip leading/trailing
             string = " ".join(string.split())

        return string


    # --- Morphological Encoding/Decoding ---

    def morphological_encode(self, indices: List[str]) -> List[str]:
        # Decompose and merge morphological sequences
        indices = self.decompose_rewriter.rewrite_bytes(indices, reverse=False)
        indices = self.merge_rewriter.rewrite_bytes(indices, reverse=False)
        return indices

    def morphological_decode(self, indices: List[str]) -> List[str]:
        # Demerge and compose morphological sequences
        indices = self.merge_rewriter.rewrite_bytes(indices, reverse=True)
        indices = self.decompose_rewriter.rewrite_bytes(indices, reverse=True)
        return indices

    # --- Overrides for PreTrainedTokenizer ---

    # Removed convert_tokens_to_string - rely on decode -> _decode

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # If using BPE, save its vocab. Otherwise, nothing to save for byte-level.
        if self.bpe_tokenizer:
            if not os.path.isdir(save_directory):
                logger.error(f"Vocabulary path ({save_directory}) should be a directory")
                return ()
            out_vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + "myt5-bpe-tokenizer.json"
            )
            self.bpe_tokenizer.save(out_vocab_file)
            logger.info(f"BPE tokenizer vocabulary saved to {out_vocab_file}")
            return (out_vocab_file,)
        else:
            # MyT5 byte-level has no explicit vocab file
            return ()

    # --- Keep other necessary overrides like build_inputs_with_special_tokens etc. ---
    # (These seem okay as they operate on IDs after tokenization)

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """ Standard implementation often works, assuming EOS is added correctly. """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1] # For X </s>
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1] # For A </s> B </s>

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """ Adds EOS if not already present. """
        if not token_ids or token_ids[-1] != self.eos_token_id:
            return token_ids + [self.eos_token_id]
        # Warn if EOS is already there? Base class might handle this.
        # warnings.warn(...)
        return token_ids

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """ T5 family models don't use token type IDs. """
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """ Builds model inputs by adding EOS. Format: X </s> or A </s> B </s> """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1