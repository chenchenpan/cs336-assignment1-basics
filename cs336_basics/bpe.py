from __future__ import annotations

import regex as re
import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

# PRETOKENIZE_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")
_GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Exact bytes->printable-unicode remapping used by GPT-2.

    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    # now get the representations of the other 68 integers that do need shifting
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def _iter_pretokens_with_specials(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return _GPT2_PRETOKENIZE_PATTERN.findall(text)

    else:
        pretokens: list[str] = []
        # re.escape() to avoid special tokens being interpreted as regular expression patterns.
        specials_pat = re.compile("|".join(re.escape(s) for s in special_tokens))
        docs = specials_pat.split(text)
        for doc in docs:
            if not doc:
                continue
            pretokens.extend(_GPT2_PRETOKENIZE_PATTERN.findall(doc))
        return pretokens

def _process_chunk(args: tuple[int, int, str, list[str]]) -> dict[tuple[bytes, ...], int]:
    """
    process a single chunk and return word frequencies
    """
    start, end, input_path, special_tokens = args

    # read the chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk = chunk_bytes.decode("utf-8", errors="ignore")

    # pretokenize the chunk
    pretokens = _iter_pretokens_with_specials(chunk, special_tokens)

    # build word frequencies for this chunk
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    for tok in pretokens:
        b = tok.encode("utf-8")
        word = tuple(bytes([c]) for c in b)
        word_freqs[word] += 1

    return dict(word_freqs)

def _get_pair_freqs(pretokens_counter: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_freqs = defaultdict(int)
    for seq, freq in pretokens_counter.items():
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            pair_freqs[(seq[i], seq[i + 1])] += freq
    return pair_freqs

def _best_pair(pair_freqs: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    # best_pair = max(pair_freqs.items(), key=lambda kv: (kv[1], kv[0]))[0]
    """optimized version: avoid creating intermediate tuples"""
    if not pair_freqs:
        return None
    best_pair: tuple[bytes, bytes] | None = None
    best_freq = -1
    for pair, freq in pair_freqs.items():
        if freq > best_freq or (freq == best_freq and best_pair is not None and pair > best_pair):
            best_pair = pair
            best_freq = freq
        elif freq == best_freq and best_pair is None:
            best_pair = pair
            best_freq = freq
    return best_pair


def _apply_merge(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    a, b = pair
    merged_token = a + b
    merged: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            merged.append(merged_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)

def train_bpe(input_path: str | Path, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. initialize vocabulary with single-byte tokens
    byte_to_unicode = gpt2_bytes_to_unicode()
    gpt2_byte_vocab_order = list(byte_to_unicode.keys())

    # 2. initialize vocabulary with special tokens and byte tokens
    vocab: dict[int, bytes] = {}
    # deduplicate special tokens
    special_tokens = list(set(special_tokens))
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    for i, b in enumerate(gpt2_byte_vocab_order):
        vocab[i + len(special_tokens)] = bytes([b])

    next_id = len(special_tokens) + 256 # 256 bytes + special tokens
    num_merges = vocab_size - next_id

    if num_merges <= 0:
        # Can't even fit bytes + specials; still return what we can.
        return vocab, []

    # # 3. build word frequencies (optimized: use set + pre-encode)
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    file_size = Path(input_path).stat().st_size
    input_path_str = str(input_path)  # Convert to string once

    use_parallel = file_size > 10_000_000 # 10MB #100 * 1024 * 1024 # 100MB

    if use_parallel and special_tokens:
        print(f"Debug: using parallel processing for large file: {input_path_str}")
        # parallel processing for large files
        num_prcesses = 4

        with open(input_path, "rb") as f:
            boundary_indices = find_chunk_boundaries(f, num_prcesses, b"<|endoftext|>")
            # boundary_indices = find_chunk_boundaries(f, num_prcesses, special_tokens[0].encode("utf-8"))

        chunk_args = [
            (start, end, input_path_str, special_tokens)
            for start, end in zip(boundary_indices[:-1], boundary_indices[1:])
        ]

        with Pool(num_prcesses) as pool:
            chunk_results = pool.map(_process_chunk, chunk_args)
        for chunk_word_freqs in chunk_results:
            for word, freq in chunk_word_freqs.items():
                word_freqs[word] += freq

    else:
        print(f"Debug: using sequential processing for small file: {input_path_str}")
        text = Path(input_path).read_text(encoding="utf-8")
        pretokens = _iter_pretokens_with_specials(text, special_tokens)

        for tok in pretokens:
            b = tok.encode("utf-8")
            word = tuple(bytes([c]) for c in b)
            word_freqs[word] += 1
    
    
    # 4. train BPE loop    
    merges: list[tuple[bytes, bytes]] = []
    # print(f"Debug: before bpe training: vocab_len: {len(vocab)}")

    for _ in range(num_merges):
        
        pair_freqs = _get_pair_freqs(word_freqs)
        if not pair_freqs:
            break # early exit if no pairs found
        best_pair = _best_pair(pair_freqs)
        if best_pair is None:
            break # early exit if no best pair found

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        # apply merge efficiently
        new_word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            if len(word) < 2:
                # because single-byte words are not merged
                continue
            new_word = _apply_merge(word, best_pair)
            new_word_freqs[new_word] += freq
        word_freqs = new_word_freqs

        if next_id >= vocab_size:
            break

    # print(f"Debug: after bpe training: vocab_len: {len(vocab)}")
    # print(f"Debug: vocab keys: {vocab.keys()}")
    # print(f"Debug: vocab values: {vocab.values()}")
    # print(f"Debug: merges: {merges}")
    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of 
        special tokens. This function should accept the following parameters:
        - vocab: dict[int, bytes]
        - merges: list[tuple[bytes, bytes]]
        - special_tokens: list[str] | None = None
        """
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)

        # normalize and deduplicate specials, preserving order
        self.special_tokens: list[str] = list(dict.fromkeys(special_tokens or []))
        self._special_token_bytes: list[bytes] = [s.encode("utf-8") for s in self.special_tokens]

        # append missing special tokens to vocab
        vocab_values = set(self.vocab.values())
        next_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
        for tok_bytes in self._special_token_bytes:
            if tok_bytes not in vocab_values:
                self.vocab[next_id] = tok_bytes
                vocab_values.add(tok_bytes)
                next_id += 1

        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        # rank of each merge in the order of creation
        self.bpe_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}
        self._bpe_cache: dict[bytes, tuple[bytes, ...]] = {}

        # precompile special-token regex patterns (longest-first to handle overlaps)
        self._special_pattern = None
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile("|".join(re.escape(s) for s in ordered))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Construct a Tokenizer from serialized vocab and merges (GPT-2 format).
        """
        with open(vocab_filepath, "r") as f:
            vocab_json = json.load(f)

        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        def _token_str_to_bytes(token:str) -> bytes:
            return bytes([byte_decoder[ch] for ch in token])

        # detect token->id vs id->token Json shape
        if vocab_json and isinstance(next(iter(vocab_json.values())), int):
            # token -> id
            vocab = {int(idx): _token_str_to_bytes(token) for token, idx in vocab_json.items()}
        else:
            # id -> token
            vocab = {}
            for idx, token in vocab_json.items():
                if isinstance(token, list):
                    vocab[int(idx)] = bytes(token)
                elif isinstance(token, str):
                    vocab[int(idx)] = _token_str_to_bytes(token)
                else:
                    raise TypeError(f"Unsupported vocab token type: {type(token)}")

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if not cleaned_line:
                    continue
                parts = cleaned_line.split()
                if len(parts) != 2:
                    continue
                merges.append((_token_str_to_bytes(parts[0]), _token_str_to_bytes(parts[1])))

        # deduplicate merges, preserving order
        return cls(vocab, merges, special_tokens)


    # helper methods:
    def _iter_pretokens(self, text: str) -> Iterator[tuple[str, bool]]:
        """
        It yields pre‑tokenized pieces of text and flags whether each piece is a special token.
        - If there are no special tokens configured (self._special_pattern is None), it just GPT‑2‑pretokenizes the whole string and yields (token, False) for each.
        - If there are special tokens:
            - It scans the text for them with finditer.
            - For each special token match:
                - It pretokenizes the chunk before the special token and yields each piece as (piece, False).
                - Then it yields the special token itself as (special, True).
            - After the loop, it pretokenizes whatever remaining tail of the string there is.
        So you get a single stream like:
        - normal pieces → (tok, False)
        - special pieces → (tok, True)
        This lets encode treat special tokens as single, unsplittable units while still applying the GPT‑2 pretokenizer to everything else.
        """
        if not self._special_pattern:
            for tok in _GPT2_PRETOKENIZE_PATTERN.findall(text):
                yield tok, False
            return

        pos = 0
        for m in self._special_pattern.finditer(text):
            if m.start() > pos:
                for tok in _GPT2_PRETOKENIZE_PATTERN.findall(text[pos:m.start()]):
                    yield tok, False
            yield m.group(0), True
            pos = m.end()

        if pos < len(text):
            for tok in _GPT2_PRETOKENIZE_PATTERN.findall(text[pos:]):
                yield tok, False

    @staticmethod
    def _get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
        """
        Given a word (sequence of bytes), return a set of all adjacent pairs.
        """
        pairs: set[tuple[bytes, bytes]] = set()
        if not word:
            return pairs
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe_merge(self, token_bytes: bytes) -> tuple[bytes, ...]:
        """
        takes one pretoken as bytes (e.g., b"hello") and applies the 
        learned BPE merges to split it into the final subword tokens (as bytes).
        """
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached

        word = tuple(bytes([b]) for b in token_bytes)
        if len(word) <= 1:
            self._bpe_cache[token_bytes] = word
            return word
        
        pairs = self._get_pairs(word)
        while True:
            best_pair = None
            best_rank = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            
            if best_pair is None:
                break

            new_word = _apply_merge(word, best_pair)
            word = new_word
            if len(word) <= 1:
                break
            pairs = self._get_pairs(word)

        self._bpe_cache[token_bytes] = word
        return word

    def _encode_text_iter(self, text:str) -> Iterator[int]:
        for tok, is_special in self._iter_pretokens(text):
            if is_special:
                tok_bytes = tok.encode("utf-8")
                yield self.token_to_id[tok_bytes]
            else:
                tok_bytes = tok.encode("utf-8")
                subwords = self._bpe_merge(tok_bytes)
                for subword in subwords:
                    yield self.token_to_id[subword]
    
    def encode(self, text: str) -> list[int]:
        return list(self._encode_text_iter(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode_text_iter(text)

    def decode(self, ids: list[int]) -> str:
        byte_chunks = [self.vocab[id] for id in ids]
        return b"".join(byte_chunks).decode("utf-8", errors="replace")
        