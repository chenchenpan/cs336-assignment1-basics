from __future__ import annotations

import regex as re
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
