import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Iterator
from collections import Counter
import regex as re
import json
from multiprocessing import Pool



# helper func copied from test code
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"),
                                                          ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


class BPETrainer():
    def __init__(self, master_count: Counter, file_path: str,  special_tokens: list[bytes] = [], num_processes=8, vocab_size=10000):
        self.master_count = master_count
        self.vocab_size = vocab_size

        # if we have user input of speical tokens then update
        self.special_tokens = [tok.encode('utf-8') for tok in special_tokens]
        self.escaped_tokens = [re.escape(tok)
                               for tok in special_tokens] if special_tokens is not None else []
        # sort the escaped tokens in reverse order of length, so that if overlapping special tokens are given, the longest one is still preserved

        self.escaped_tokens = sorted(
            self.escaped_tokens, key=len, reverse=True)

        # this is split pattrn for speical tokens. will look like this: "\<\|endoftext\|\>|\<\|somespeicaltoken\|\>"
        # (pat?) splits and preserves the delimiter as well
        # in this case the special tokens
        self.split_pattern = f'({"|".join(self.escaped_tokens)}?)'

        # initialise vocab with 256 chars and the special tokens
        self.vocab = {i: bytes([i]) for i in range(256)}  # dict[int, bytes]
        self.len_vocab = len(self.vocab)
        for tok in self.special_tokens:
            self.vocab[self.len_vocab] = tok
            self.len_vocab += 1

        self.merges: List[Tuple[bytes, bytes]] = []

        self.file_path = file_path

    def train(self,):
        # prepare data structures for optimized merging
        words = []
        word_freqs = []
        pair_counts = defaultdict(int)
        pair_locs = defaultdict(set)

        for word_idx, (string_word, freq) in enumerate(self.master_count.items()):
            # transform string pre-tokens into lists of single utf-8 bytes
            byte_word = [bytes([b]) for b in string_word.encode("utf-8")]
            words.append(byte_word)
            word_freqs.append(freq)

            # index the initial pairs and record which words contain them
            for i in range(len(byte_word) - 1):
                p = (byte_word[i], byte_word[i+1])
                pair_counts[p] += freq
                pair_locs[p].add(word_idx)

        # continue merging until we hit the maximum vocabulary size
        while len(self.vocab) < self.vocab_size:
            if not pair_counts:
                break

            # find the most frequent pair, breaking ties by preferring the lexicographically greater pair
            best_pair = max(pair_counts.keys(),
                            key=lambda p: (pair_counts[p], p))

            # merge the bytes to create the new token
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[len(self.vocab)] = new_token

            # incrementally update counts
            # only update words that actually contain the merged pair to save compute time
            words_to_update = pair_locs[best_pair].copy()

            # remove the merged pair from our tracking dictionaries
            del pair_counts[best_pair]
            del pair_locs[best_pair]

            for word_idx in words_to_update:
                word = words[word_idx]
                freq = word_freqs[word_idx]

                # subtract all existing pairs for this specific word
                for i in range(len(word) - 1):
                    p = (word[i], word[i+1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        if word_idx in pair_locs[p]:
                            pair_locs[p].remove(word_idx)

                # construct the newly merged word
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                words[word_idx] = new_word

                # add the new pairs formed by the merge back into the tracking dictionaries
                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i+1])
                    pair_counts[p] += freq
                    pair_locs[p].add(word_idx)

        return self.vocab, self.merges

    # save in the same format that they are saving
    def save(self, vocab_filepath, merges_filepath):
        with open(vocab_filepath, 'w') as f:
            vocab = {key: self.vocab[key].decode(
                'utf-8') for key in self.vocab}
            json.dump(vocab, f)
        with open(merges_filepath, 'w') as f:
            for merge_pair in self.merges:
                f.write(
                    f"{merge_pair[0].decode('utf-8')} {merge_pair[1].decode('utf-8')}\n")


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab.copy()

        self.merges = merges
        self.special_tokens = [tok.encode(
            'utf-8') for tok in special_tokens] if special_tokens is not None else []
        self.escaped_tokens = [re.escape(tok)
                               for tok in special_tokens] if special_tokens is not None else []
        # sort the escaped tokens in reverse order of length, so that if overlapping special tokens are given, the longest one is still preserved
        self.escaped_tokens = sorted(
            self.escaped_tokens, key=len, reverse=True)
        # this is split pattrn for speical tokens. will look like this: "\<\|endoftext\|\>|\<\|somespeicaltoken\|\>"
        # (pat?) splits and preserves the delimiter as well
        # in this case the special tokens
        self.split_pattern = f'({"|".join(self.escaped_tokens)}?)'

        # add special tokens to vocab
        id = len(self.vocab)
        for tok in self.special_tokens:
            if tok not in self.vocab.values():
                self.vocab[id] = tok
                id += 1

        # inverse lookup for encoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    # load tokenizer from a saved version
    # copied from their version. uses gpt2_bytes_to_unicode
    @classmethod
    def from_files(cls, vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None
                   ):

        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token]
                                    for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        return cls(vocab, merges, special_tokens)

    def _apply_bpe(self, word_bytes: bytes) -> list[int]:
        # split pre-tokenized word into individual base bytes
        chunks = [bytes([b]) for b in word_bytes]
        # apply merges iteratively strictly in their order of creation
        for merge_pair in self.merges:
            if len(chunks) < 2:
                break

            new_chunks = []
            i = 0
            while i < len(chunks):
                # if we find a matching pair of bytes, merge them into one chunk
                if i < len(chunks) - 1 and chunks[i] == merge_pair[0] and chunks[i+1] == merge_pair[1]:
                    new_chunks.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_chunks.append(chunks[i])
                    i += 1
            chunks = new_chunks

        # map the final merged byte chunks to their integer ids
        return [self.inverse_vocab[chunk] for chunk in chunks]

    def encode(self, text: str) -> list[int]:

        if len(self.special_tokens) > 0:
            isolated_docs = re.split(self.split_pattern, text)
        else:
            isolated_docs = [text]
        token_ids = []

        for doc in isolated_docs:
            if not doc:
                continue

            # if the token is a special token itself, then add its token id directly
            if doc.encode('utf-8') in self.special_tokens:
                token_ids.append(self.inverse_vocab[doc.encode('utf-8')])

            # do merging then append token id
            else:
                for match in re.finditer(self.PAT, doc):
                    word_bytes = match.group().encode('utf-8')
                    token_ids.extend(self._apply_bpe(word_bytes))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def encode_parallel(self, iterable: Iterable[str], num_processes: int = 8) -> Iterator[int]:
        # we use a context manager to spin up a pool of workers
        with Pool(num_processes) as p:
            # pool.imap consumes the iterable lazily and guarantees the output
            # exactly matches the original input order.
            # chunksize tells workers to grab a batch of documents at once to reduce overhead.
            for ids in p.imap(self.encode, iterable, chunksize=100):
                yield from ids

    def decode(self, ids: list[int]) -> str:

        # fetch the raw bytes for each id and concatenate them
        raw_bytes = b"".join([self.vocab[idx] for idx in ids])

        # decode safely falling back to the unicode replacement character for broken sequences
        return raw_bytes.decode("utf-8", errors="replace")


def document_generator(filepath):
    # lazily yield chunks of text (e.g., line by line or split by your special tokens)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


if __name__ == "__main__":

    bpe = BPETokenizer.from_files("tests/fixtures/gpt2_vocab.json",
                                  "tests/fixtures/gpt2_merges.txt", special_tokens=["<|endoftext|>"])

    token_stream = bpe.encode_parallel(
        document_generator("cs336_basics/tokenizer/dataset.txt"), num_processes=8)

    # write the tokens directly to disk as uint16 without holding them in ram
    with open("cs336_basics/tokenizer/encoded_tokens.bin", "wb") as f:
        for token_id in token_stream:
            f.write(np.uint16(token_id).tobytes())
