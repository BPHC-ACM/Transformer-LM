from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Iterator
from .pretokenizer import ChunkPreTokenizer
from collections import Counter


class BPETrainer(ChunkPreTokenizer):
    def __init__(self, master_count:Counter,vocab_size=10000):
        self.master_count = master_count
        self.vocab_size = vocab_size

        # initialise vocab with 256 chars and the special tokens
        self.vocab = {i: bytes([i]) for i in range(256)}  # dict[int, bytes]
        self.len_vocab = len(self.vocab)
        for tok in self.special_tokens:
            self.vocab[self.len_vocab] = tok
            self.len_vocab += 1

        self.merges: List[Tuple[bytes, bytes]] = []  

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
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

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




class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        pass
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    pass
def encode(self, text: str) -> list[int]:
    pass 
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    pass
def decode(self, ids: list[int]) -> str :
    pass


    
