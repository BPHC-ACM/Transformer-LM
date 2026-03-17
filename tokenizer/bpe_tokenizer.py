# Byte pair Encoding Tokenizer (BPE Tokenizer)
import regex as re
from collections import defaultdict
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Tokenizer --> Converts a string of text into a sequence of integers (ID) that a natural language model actually sees, interprets and operates on.

# BPE
# Iteratively merge the most frequest adjacent pair of symbols until a good vocab size is built to work with

# Architecture

"""
┌──────────────────────────────────────────────────────┐
│                   BPETokenizer                       │
│                                                      │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │Pre-tokenizer|    │     Trainer  │                 │
│  │             │    │              │                 │
│  │ text→words  │    │ corpus→merges│                 │
│  └─────────────┘    └──────────────┘                 │
│                                                      │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │  Vocabulary │    │  Merge Rules │                 │
│  │  {token:id} │    │  [(a,b)→ab]  │                 │
│  └─────────────┘    └──────────────┘                 │
│                                                      │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │   Encoder   │    │   Decoder    │                 │
│  │ text→[ids]  │    │  [ids]→text  │                 │
│  └─────────────┘    └──────────────┘                 │
│                                                      │
│  ┌──────────────────────────────┐                    │
│  │     Save / Load (JSON)       │                    │
│  └──────────────────────────────┘                    │
└──────────────────────────────────────────────────────┘
"""


"""
BPETokenizer
│
├── train(corpus, vocab_size)
│   ├── pre_tokenize()        → word frequency dict
│   ├── build_base_vocab()    → all unique characters
│   └── merge_loop()
│       ├── count_pairs()     → frequency of all adjacent pairs
│       ├── get_best_pair()   → most frequent pair
│       ├── merge_pair()      → apply merge to all words
│       └── record_merge()    → save rule to merge list
│
├── encode(text) → [int, ...]
│   ├── pre_tokenize()
│   ├── apply_merges()        → apply learned rules in order
│   └── map to token IDs
│
├── decode([int, ...]) → str
│   └── map IDs → tokens → join → strip </w>
│
└── save() / load()           → JSON
"""

"""
Step 1 → pre_tokenize()
Step 2 → build_base_vocab()
Step 3 → count_pairs()
Step 4 → merge_pair()
Step 5 → train()  ← orchestrates 1-4
Step 6 → encode()
Step 7 → decode()
Step 8 → save() / load()
"""



# Step-1 Pretokenizer

def pre_tokenize(text: str) -> dict:
    """
    Converts the raw text into a word-freqency dictionary, with each word split into characters + </w> 
    Ex: "low lower low" → {"l o w </w>": 2, "l o w e r </w>": 1}
    """

    words = re.findall(r"[a-zA-Z]+" , text.lower())

    word_freqs = defaultdict(int)
    for word in words: 
        tokenized = " ".join(list(word)) + " </w>"    # </w> 0 word boudary
        word_freqs[tokenized] += 1

    return dict(word_freqs)


# Step-2 Base Vocab
#    Scan the word freqency dict and collect every unique symbol as the initial vocabulary

def build_base_vocab(word_freqs : dict) -> dict:
    """
    Extracts all unique characters (+ </w>) from word_freqs.
    Returns vocab: {token: id}
    """
    vocab = set()

    # Adding each unique char to vocab
    for word in word_freqs:
        symbols = word.split()
        for symbol in symbols:
            vocab.add(symbol)

    # Sorting for determinism, to assign IDs
    vocab = sorted(vocab)
    return {token: idx for idx,token in enumerate(vocab)}


# Step-3 Count Pairs
# Scans all words in the corpus and count how often every adjacet pair appears, weighted by word frequency

def count_pairs(word_freqs: dict) -> dict:
    """
    word_freqs: {(b'l', b'o', b'w'): 2, ...}
    returns:    {(b'l', b'o'): 5, (b'o', b'w'): 5, ...}
    """

    pair_counts = defaultdict(int)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq  # Word frequency weight
    
    return dict(pair_counts)


# Step 4 - get best pair + merge pair
# Find the most frequent pair. Tie-break by lexicographically greater pair.

def get_best_pair(pair_counts: dict) -> tuple:
    """
    Returns the most frequent pair.
    Ties broken by lexicographically greater pair.
    """

    return max(pair_counts, key=lambda p: (pair_counts[p], p))


def merge_pair(word_freqs: dict, pair: tuple) -> dict:
    """
    Merges all occurrences of `pair` in every word.
    e.g. pair = (b'l', b'o')
    (b'l', b'o', b'w') → (b'lo', b'w')
    """
    new_word_freqs = {}
    tok_a, tok_b = pair
    merged = tok_a + tok_b          # b'l' + b'o' → b'lo'

    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            # If current and next token match the pair, merge them
            if i < len(word) - 1 and word[i] == tok_a and word[i+1] == tok_b:
                new_word.append(merged)
                i += 2      # skip both tokens
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs



# Step 5 - Training step
# Running merge_pair until the vocabulary reaches a certain size

def train(text: str, vocab_size: int, special_tokens: list[str] = None):
    """
    text       : raw training corpus
    vocab_size : target vocabulary size
    special_tokens : e.g. ['<|endoftext|>']
    
    returns: (vocab, merges)
    """
    if special_tokens is None:
        special_tokens = []

    # Step 1: Handle special tokens — remove from text before pre-tokenizing
    # so they never get merged
    pattern = '|'.join(re.escape(tok) for tok in special_tokens)
    chunks = re.split(pattern, text) if pattern else [text]

    # Step 2: Pre-tokenize all chunks
    word_freqs = defaultdict(int)
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            token_bytes = tuple(bytes([b]) for b in match.group().encode('utf-8'))
            word_freqs[token_bytes] += 1
    word_freqs = dict(word_freqs)

    # Step 3: Build base vocab — 256 bytes + special tokens
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode('utf-8')

    # Step 4: Merge loop
    merges = []
    num_merges = vocab_size - len(vocab)   # how many merges we need

    for _ in range(num_merges):
        pair_counts = count_pairs(word_freqs)
        
        if not pair_counts:
            break                          # nothing left to merge
        
        best = get_best_pair(pair_counts)
        word_freqs = merge_pair(word_freqs, best)
        
        # Add new token to vocab
        new_token = best[0] + best[1]
        vocab[len(vocab)] = new_token
        merges.append(best)

    return vocab, merges


# Step 6 - Encode
# inference step - takes new text and applies merges and returns tokenIDs

def encode(text: str, vocab: dict, merges: list, special_tokens: list[str] = None) -> list[int]:
    """
    text   : string to encode
    vocab  : {id: bytes}
    merges : [(b's', b't'), (b'e', b'st'), ...]
    
    returns: [int, int, ...]
    """
    if special_tokens is None:
        special_tokens = []

    # Reverse vocab for lookup: bytes → id
    bytes_to_id = {v: k for k, v in vocab.items()}

    # Step 1: Handle special tokens first — they get their own IDs, never split
    if special_tokens:
        pattern = '|'.join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(f'({pattern})', text)   # keep the delimiters
    else:
        chunks = [text]

    ids = []

    for chunk in chunks:
        # If chunk is a special token, directly map to ID
        if chunk in special_tokens:
            ids.append(bytes_to_id[chunk.encode('utf-8')])
            continue

        # Step 2: Pre-tokenize chunk
        for match in re.finditer(PAT, chunk):
            # Represent word as tuple of individual bytes
            word = tuple(bytes([b]) for b in match.group().encode('utf-8'))

            # Step 3: Apply merges in order
            for tok_a, tok_b in merges:
                merged = tok_a + tok_b
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == tok_a and word[i+1] == tok_b:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = tuple(new_word)

            # Step 4: Map each token to its ID
            for token in word:
                ids.append(bytes_to_id[token])

    return ids


# Step 7 - Decode
# Just decoding the ids to the tokens

def decode(ids: list[int], vocab: dict) -> str:
    """
    ids   : [108, 111, 119, ...]
    vocab : {id: bytes}
    
    returns: decoded string
    """
    # Step 1: Concatenate all byte sequences
    raw_bytes = b''.join(vocab[id] for id in ids)
    
    # Step 2: Decode bytes → string, replace bad bytes with U+FFFD
    return raw_bytes.decode('utf-8', errors='replace')


# ──────────────────────────────────────────────
# Step 8 - BPETokenizer CLASS
# Wraps everything into a single clean interface
# ──────────────────────────────────────────────

class BPETokenizer:

    def __init__(self, vocab: dict = None, merges: list = None, special_tokens: list[str] = None):
        self.special_tokens = special_tokens or []
        self.vocab          = vocab or {}       # {id: bytes}
        self.merges         = merges or []      # [(bytes, bytes), ...]
        self.bytes_to_id    = {v: k for k, v in self.vocab.items()}


    def train(self, text: str, vocab_size: int):
        # Strip special tokens before pre-tokenizing
        pattern = '|'.join(re.escape(tok) for tok in self.special_tokens)
        chunks  = re.split(pattern, text) if pattern else [text]

        # Pre-tokenize all chunks
        word_freqs = defaultdict(int)
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                token_bytes = tuple(bytes([b]) for b in match.group().encode('utf-8'))
                word_freqs[token_bytes] += 1
        word_freqs = dict(word_freqs)

        # Base vocab: 256 bytes + special tokens
        self.vocab = {}
        for i in range(256):
            self.vocab[i] = bytes([i])
        for i, tok in enumerate(self.special_tokens):
            self.vocab[256 + i] = tok.encode('utf-8')

        # Merge loop
        self.merges  = []
        num_merges   = vocab_size - len(self.vocab)

        for _ in range(num_merges):
            pair_counts = count_pairs(word_freqs)
            if not pair_counts:
                break

            best       = get_best_pair(pair_counts)
            word_freqs = merge_pair(word_freqs, best)

            new_token = best[0] + best[1]
            self.vocab[len(self.vocab)] = new_token
            self.merges.append(best)

        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        print(f"Training complete. Vocab size: {len(self.vocab)}, Merges: {len(self.merges)}")


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = '|'.join(re.escape(tok) for tok in self.special_tokens)
            chunks  = re.split(f'({pattern})', text)
        else:
            chunks = [text]

        ids = []
        for chunk in chunks:
            # Special token → direct ID lookup
            if chunk in self.special_tokens:
                ids.append(self.bytes_to_id[chunk.encode('utf-8')])
                continue

            # Pre-tokenize then apply merges
            for match in re.finditer(PAT, chunk):
                word = tuple(bytes([b]) for b in match.group().encode('utf-8'))

                for tok_a, tok_b in self.merges:
                    merged   = tok_a + tok_b
                    new_word = []
                    i        = 0
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == tok_a and word[i+1] == tok_b:
                            new_word.append(merged)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = tuple(new_word)

                for token in word:
                    ids.append(self.bytes_to_id[token])

        return ids


    def decode(self, ids: list[int]) -> str:
        raw_bytes = b''.join(self.vocab[i] for i in ids)
        return raw_bytes.decode('utf-8', errors='replace')


    def save(self, path: str):
        data = {
            "special_tokens": self.special_tokens,
            "vocab":          {str(k): list(v) for k, v in self.vocab.items()},
            "merges":         [[list(a), list(b)] for a, b in self.merges],
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved → {path}")


    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        with open(path, 'r') as f:
            data = json.load(f)

        vocab          = {int(k): bytes(v) for k, v in data['vocab'].items()}
        merges         = [(bytes(a), bytes(b)) for a, b in data['merges']]
        special_tokens = data['special_tokens']

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


# ──────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────

if __name__ == "__main__":

    from datasets import load_dataset
    
    print("Fetching dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]")  
    corpus = "\n".join(dataset['text'])
    print(f"Dataset loaded successfully. Total characters: {len(corpus)}")

    # ── Train ──
    tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])
    # 2048 is a respectable educational vocabulary size that will execute in a reasonable time
    tokenizer.train(corpus, vocab_size=2048)

    print("\nTop 10 merges:")
    for a, b in tokenizer.merges[:10]:
        print(f"  {a.decode(errors='replace')} + {b.decode(errors='replace')} → {(a+b).decode(errors='replace')}")

    # ── Encode / Decode round-trip ──
    text    = "Once upon a time, there was a little girl.<|endoftext|>"
    ids     = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"\nOriginal : {text}")
    print(f"IDs      : {ids}")
    print(f"Decoded  : {decoded}")
    print(f"Match    : {text == decoded}")

    # ── Save / Load ──
    tokenizer.save("tokenizer.json")
    loaded = BPETokenizer.load("tokenizer.json")

    print(f"\nLoaded IDs    : {loaded.encode(text)}")
    print(f"Round-trip OK : {loaded.encode(text) == ids}")