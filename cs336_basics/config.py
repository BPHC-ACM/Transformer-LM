TINYSTORIES_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
OWT_PATH = "data/owt_valid.txt"
NUM_PROCESSORS = 8
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]