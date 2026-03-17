from .pretokenizer import ChunkPreTokenizer
from .bpe_tokenizer import BPETrainer
from .config import TINYSTORIES_PATH, VOCAB_SIZE, SPECIAL_TOKENS, NUM_PROCESSORS


def train_tokenizer(input_path, vocab_size, special_tokens, **kwargs):
    pretokenizer = ChunkPreTokenizer(
        input_path, special_tokens=special_tokens, num_processes=NUM_PROCESSORS)
    pretokenizer.create_pretokens()
    trainer = BPETrainer(pretokenizer.master_count, vocab_size=vocab_size)
    vocab, merges = trainer.train()

    del pretokenizer
    del trainer

    return vocab, merges


if __name__ == "__main__":
    train_tokenizer(TINYSTORIES_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
