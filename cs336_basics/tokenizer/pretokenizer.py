import os
from typing import BinaryIO
from multiprocessing import Pool, Process
from collections import Counter
import regex as re


class ChunkPreTokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, file_path: str,  special_tokens: list[str] = [], num_processes=8):
        self.file_path = file_path

        # if we have user input of speical tokens then update

        self.special_tokens = [tok.encode('utf-8') for tok in special_tokens]
        self.escaped_tokens = [re.escape(tok)
                               for tok in special_tokens]
        self.endoftext = b"<|endoftext|>"

        # this is split pattrn for speical tokens. will look like this: "\<\|endoftext\|\>|\<\|somespeicaltoken\|\>"
        self.split_pattern = "|".join(self.escaped_tokens)

        # master count -> this will be passes to the tokenizer and will holdl final count map of each pretoken
        self.master_count = Counter()

        self.num_processes = num_processes

    def find_chunk_boundaries(self, file, desired_num_chunks):
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """

        # updated the assert code they gave to work for multiple special tokens
        assert all([isinstance(special_token,
                               bytes)
                    for special_token in self.special_tokens]), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [
            i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(self.endoftext)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def process_chunk(self, args):
        """
        helper function to process each chunk parallely.
        splits the chunk of the special tokens, and for each isolated doc increments the token counter
        this local token counter is returned, so that the global counter can be updated
        """
        file_path, start, end = args
        chunk = ""
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end-start).decode("utf-8",
                                             errors="ignore")

        isolated_docs = re.split(self.split_pattern, chunk)

        local_couter = Counter()
        for doc in isolated_docs:
            local_couter.update(Counter(re.findall(self.PAT, doc)))

        return local_couter

    def create_pretokens(self):
        """
        ties everything together, and creates the master count of pretokens accross the entire corpus
        """
        boundaries = []
        with open(self.file_path, "rb") as f:
            # split the corpus into num_processor chunks
            # (omitting the special end of text token)
            # for the parallel processing in the next step
            boundaries = self.find_chunk_boundaries(f,  self.num_processes)
            boundaries = zip([self.file_path]*len(boundaries),
                             boundaries[:-1], boundaries[1:])

        with Pool(4) as p:
            for result in p.imap_unordered(self.process_chunk, (boundaries)):
                self.master_count.update(result)

        return self.master_count


if __name__ == "__main__":
    from cs336_basics.tokenizer.config import NUM_PROCESSORS, TINYSTORIES_PATH, PAT

    pretokenizer = ChunkPreTokenizer(TINYSTORIES_PATH, ["<|endoftext|>"], 6)
    print(len(pretokenizer.create_pretokens()))
