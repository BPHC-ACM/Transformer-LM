import sys
from bpe_tokenizer import BPETokenizer

def main():
    # 1. Load the pre-trained tokenizer
    try:
        tokenizer = BPETokenizer.load("tokenizer.json")
        print("Successfully loaded 'tokenizer.json'")
    except FileNotFoundError:
        print("Error: 'tokenizer.json' not found. Run bpe_tokenizer.py first.")
        sys.exit(1)

    print("\nBPE Tokenizer Inference")

    # 2. Open a text file to append the outputs
    output_filename = "inference_output.txt"
    with open(output_filename, "a", encoding="utf-8") as file:
        
        # 3. Interactive loop
        while True:
            text = input("Enter text: ")
            
            if text.lower() == "exit":
                break
            if not text.strip():
                continue

            # 4. Encode text to IDs
            ids = tokenizer.encode(text)
            
            # 5. Extract the string representation for each token ID
            tokens = [tokenizer.vocab[tid].decode('utf-8', errors='replace') for tid in ids]

            # 6. Format the results
            output_lines = [
                "-" * 50,
                f"Original Text : {text}",
                f"Token IDs     : {ids}",
                f"Tokens Array  : {tokens}",
                "Mapping Breakdown:"
            ]
            
            for token_str, token_id in zip(tokens, ids):
                output_lines.append(f"  * {repr(token_str):<15} -> {token_id}")
            
            final_output = "\n".join(output_lines) + "\n"

            # 7. Print to terminal and write to file
            print(final_output)
            file.write(final_output)

    print(f"\nSession ended. All results have been appended to '{output_filename}'.")

if __name__ == "__main__":
    main()