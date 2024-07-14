import argparse
import json
import logging
import os
import random
import re
import unicodedata

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sanitize_and_format_text(text, min_words_per_line=20):
    logging.info(f"Sanitizing and formatting text of length {len(text)}")
    original_word_count = len(text.split())

    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    text = "".join(char for char in text if char.isprintable() or char in ["\n", "\r"])
    text = re.sub(r"\s+", " ", text)

    after_cleaning_word_count = len(text.split())
    if original_word_count != after_cleaning_word_count:
        logging.warning(
            f"Word count changed during cleaning: {original_word_count} -> {after_cleaning_word_count}"
        )

    sentences = re.split(r"(?<=[.!?])\s+", text)

    formatted_text = ""
    current_line = ""
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word_count >= min_words_per_line and word.endswith((".", "!", "?")):
                current_line += word
                formatted_text += current_line.strip() + "\n"
                current_line = ""
                word_count = 0
            else:
                current_line += word + " "
                word_count += 1

        if word_count >= min_words_per_line * 2:
            formatted_text += current_line.strip() + "\n"
            current_line = ""
            word_count = 0

    if current_line:
        formatted_text += current_line.strip()

    final_word_count = len(formatted_text.split())
    logging.info(
        f"Sanitized and formatted text. New length: {len(formatted_text)}, Words: {final_word_count}"
    )
    if final_word_count != original_word_count:
        logging.warning(
            f"Word count changed during formatting: {original_word_count} -> {final_word_count}"
        )

    return formatted_text.strip()


def read_json_files(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            logging.info(f"Processing JSON file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    logging.info(f"Read {len(data)} items from {filename}")
                    json_data.extend(data)
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    logging.info(f"Total items read from JSON files: {len(json_data)}")
    return json_data


def split_into_chunks(text, tokenizer, max_tokens):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)

        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(tokenizer.bos_token + chunk_text + tokenizer.eos_token)
                current_chunk = []
                current_length = 0

            if sentence_length > max_tokens:
                chunks.append(tokenizer.bos_token + sentence + tokenizer.eos_token)
            else:
                current_chunk.append(sentence)
                current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(tokenizer.bos_token + chunk_text + tokenizer.eos_token)

    return chunks


def read_and_chunk_text_files(directory, tokenizer, max_tokens):
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            logging.info(f"Processing text file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    word_count = len(text.split())
                    logging.info(
                        f"Read {len(text)} characters, {word_count} words from {filename}"
                    )
                    sanitized_text = sanitize_and_format_text(text)
                    sanitized_word_count = len(sanitized_text.split())
                    logging.info(
                        f"After sanitization: {len(sanitized_text)} characters, {sanitized_word_count} words"
                    )
                    file_chunks = split_into_chunks(
                        sanitized_text, tokenizer, max_tokens
                    )
                    logging.info(f"Created {len(file_chunks)} chunks from {filename}")
                    chunks.extend(file_chunks)
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    logging.info(f"Total chunks created from text files: {len(chunks)}")
    return chunks


def intersperse_chunks(chunks, json_data):
    chunks *= 5  # Replicate the list of chunks five times
    random.shuffle(json_data)
    combined = chunks + json_data
    random.shuffle(combined)
    return combined


def save_chunks_to_json(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=True, indent=2)
    logging.info(f"Saved {len(chunks)} items to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine and intersperse text files and JSON data using Llama 3 tokenization"
    )
    parser.add_argument("text_directory", help="Directory containing the text files")
    parser.add_argument("json_directory", help="Directory containing the JSON files")
    parser.add_argument("output", help="Output JSON file name")
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1000,
        help="Maximum chunk size in tokens (default: 1000)",
    )
    parser.add_argument(
        "-w",
        "--words",
        type=int,
        default=10,
        help="Minimum words per line (default: 10)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for shuffling (optional)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        logging.info(f"Set random seed to {args.seed}")

    logging.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    logging.info(f"Processing text files from {args.text_directory}")
    text_chunks = read_and_chunk_text_files(args.text_directory, tokenizer, args.size)

    logging.info(f"Processing JSON files from {args.json_directory}")
    json_data = read_json_files(args.json_directory)

    logging.info("Interspersing chunks and JSON data")
    interspersed_chunks = intersperse_chunks(text_chunks, json_data)

    logging.info(f"Saving output to {args.output}")
    save_chunks_to_json(interspersed_chunks, args.output)

    logging.info(
        f"Processing complete. {len(interspersed_chunks)} items saved to {args.output}"
    )


if __name__ == "__main__":
    main()
