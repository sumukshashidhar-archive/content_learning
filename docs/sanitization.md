# Documentation for Text Sanitization and Chunking Script

## Overview

This doc serves as a guide to using the script designed for sanitizing, formatting, and chunking textual data, followed by combining these chunks with structured JSON data. The script processes large volumes of text and structured data, making them suitable for applications in natural language processing, such as training language models or data analysis.

The primary functionality of the script includes:
1. **Sanitization and Formatting**: Cleanses text data by removing non-ASCII characters and redundant spaces, and formats the content into manageable lines.
2. **Chunking**: Splits the sanitized text into chunks that do not exceed a specified token limit, facilitating better handling in tokenization processes typically used in language model training.
3. **Data Merging**: Intersperses text chunks with entries from JSON files to create a heterogeneous dataset.
4. **Output**: Saves the processed and combined data into a JSON file for subsequent use.

## File Description: `sanitization.md`

### System Requirements

- Python 3.7 or higher
- Libraries: `transformers`, `re`, `unicodedata`, `json`, `os`, `random`, `logging`, `argparse`
- Compatible with UNIX and Windows operating systems

### Installation

Before running the script, ensure that all dependent Python libraries are installed. You can install them using the following command:

```bash
pip install transformers
```

### Components of the Script

#### 1. Sanitization and Formatting (`sanitize_and_format_text`)

**Purpose**:
- Normalize Unicode characters to ASCII.
- Remove non-printable characters except for whitespace characters.
- Condense all forms of whitespace into single spaces.
- Split text into sentences and reformat these to ensure each line meets a minimum word count, enhancing readability and consistency across datasets.

**Parameters**:
- `text`: A string containing the input text to be sanitized and formatted.
- `min_words_per_line`: An integer defining the minimum number of words per line, defaulting to 20.

#### 2. JSON File Reader (`read_json_files`)

**Purpose**:
- Read and aggregate data from multiple JSON files within a specified directory.

**Parameters**:
- `directory`: Path to the directory containing JSON files.

#### 3. Text Chunking (`split_into_chunks`)

**Purpose**:
- Divide the sanitized text into smaller segments ("chunks") that comply with a specified maximum token count, facilitating their usability in machine learning models without exceeding tokenization limits.

**Parameters**:
- `text`: Pre-sanitized text to be chunked.
- `tokenizer`: An instance of `AutoTokenizer` from the Hugging Face `transformers` library.
- `max_tokens`: The maximum number of tokens allowed in each chunk.

#### 4. Data Merging and Shuffling (`intersperse_chunks`)

**Purpose**:
- Combine text chunks and JSON data entries in a shuffled order to ensure diversity and randomness in the dataset.

**Parameters**:
- `chunks`: A list of text chunks.
- `json_data`: A list of JSON data entries.

#### 5. Output Writer (`save_chunks_to_json`)

**Purpose**:
- Save the combined and processed data to a JSON file.

**Parameters**:
- `chunks`: Combined data chunks.
- `output_file`: Destination file path for the output JSON.

### Usage

To utilize the script, you need to specify directories containing text and JSON files along with the output file name through the command line. An example usage is as follows:

```bash
python script_name.py /path/to/text/files /path/to/json/files output.json
```

Additional command-line arguments include:
- `-s, --size`: Maximum size of text chunks in tokens.
- `-w, --words`: Minimum words per formatted line.
- `--seed`: Seed for random number generation to ensure reproducibility.