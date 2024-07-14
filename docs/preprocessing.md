# Data Preprocessing Documentation

## Overview

This document provides information on the data preprocessing scripts located in the `src/data_preprocessing` directory. These scripts are designed to convert EPUB and PDF files into text format, making them suitable for further processing through a language model.

## Scripts

### 1. `convert_epub_to_txt.py`

This script converts an EPUB file to a plain text file.

#### Usage

```bash
python convert_epub_to_txt.py <input_epub_file> <output_txt_file>
```

#### Parameters

- `<input_epub_file>`: Path to the input EPUB file.
- `<output_txt_file>`: Path to the output text file.

#### Description

1. **Reading the EPUB File**: The script reads the EPUB file using the `ebooklib` library.
2. **Extracting Text**: Text is extracted from each chapter using BeautifulSoup to parse the HTML content.
3. **Writing to Text File**: The extracted text from all chapters is joined and written to the specified text file.

### 2. `convert_pdf_to_txt.py`

This script converts a PDF file to a plain text file.

#### Usage

```bash
python convert_pdf_to_txt.py <input_pdf_file> <output_txt_file>
```

#### Parameters

- `<input_pdf_file>`: Path to the input PDF file.
- `<output_txt_file>`: Path to the output text file.

#### Description

1. **Reading the PDF File**: The script opens and reads the PDF file using the `PyPDF2` library.
2. **Extracting Text**: Text is extracted from each page of the PDF.
3. **Writing to Text File**: The extracted text from all pages is joined and written to the specified text file.

## Example

### Converting an EPUB File

```bash
python convert_epub_to_txt.py books/sample.epub text/sample.txt
```

### Converting a PDF File

```bash
python convert_pdf_to_txt.py documents/sample.pdf text/sample.txt
```

## Error Handling

Both scripts include basic error handling to ensure that the input files exist before attempting conversion. If the specified input file does not exist, an error message is displayed, and the script exits.

## Further Processing

These preprocessing steps ensure that the text data is in a suitable format for analysis and model input.

## Dependencies

Ensure that the following Python libraries are installed:

- `ebooklib`
- `beautifulsoup4`
- `PyPDF2`

You can install these dependencies using `pip`:

```bash
pip install ebooklib beautifulsoup4 PyPDF2
```