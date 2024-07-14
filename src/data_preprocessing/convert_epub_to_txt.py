import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import sys
import os


def epub_to_txt(epub_path, txt_path):
    """Convert an EPUB file to a text file.

    Args:
        epub_path (str): The path to the input EPUB file.
        txt_path (str): The path to the output text file.
    """
    # Read the EPUB file
    book = epub.read_epub(epub_path)

    # Extract text from each chapter
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), "html.parser")
            # Extract text
            chapters.append(soup.get_text())

    # Join all chapters into a single string
    full_text = "\n\n".join(chapters)

    # Write to txt file
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(full_text)


def main():
    """Main function to handle argument parsing and call the conversion function."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_epub_file> <output_txt_file>")
        sys.exit(1)

    epub_file_path = sys.argv[1]
    txt_file_path = sys.argv[2]

    if not os.path.exists(epub_file_path):
        print(f"Error: The file {epub_file_path} does not exist.")
        sys.exit(1)

    epub_to_txt(epub_file_path, txt_file_path)
    print(f"Converted {epub_file_path} to {txt_file_path}")


if __name__ == "__main__":
    main()
