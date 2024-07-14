import PyPDF2
import sys
import os


def pdf_to_txt(pdf_path, txt_path):
    """Convert a PDF file to a text file.

    Args:
        pdf_path (str): The path to the input PDF file.
        txt_path (str): The path to the output text file.
    """
    # Open the PDF file
    with open(pdf_path, "rb") as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract text from each page
        pages_text = []
        for page in pdf_reader.pages:
            pages_text.append(page.extract_text())

        # Join all pages into a single string
        full_text = "\n\n".join(pages_text)

        # Write to txt file
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(full_text)


def main():
    """Main function to handle argument parsing and call the conversion function."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_pdf_file> <output_txt_file>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    txt_file_path = sys.argv[2]

    if not os.path.exists(pdf_file_path):
        print(f"Error: The file {pdf_file_path} does not exist.")
        sys.exit(1)

    pdf_to_txt(pdf_file_path, txt_file_path)
    print(f"Converted {pdf_file_path} to {txt_file_path}")


if __name__ == "__main__":
    main()
