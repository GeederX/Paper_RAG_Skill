"""
PDF to Text Batch Converter
Converts all PDF files in the papers directory to text files in the texts directory.
Uses pdfplumber library for reliable text extraction.
"""

import pdfplumber
from pathlib import Path
from tqdm import tqdm


def convert_pdf_to_text(pdf_path, txt_path):
    """
    Convert a single PDF file to text using pdfplumber.

    Args:
        pdf_path: Path to the input PDF file
        txt_path: Path to the output text file

    Returns:
        True if successful, False otherwise
    """
    try:
        text_content = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()

                if page_text:
                    text_content.append(f"--- Page {page_num}/{total_pages} ---\n")
                    text_content.append(page_text)
                    text_content.append("\n\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("".join(text_content))

        return True

    except Exception as e:
        print(f"[ERROR] Failed to convert {pdf_path.name}: {str(e)}")
        return False


def batch_convert_pdfs(papers_dir, texts_dir, skip_existing=True):
    """
    Batch convert all PDF files in papers_dir to text files in texts_dir.

    Args:
        papers_dir: Directory containing PDF files
        texts_dir: Directory to save text files
        skip_existing: If True, skip files that already exist in texts_dir
    """
    texts_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(papers_dir.glob("*.pdf"))
    total_files = len(pdf_files)

    if total_files == 0:
        print(f"[WARNING] No PDF files found in {papers_dir}")
        return

    print(f"Found {total_files} PDF files")

    files_to_convert = []
    files_to_skip = []

    for pdf_file in pdf_files:
        txt_filename = pdf_file.stem + ".txt"
        txt_path = texts_dir / txt_filename

        if skip_existing and txt_path.exists():
            files_to_skip.append(pdf_file.name)
        else:
            files_to_convert.append((pdf_file, txt_path))

    skipped = len(files_to_skip)
    to_convert = len(files_to_convert)

    print("\n" + "=" * 60)
    print("FILE SCAN SUMMARY")
    print("=" * 60)
    print(f"Total PDF files:          {total_files}")
    print(f"Already converted (skip): {skipped}")
    print(f"Need to convert:          {to_convert}")
    print("=" * 60 + "\n")

    if to_convert == 0:
        print("All files already converted. Nothing to do.")
        return

    successful = 0
    failed = 0

    for pdf_file, txt_path in tqdm(
        files_to_convert, desc="Converting PDFs", unit="file"
    ):
        if convert_pdf_to_text(pdf_file, txt_path):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total PDF files found:    {total_files}")
    print(f"Successfully converted:   {successful}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed:                   {failed}")
    print("=" * 60)


def main():
    """Main function to run the batch conversion."""
    script_dir = Path(__file__).parent

    papers_dir = script_dir / "papers"
    texts_dir = script_dir / "texts"

    print("=" * 60)
    print("PDF to Text Batch Converter")
    print("=" * 60)
    print(f"Input directory:  {papers_dir}")
    print(f"Output directory: {texts_dir}")
    print("=" * 60 + "\n")

    if not papers_dir.exists():
        print(f"[ERROR] Papers directory not found: {papers_dir}")
        return

    batch_convert_pdfs(papers_dir, texts_dir, skip_existing=True)


if __name__ == "__main__":
    main()
