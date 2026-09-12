from pathlib import Path

try:
    from .pdf_reader import read_all_pdfs
except ImportError:  # pragma: no cover
    from pdf_reader import read_all_pdfs


PDF_DIRECTORY = Path("data/documents/PDF Files")
pages = read_all_pdfs(PDF_DIRECTORY)

print(f"\nDirectory: {PDF_DIRECTORY}")
print(f"PDF pages with text: {len(pages)}")

for page in pages[:2]:
    print("\n" + "=" * 60)
    print(f"Page: {page.page_number}")
    print(f"Source: {page.source}")
    print("=" * 60)
    print(page.text[:1000])
