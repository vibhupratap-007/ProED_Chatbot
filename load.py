import fitz
from pathlib import Path

def load_pdfs(folder_path: str) -> list[dict]:
    documents = []
    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in folder:", folder_path)
        return []

    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        print(f"Reading file: {pdf_path.name} ({len(doc)} pages)")

        for i in range(len(doc)):
            text = doc[i].get_text("text")
            if not text.strip():
                continue
            documents.append({
                "text": text,
                "source": pdf_path.name,
                "page": i + 1,
                "word_count": len(text.split())
            })
        doc.close()

    print(f"Total pages extracted are: {len(documents)}")
    return documents


if __name__ == "__main__":
    pages = load_pdfs("./data")
    print("\n--- PREVIEW ---")
    print(f"Source : {pages[0]['source']}")
    print(f"Page   : {pages[0]['page']}")
    print(f"Text   : {pages[0]['text'][:300]}")