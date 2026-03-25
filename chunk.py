import re
from load import load_pdfs

def clean_text(text: str) -> str:
    #for removing extra blank lines 
    text = re.sub(r'\n{3,}', '\n\n', text)
    #for removing non-english characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    #replacing multiple spaces and tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    #remove whitespace
    text = text.strip()
    return text


def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunk_list = []
    chunk_id = 0

    for page in pages:
        cleaned = clean_text(page["text"])
        words = cleaned.split()

        if len(words) < 20:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text_str = " ".join(words[start:end])

            chunk_list.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text_str,
                "source": page["source"],
                "page": page["page"]
            })

            chunk_id += 1
            start += chunk_size - overlap

    print(f"Total chunks created: {len(chunk_list)}")
    return chunk_list


if __name__ == "__main__":
    pages = load_pdfs("./data")
    chunks = chunk_text(pages)

    print("\n--- PREVIEW: First 2 chunks ---")
    for chunk in chunks[:2]:
        print(f"\nID     : {chunk['id']}")
        print(f"Source : {chunk['source']}")
        print(f"Page   : {chunk['page']}")
        print(f"Text   : {chunk['text'][:200]}")
        print("-" * 50)