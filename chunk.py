import re
from load import load_pdfs


def clean_text(text: str) -> str:
    # for removing extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # for removing non-english characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # replacing multiple spaces and tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # remove whitespace
    text = text.strip()
    return text


def split_into_sections(text: str) -> list[str]:
    # split text at natural boundaries - double newlines mean new paragraph
    # this is the core of structure-based chunking
    sections = re.split(r'\n\n+', text)

    # remove sections that are too small to be useful
    sections = [s.strip() for s in sections if len(s.strip()) > 50]
    return sections


def chunk_text(pages: list[dict], max_words: int = 500, overlap_words: int = 50) -> list[dict]:
    chunk_list = []
    chunk_id = 0

    for page in pages:
        cleaned = clean_text(page["text"])

        # split page into natural sections first
        sections = split_into_sections(cleaned)

        current_chunk = ""
        current_word_count = 0

        for section in sections:
            section_words = section.split()
            section_word_count = len(section_words)

            # if adding this section exceeds max size → save current chunk first
            if current_word_count + section_word_count > max_words and current_chunk:
                chunk_list.append({
                    "id": f"chunk_{chunk_id}",
                    "text": current_chunk.strip(),
                    "source": page["source"],
                    "page": page["page"]
                })
                chunk_id += 1

                # keep last few words as overlap for context continuity
                overlap_text = " ".join(current_chunk.split()[-overlap_words:])
                current_chunk = overlap_text + " " + section
                current_word_count = len(current_chunk.split())

            else:
                # section fits — add it to current chunk
                current_chunk += "\n\n" + section
                current_word_count += section_word_count

        # save whatever is left at the end of the page
        if current_chunk.strip():
            chunk_list.append({
                "id": f"chunk_{chunk_id}",
                "text": current_chunk.strip(),
                "source": page["source"],
                "page": page["page"]
            })
            chunk_id += 1

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