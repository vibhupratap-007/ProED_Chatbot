# rag.py

import fitz
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ─────────────────────────────────────────
# STEP 1: EXTRACT TEXT FROM PDFs
# ─────────────────────────────────────────

def load_pdfs(folder_path: str) -> list[dict]:
    all_pages = []
    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDFs found in folder:", folder_path)
        return []

    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        print(f"📄 Loading: {pdf_path.name} ({len(doc)} pages)")

        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            if not text.strip():
                continue
            all_pages.append({
                "text": text,
                "source": pdf_path.name,
                "page": page_num + 1
            })
        doc.close()

    print(f"\n✅ Done! Total pages extracted: {len(all_pages)}")
    return all_pages


# ─────────────────────────────────────────
# STEP 2: CLEAN + CHUNK THE TEXT
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text


def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    all_chunks = []
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

            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text_str,
                "source": page["source"],
                "page": page["page"]
            })

            chunk_id += 1
            start += chunk_size - overlap

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────
# STEP 3: EMBED + STORE IN PINECONE
# ─────────────────────────────────────────

def store_in_pinecone(chunks: list[dict]):
    # Load embedding model (free, runs locally)
    print("\n⏳ Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded!")

    # Connect to Pinecone
    print("\n⏳ Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "rag-index"

    # Create index if it doesn't exist
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"⏳ Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,        # all-MiniLM-L6-v2 produces 384-dim vectors
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Index '{index_name}' created!")
    else:
        print(f"✅ Index '{index_name}' already exists, skipping creation.")

    index = pc.Index(index_name)

    # Embed and upload in batches of 50
    print(f"\n⏳ Embedding and uploading {len(chunks)} chunks to Pinecone...")
    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Generate embeddings for this batch
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        # Prepare vectors for Pinecone
        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            })

        # Upload batch
        index.upsert(vectors=vectors)
        print(f"  ✅ Uploaded chunks {i} to {i + len(batch)}")

    print(f"\n🎉 All {len(chunks)} chunks stored in Pinecone!")


# ─────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Step 1
    pages = load_pdfs("./data")

    # Step 2
    chunks = chunk_text(pages)

    # Step 3
    store_in_pinecone(chunks)