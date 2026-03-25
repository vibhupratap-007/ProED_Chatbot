import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from load import load_pdfs
from chunk import chunk_text

load_dotenv()

def store_in_pinecone(chunks: list[dict]):
    # Load embedding model
    print("⏳ Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded!")

    # Connect to Pinecone
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "rag-index"

    # Create index if it doesn't exist
    current_index = [i.name for i in pc.list_indexes()]
    if index_name not in current_index:
        print(f"⏳ Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created!")
    else:
        print(f"Index '{index_name}' already exists!")

    index = pc.Index(index_name)

    # Embed + upload in batches
    print(f"\nEmbedding and uploading {len(chunks)} chunks...")
    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

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

        index.upsert(vectors=vectors)
        print(f"Uploaded chunks {i} to {i + len(batch)}")

    print(f"\nAll {len(chunks)} chunks stored in Pinecone!")


if __name__ == "__main__":
    pages = load_pdfs("./data")
    chunks = chunk_text(pages)
    store_in_pinecone(chunks)