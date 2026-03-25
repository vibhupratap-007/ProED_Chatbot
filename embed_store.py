import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from load import load_pdfs
from chunk import chunk_text

load_dotenv()


def store_in_pinecone(chunks: list[dict]):
    # safety check - no point connecting if nothing to upload
    if not chunks:
        print("No chunks found. Please run chunk.py first.")
        return

    # using sentence transformers to convert text into vectors - free and local
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model ready!")

    # connecting to pinecone using api key from .env file
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "rag-index"

    # check if index already exists so we dont create it twice
    current_index = [i.name for i in pc.list_indexes()]
    if index_name not in current_index:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,        # all-MiniLM-L6-v2 produces 384 dim vectors
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

    # uploading in batches of 50 because of api request size limits
    print(f"\nEmbedding and uploading {len(chunks)} chunks...")
    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        current_batch = chunks[i:i + batch_size]

        # convert chunk text into vectors using sentence transformers
        texts = [c["text"] for c in current_batch]
        encoded_vectors = model.encode(texts, show_progress_bar=False)

        # prepare data in the format pinecone expects
        vector_list = []
        for chunk, embedding in zip(current_batch, encoded_vectors):
            vector_list.append({
                "id": chunk["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            })

        # upload this batch to pinecone
        index.upsert(vectors=vector_list)

        # show progress as percentage
        progress = round((i + len(current_batch)) / len(chunks) * 100)
        print(f"  Uploaded {i + len(current_batch)}/{len(chunks)} chunks ({progress}% done)")

    print(f"\nDone! All {len(chunks)} chunks stored in Pinecone!")


if __name__ == "__main__":
    pages = load_pdfs("./data")
    chunks = chunk_text(pages)
    store_in_pinecone(chunks)