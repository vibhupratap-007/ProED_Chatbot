import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

load_dotenv()

def ask_question(question: str):
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connection to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("rag-index")

    # Embed the question
    print(f"\nSearching for: '{question}'")
    query_embedding = model.encode(question).tolist()

    # Search top 3 most relevant chunks
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Build context from retrieved chunks
    context = ""
    print("\nRetrieved sources:")
    for i, match in enumerate(results["matches"]):
        source = match["metadata"]["source"]
        page = match["metadata"]["page"]
        text = match["metadata"]["text"]
        score = match["score"]
        print(f"  {i+1}. {source} | Page {page} | Score: {score:.2f}")
        context += f"\n[Source: {source}, Page {page}]\n{text}\n"

    # Ask Groq (Free!)
    print("\nAsking Groq (Llama 3)...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions using ONLY the provided context. If the answer is not in the context, say 'I don't know based on the provided documents.'"
            },
            {
                "role": "user",
                "content": f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
            }
        ]
    )

    print("\nAnswer:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    ask_question("What is the last date to register?")