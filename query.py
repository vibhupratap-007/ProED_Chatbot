import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

load_dotenv()


def ask_question(question: str):
    # loading same embedding model used during ingestion - must match
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # connecting to pinecone and opening our index
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("rag-index")

    # convert the question into a vector so we can search pinecone
    print(f"\nSearching for: '{question}'")
    question_vector = model.encode(question).tolist()

    # find top 5 chunks most similar to the question
    search_results = index.query(
        vector=question_vector,
        top_k=5,
        include_metadata=True
    )

    # check if pinecone returned anything at all
    if not search_results["matches"]:
        print("No relevant chunks found in the documents.")
        return

    # build context from retrieved chunks to send to the LLM
    retrieved_context = ""
    print("\nRetrieved sources:")
    for i, result in enumerate(search_results["matches"]):
        source = result["metadata"]["source"]
        page = result["metadata"]["page"]
        text = result["metadata"]["text"]
        score = result["score"]

        # skip chunks below 0.5 score - not relevant enough
        if score < 0.5:
            print(f"  Skipping low relevance result (score: {score:.2f})")
            continue

        print(f"  {i+1}. {source} | Page {page} | Score: {score:.2f}")
        retrieved_context += f"\n[Source: {source}, Page {page}]\n{text}\n"

    # send question + context to groq llama model
    print("\nAsking Groq (Llama 3)...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. 
                Answer questions using ONLY the provided context. 
                Always mention which document and page your answer comes from.
                If the answer is not in the context say 
                'I don't know based on the provided documents.'"""
            },
            {
                "role": "user",
                "content": f"""CONTEXT:
{retrieved_context}

QUESTION:
{question}

ANSWER:"""
            }
        ]
    )

    print("\nAnswer:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    ask_question("What are the required documents for verification?")
