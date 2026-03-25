# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

load_dotenv()

# initialize fastapi app
app = FastAPI()

# allow requests from vercel frontend
# cors lets the frontend talk to our backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production replace * with your vercel url
    allow_methods=["*"],
    allow_headers=["*"],
)

# load embedding model once when server starts - not on every request
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")

# connect to pinecone once when server starts
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index")
print("Pinecone connected!")


# this defines what the request body should look like
class QuestionRequest(BaseModel):
    question: str


# this defines what the response will look like
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list


# health check endpoint - to verify server is running
@app.get("/")
def root():
    return {"status": "RAG server is running!"}


# main endpoint - frontend sends question here
@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question

    # convert question into vector
    question_vector = model.encode(question).tolist()

    # search pinecone for top 5 relevant chunks
    search_results = index.query(
        vector=question_vector,
        top_k=5,
        include_metadata=True
    )

    # check if anything was found
    if not search_results["matches"]:
        return AnswerResponse(
            question=question,
            answer="I could not find relevant information in the documents.",
            sources=[]
        )

    # build context and sources list from retrieved chunks
    retrieved_context = ""
    sources = []

    for result in search_results["matches"]:
        score = result["score"]

        # skip low relevance chunks
        if score < 0.5:
            continue

        source = result["metadata"]["source"]
        page = result["metadata"]["page"]
        text = result["metadata"]["text"]

        retrieved_context += f"\n[Source: {source}, Page {page}]\n{text}\n"
        sources.append({
            "source": source,
            "page": page,
            "score": round(score, 2)
        })

    # send question + context to groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a financial aid compliance assistant.
                Answer questions strictly using the context provided.
                Always mention which document and page your answer comes from.
                If you cannot find the answer say
                'I could not find this information in the provided documents.'"""
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

    answer = response.choices[0].message.content

    # return answer + sources back to frontend
    return AnswerResponse(
        question=question,
        answer=answer,
        sources=sources
    )