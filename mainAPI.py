#pip install sentence-transformers langchain faiss-cpu transformers torch accelerate fastapi pymupdf uvicornpython-multipart
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import sqlite3
import fitz

app = FastAPI()

# Function to read the PDF file uploaded/received from FastAPI
def read_pdf(file: UploadFile):
    try:
        pdf_text = ""
        with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()
        return pdf_text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

# Chunking the PDF data
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError(f"Error loading embedding model: {e}")

def create_vector_store(chunks):
    try:
        embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks], dtype=np.float32)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, chunks
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {e}")

#Retrieving chunks of data from the PDF relevant to the user query.
def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    try:
        query_embedding = embedding_model.encode(query).reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        return " ".join([chunks[i] for i in indices[0]])
    except Exception as e:
        raise RuntimeError(f"Error retrieving relevant chunks: {e}")

# SQLite connection
try:
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
except sqlite3.Error as e:
    raise RuntimeError(f"Database connection failed: {e}")

#Saving the user query and LLM answer in the database with respect to the session
def save_message(role, message, session_id):
    try:
        cursor.execute("INSERT INTO chat (role, message, session_id) VALUES (?, ?, ?)", (role, message, session_id))
        conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error saving message to DB: {e}")

#Getting the previous chat history from the database to feed as context to the LLM
def get_last_n_messages(n, session_id):
    try:
        cursor.execute("""
            SELECT role, message FROM (
                SELECT * FROM chat
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) sub
            ORDER BY id ASC
        """, (session_id, n * 2)) #n*2 because we need the user query and LLM response to feed as context
        rows = cursor.fetchall()
        return [{"role": role, "content": message} for role, message in rows]
    except sqlite3.Error as e:
        raise RuntimeError(f"Error retrieving chat history: {e}")

# Load Hugging Face model
try:
    qa_model = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", torch_dtype="auto", device_map="auto")
except Exception as e:
    raise RuntimeError(f"Error loading LLM pipeline: {e}")

#LLM Functionality
@app.post('/generate_answer')
async def generate_answer(
    query: str = Form(...),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        history = get_last_n_messages(2, session_id)

        if file is None:
            messages = [
                {"role": "system", "content": f"""
                You are a helpful assistant.
                Instructions:
                - Respond helpfully and concisely.
                - Do not repeat the user's question.
                - If the question is vague, ask for clarification.
                - Use this conversation history: {history}
                """},
                {"role": "user", "content": query},
            ]
        else:
            text = read_pdf(file)
            chunks = chunk_text(text)
            index, stored_chunks = create_vector_store(chunks)
            retrieved_context = retrieve_relevant_chunks(query, index, stored_chunks)

            messages = [
                {"role": "system", "content": f"""
                You are a helpful assistant.
                Instructions:
                - Respond helpfully and concisely.
                - Do not repeat the user's question.
                - If the question is vague, ask for clarification.
                - Use this as content: {retrieved_context}
                Conversation History: {history}
                """},
                {"role": "user", "content": query},
            ]

        response = qa_model(messages, max_new_tokens=500)
        answer = response[0]['generated_text'][-1]

        save_message("user", query, session_id)
        save_message("assistant", answer['content'], session_id)

        return {"answer": answer}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
