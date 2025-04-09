#pip install sentence-transformers langchain faiss-cpu transformers torch accelerate fastapi pymupdf uvicornpython-multipart
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import sqlite3
from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import fitz

app = FastAPI()

#Function to read the PDF file uploaded/recieved from the FastAPI
def read_pdf(file: UploadFile):
    pdf_text = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text.strip()

#For chunking the data from PDF to manageable chunks instead of one chunk for the entire PDF
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#Creating embeddings for the data from the PDF
def create_vector_store(chunks):
    embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks], dtype=np.float32)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, chunks 

#Retrieving chunks of data from the PDF relevant to the user query.
def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return " ".join(retrieved_chunks)

# Connecting to a local database file
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

# Creating a table to store chat messages. It will create if no table called chat exists, if it already exists, it won't create a duplicate.
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

#Saving the user query and LLM answer in the database with respect to the session
def save_message(role, message, session_id):
    cursor.execute("INSERT INTO chat (role, message, session_id) VALUES (?, ?, ?)", (role, message, session_id))
    conn.commit()

#Getting the previous chat history from the database to feed as context to the LLM
def get_last_n_messages(n, session_id):
    cursor.execute("""
        SELECT role, message FROM (
            SELECT * FROM chat
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        ) sub
        ORDER BY id ASC
    """, (session_id, n * 2))
    rows = cursor.fetchall()

    history = []
    for role, message in rows:
        history.append({"role": role, "content": message})
    return history

# Load a Hugging Face LLM
qa_model = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", torch_dtype="auto", device_map="auto")

#LLM functionality
@app.post('/generate_answer')
async def generate_answer(
    query: str = Form(...),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(None)
    ):
    
    history = get_last_n_messages(2,session_id)

    if file is None:
        messages = [
            {"role":"system", "content":f"""You are a helpful assistant.
Instructions:
- Respond helpfully and concisely.
- Do not repeat the user's question.
- If the question is vague, ask for clarification.
- If a question is asked, answer with respect to the conversation history. Answer in 1-2 lines.
Conversation History: {history}"""},
            {"role":"user", "content":query},
        ]
    else:
        text = read_pdf(file)
        chunks = chunk_text(text)
        index, stored_chunks = create_vector_store(chunks)
        retrieved_context = retrieve_relevant_chunks(query, index, stored_chunks)

        messages = [
            {"role":"system", "content":f"""You are a helpful assistant.
Instructions:
- Respond helpfully and concisely.
- Do not repeat the user's question.
- If the question is vague, ask for clarification.
- If a question is asked, answer with respect to the conversation history. Answer in 1-2 lines.
- Use this as content: {retrieved_context}
Conversation History: {history}"""},
            {"role":"user", "content":query},
        ]
    response = qa_model(messages, max_new_tokens=500)
    answer = response[0]['generated_text'][-1]

    save_message("user", query, session_id)
    save_message("assistant", answer['content'], session_id)

    return answer