import sqlite3

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

def display_chat_history():
    cursor.execute("SELECT * FROM chat ORDER BY id ASC")
    rows = cursor.fetchall()

    if not rows:
        print("No chat history found.")
        return

    print("Chat History:\n")
    for row in rows:
        print(row)

# display_chat_history()
# cursor.execute("""drop table chat""")

import requests

# URL of your FastAPI endpoint
url = "http://localhost:8000/generate_answer"

# Inputs
# query = "Can you tell me what this PDF is about?"
# query = "Can you tell me about the PAC learnability in short?"
# query = "You mentioned complexity of the data, right? Can you please elaborate in short?"
query = "Did we have a conversation about PAC or no?"
session_id = "test_session_124"
pdf_file_path = "I:\\Taimur Freelancing, Content Writing\\AI Stuff\\Practice\\Pdf-RAG\\1709.02840v3.pdf"

# Open the file in binary mode if it exists, else skip
files = {"file": open(pdf_file_path, "rb")} if pdf_file_path else {}

# Prepare form data
data = {
    "query": query,
    "session_id": session_id
}

# Send POST request
response = requests.post(url, data=data, files=files)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
