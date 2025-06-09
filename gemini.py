from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import faiss
import numpy as np
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="YOUR_GOOGLE_GENERATIVE_AI_API_KEY")
llm = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")

# MongoDB setup
mongo_uri = "YOUR_MONGODB_CONNECTION_STRING"
client = MongoClient(mongo_uri)
collection = client["knowledge_base"]["Building_Rules"] #you may change the collection name 

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

chunk_map = {}
index = None

@app.on_event("startup")
async def startup_event():
    global chunk_map, index
    print("🔄 Loading chunks from MongoDB...")
    chunks = list(collection.find({}))
    texts = [chunk["content"] for chunk in sorted(chunks, key=lambda x: x["chunk_number"])]
    chunk_map = {i: text for i, text in enumerate(texts)}

    print("🔎 Embedding chunks using Gemini...")
    embeddings = [
        genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )["embedding"]
        for text in texts
    ]
    vectors = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    print(f"✅ Indexed {len(texts)} chunks using Gemini Embeddings.")

@app.post("/ask")
async def ask_question(query: Query):
    user_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query.question,
        task_type="retrieval_query"
    )["embedding"]
    user_vector = np.array([user_embedding], dtype="float32")

    D, I = index.search(user_vector, k=3)
    context = "\n".join(chunk_map[i] for i in I[0])

    prompt = f"""You are an AI assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {query.question}\nAnswer:"""
    response = llm.generate_content(prompt)
    return {"answer": response.text.strip()}

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Gemini MongoDB Chatbot</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f4f9;
      padding: 20px;
    }
    #chat {
      max-width: 800px;
      margin: auto;
      background: white;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 0 10px #ccc;
    }
    .message {
      margin-bottom: 15px;
      padding: 10px 15px;
      border-radius: 10px;
      line-height: 1.6;
    }
    .user {
      background-color: #d0ebff;
      text-align: right;
    }
    .bot {
      background-color: #e6ffe6;
      white-space: pre-line;
    }
    #messages {
      height: 400px;
      overflow-y: auto;
      border: 1px solid #ddd;
      padding: 10px;
      margin-bottom: 10px;
      border-radius: 5px;
    }
    textarea {
      width: 100%;
      height: 60px;
      padding: 10px;
      resize: none;
    }
    button {
      padding: 10px 20px;
      background-color: #0077cc;
      color: white;
      border: none;
      border-radius: 5px;
      margin-top: 10px;
      cursor: pointer;
    }
    button:hover {
      background-color: #005fa3;
    }
  </style>
</head>
<body>
  <div id="chat">
    <h2>Gemini MongoDB Chatbot</h2>
    <div id="messages"></div>
    <textarea id="question" placeholder="Ask something..."></textarea><br/>
    <button onclick="askQuestion()">Ask</button>
  </div>

  <script>
    async function askQuestion() {
      const questionInput = document.getElementById("question");
      const question = questionInput.value.trim();
      if (!question) return;

      const messages = document.getElementById("messages");
      messages.innerHTML += `<div class="message user"><strong>You:</strong><br/>${question}</div>`;
      questionInput.value = "";

      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      const data = await response.json();
      const formattedAnswer = data.answer
        .replace(/\\n/g, "<br/>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\d+\.\s/g, "<br/><strong>$&</strong>");

      messages.innerHTML += `<div class="message bot"><strong>Bot:</strong><br/>${formattedAnswer}</div>`;
      messages.scrollTop = messages.scrollHeight;
    }
  </script>
</body>
</html>
"""
