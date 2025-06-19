from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import faiss
import numpy as np
import google.generativeai as genai
from collections import deque
import json
import os

# Configure Gemini
genai.configure(api_key="AIzaSyBBjSELLLXtwbyV7KqklTHsLNnY-04PPco")
llm = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")

# MongoDB setup
mongo_uri = "mongodb+srv://mokhithraa2004:wfwsxhTK9x8TuKUW@cluster0.p5j2bi5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
collection = client["knowledge_base"]["Building_Rules"]

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

# History management
HISTORY_FILE = "history.json"
MAX_HISTORY = 3
history = deque(maxlen=MAX_HISTORY)

def load_history():
    global history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                saved = json.load(f)
                history = deque(saved, maxlen=MAX_HISTORY)
            print("✅ History loaded from file.")
        else:
            print("ℹ️ No history file found, starting fresh.")
    except Exception as e:
        print(f"⚠️ Error loading history file: {e}. Starting fresh.")
        history = deque(maxlen=MAX_HISTORY)

def save_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(list(history), f, indent=4)
        print("✅ History saved.")
    except Exception as e:
        print(f"⚠️ Error saving history: {e}")

@app.on_event("startup")
async def startup_event():
    global chunk_map, index
    load_history()

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

    # Build history context here
    history_context = ""
    for h in list(history):
        history_context += f"Previous Question: {h['question']}\nPrevious Answer: {h['answer']}\n\n"

    # Complete prompt including history
    prompt = f"""
You are a Real Estate expert chatbot.

Use the following retrieved document context to answer user's question:

Retrieved Context:
{context}

Recent Conversation History:
{history_context}

Now answer the user's current question based on both retrieved context and recent history.

Current Question:
{query.question}

Answer:
"""

    response = llm.generate_content(prompt)
    answer = response.text.strip()

    # Update local history
    history.appendleft({"question": query.question, "answer": answer})
    save_history()

    return {"answer": answer}

@app.get("/history")
async def get_history():
    return {"history": list(history)}

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Gemini MongoDB Chatbot</title>
  <style>
    body {font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 0; height: 100vh; display: flex;}
    #sidebar {
      width: 300px; background-color: #222831; color: white; padding: 20px;
      box-shadow: 2px 0 5px rgba(0,0,0,0.1); display: flex; flex-direction: column;
    }
    #sidebar h2 {margin-top: 0; font-size: 20px; margin-bottom: 20px;}
    #historyContainer { flex: 1; overflow-y: auto; }
    #historyList { list-style-type: none; padding: 0; margin: 0; }
    .historyItem {
      padding: 10px; background-color: #393e46; margin-bottom: 10px; border-radius: 5px;
      word-break: break-word; position: relative;
    }
    .answer { display: none; margin-top: 10px; }
    .toggleBtn {
      position: absolute; top: 10px; right: 10px;
      background: #0077cc; border: none; color: white; padding: 5px 10px; border-radius: 5px; cursor: pointer;
    }
    .toggleBtn:hover { background: #005fa3; }
    #chat { flex: 1; display: flex; flex-direction: column; padding: 20px; }
    #messages {
      flex: 1; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;
      border-radius: 5px; background-color: white;
    }
    .message {
      margin-bottom: 15px; padding: 10px 15px; border-radius: 10px;
      line-height: 1.6; max-width: 80%;
    }
    .user { background-color: #d0ebff; align-self: flex-end; text-align: right; }
    .bot { background-color: #e6ffe6; align-self: flex-start; white-space: pre-line; }
    textarea {
      width: 100%; height: 60px; padding: 10px; resize: none;
      border-radius: 5px; border: 1px solid #ccc;
    }
    button.askBtn {
      padding: 10px 20px; background-color: #0077cc; color: white;
      border: none; border-radius: 5px; margin-top: 10px; cursor: pointer;
    }
    button.askBtn:hover { background-color: #005fa3; }
  </style>
</head>

<body>
  <div id="sidebar">
    <h2>Recent History</h2>
    <div id="historyContainer">
      <ul id="historyList"></ul>
    </div>
  </div>

  <div id="chat">
    <h2>Gemini MongoDB Chatbot</h2>
    <div id="messages"></div>
    <textarea id="question" placeholder="Ask something..."></textarea><br/>
    <button class="askBtn" onclick="askQuestion()">Ask</button>
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
        .replace(/\d+\\.\\s/g, "<br/><strong>$&</strong>");

      messages.innerHTML += `<div class="message bot"><strong>Bot:</strong><br/>${formattedAnswer}</div>`;
      messages.scrollTop = messages.scrollHeight;

      updateHistory();
    }

    async function updateHistory() {
      const response = await fetch("/history");
      const data = await response.json();
      const historyList = document.getElementById("historyList");
      historyList.innerHTML = "";

      data.history.forEach((item, idx) => {
        const listItem = document.createElement("li");
        listItem.className = "historyItem";
        listItem.innerHTML = `<strong>Q:</strong> ${item.question}
          <button class="toggleBtn" onclick="toggleAnswer(${idx})">Show</button>
          <div class="answer" id="answer-${idx}"><strong>A:</strong> ${item.answer}</div>`;
        historyList.appendChild(listItem);
      });
    }

    function toggleAnswer(idx) {
      const answerDiv = document.getElementById(`answer-${idx}`);
      const button = answerDiv.previousElementSibling;
      if (answerDiv.style.display === "none" || answerDiv.style.display === "") {
        answerDiv.style.display = "block";
        button.textContent = "Hide";
      } else {
        answerDiv.style.display = "none";
        button.textContent = "Show";
      }
    }

    window.onload = updateHistory;
  </script>
</body>
</html>
"""
