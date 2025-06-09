# Gemini MongoDB Chatbot

An AI-powered chatbot built with **FastAPI**, **Google Gemini Generative AI**, and **MongoDB**.

This chatbot retrieves knowledge base documents from MongoDB, embeds them using Gemini Embeddings, and uses FAISS for fast similarity search. When you ask a question, it finds the most relevant chunks and uses Gemini's language model to articulate a precise answer.

---

## 🔥 Features

* Uses Google Gemini Embeddings for semantic search
* Stores and retrieves documents from MongoDB Atlas
* Fast vector similarity search using FAISS
* Easy-to-use web chat UI with FastAPI
* CORS enabled for frontend flexibility

---

## 💻 Prerequisites

* Python 3.8+
* MongoDB Atlas account with a populated collection
* Google Generative AI API key (Gemini)
* VS Code or any code editor

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gemini-mongo-chatbot.git
cd gemini-mongo-chatbot
```

### 2. Open in VS Code

Open this folder in VS Code or your favorite editor.

### 3. Create and activate a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API keys and MongoDB URI

Open `main.py` and update:

```python
# Replace with your actual Google Generative AI API key
genai.configure(api_key="YOUR_GOOGLE_GENERATIVE_AI_API_KEY")

# Replace with your actual MongoDB Atlas connection URI and the respective Collections
mongo_uri = "YOUR_MONGODB_CONNECTION_STRING"
```

---

### 6. Prepare MongoDB

Make sure your MongoDB Atlas database contains:

* Database: `knowledge_base` (or a name of your wish)
* Collection: `Building_Rules` (or a name of your wish)
* Documents structured as:

```json
{
  "chunk_number": 0,
  "content": "Text of the first chunk..."
}
```



---

### 7. Run the FastAPI server

Name the python file as .main.py' and run the below code in terminal.

```bash
uvicorn main:app --reload
```

---

### 8. Wait for indexing to complete

On startup, watch the console logs:

```
🔄 Loading chunks from MongoDB...
🔎 Embedding chunks using Gemini...
✅ Indexed <number> chunks using Gemini Embeddings.
```

This means your knowledge base has been loaded, embedded, and indexed — the app is ready to serve queries.

---

### 9. Use the Chatbot UI

Open your browser and visit:

```
http://localhost:8000/
```

Type your questions into the chat interface and get AI-generated answers based on your knowledge base.

---

## 🧑‍💻 How It Works

* On startup, text chunks from MongoDB are fetched and embedded with Gemini Embeddings.
* The embeddings are indexed with FAISS for efficient nearest neighbor search.
* When a question is asked, it's embedded and matched with the closest text chunks.
* The top relevant chunks provide context to the Gemini LLM for generating an answer.
* The answer is sent back and displayed in the chat UI.

---

## 📦 API Endpoints

| Method | Endpoint | Description                                   |
| ------ | -------- | --------------------------------------------- |
| GET    | `/`      | Returns the chat UI HTML                      |
| POST   | `/ask`   | Ask a question. Body: `{ "question": "..." }` |

---

## ⚠️ Troubleshooting

* Ensure your Google Generative AI API key and MongoDB URI are correct.
* Verify your MongoDB cluster is accessible and the `Building_Rules` collection is populated.
* If indexing hangs or fails, check your internet connection (embedding calls require network).
* For any other errors, check console logs for stack traces.

---

## 🙋‍♂️ Author


[Mokhithraa. J](https://github.com/MokhithraaJ)
