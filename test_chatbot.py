import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
import faiss
import json

import main  # Import main so we can access global variables
from main import app

client = TestClient(app)

# Test 1: UI endpoint
def test_get_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "Gemini MongoDB Chatbot" in response.text

# Test 2: History endpoint initially empty
def test_get_history_initially_empty():
    main.history.clear()
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json()["history"] == []

# Test 3: Ask question (mocked Gemini + FAISS)
@patch("main.genai.embed_content")
@patch("main.llm.generate_content")
def test_ask_question_success(mock_generate, mock_embed):
    # Mock Gemini embedding & response
    mock_embed.return_value = {"embedding": [0.1] * 768}
    mock_generate.return_value.text = "This is a mock answer."

    # Setup fake FAISS index and assign it to main.index
    dummy_vectors = np.array([[0.1] * 768] * 3, dtype="float32")
    index = faiss.IndexFlatL2(768)
    index.add(dummy_vectors)
    main.index = index

    # Dummy chunks
    main.chunk_map.clear()
    main.chunk_map.update({0: "Chunk 0", 1: "Chunk 1", 2: "Chunk 2"})

    response = client.post("/ask", json={"question": "What are the rules?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a mock answer."

# Test 4: History persistence using temp file
def test_history_persistence(tmp_path, monkeypatch):
    test_file = tmp_path / "test_history.json"
    monkeypatch.setattr("main.HISTORY_FILE", str(test_file))

    main.history.clear()
    main.history.appendleft({"question": "Q1", "answer": "A1"})
    main.save_history()

    # Simulate a restart
    main.history.clear()
    main.load_history()

    assert len(main.history) == 1
    assert main.history[0]["question"] == "Q1"
    assert main.history[0]["answer"] == "A1"
