# 🎓 RAG-Based AI Teaching Assistant (Video Pipeline Version)

An AI-powered Retrieval-Augmented Generation (RAG) system that processes educational videos and enables contextual question answering.

This project focuses on building a complete backend pipeline for:

- 🎥 Video Processing
- 🎙 Speech-to-Text (Whisper)
- 📄 JSON Preprocessing
- 🧠 Embedding Generation
- 🔎 FAISS Vector Indexing
- 🤖 LLM-based Question Answering

---

## 🚀 Project Overview

This system performs the following steps:

1. Takes uploaded video/audio input
2. Converts speech to text using Whisper
3. Stores transcript in structured JSON format
4. Generates embeddings for text chunks
5. Builds a FAISS vector index
6. Retrieves relevant context
7. Uses an LLM to generate answers

This project demonstrates a real-world RAG architecture implementation.

---

## 🧠 Architecture

Video Input
↓
Whisper Transcription
↓
JSON Processing
↓
Embedding Generation
↓
FAISS Index Creation
↓
Context Retrieval
↓
LLM Response



---

## 🛠 Tech Stack

- Python
- Whisper
- SentenceTransformers
- FAISS
- NumPy
- Hugging Face Inference API

---

## 📂 Project Structure
├── build_faiss_index.py
├── process_uploaded_video.py
├── process_incoming.py
├── mp3_to_json.py
├── preprocess_json.py
├── data/
├── audios/
├── videos/
└── requirements.txt

---

## ▶ How to Run

1. Install dependencies:
pip install -r requirements.txt


2. Process video:

python process_uploaded_video.py


3. Generate embeddings & FAISS index:

python build_faiss_index.py


4. Run question-answer pipeline:

python process_incoming.py


---

## 🔐 Environment Setup

Set your Hugging Face token:

Windows:

setx HF_TOKEN "your_token_here"


Linux/Mac:

export HF_TOKEN="your_token_here"


---

## 📌 Key Learnings

- End-to-end RAG implementation
- Vector database engineering
- Multimedia AI pipeline design
- Production-style modular architecture
- Secure API handling

---

## 👨‍💻 Developed By

**Shivam Kumar**  
B.Tech AI & ML  
Aspiring AI/ML Engineer