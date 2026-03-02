# 🎓 RAG-Based AI Teaching Assistant (Video + YouTube Supported)

An AI-powered Retrieval-Augmented Generation (RAG) system that processes educational videos (Local or YouTube) and enables contextual question answering using vector search and LLMs.

This project demonstrates a complete end-to-end backend RAG pipeline built from scratch.

---

## 🚀 Features

✅ Process Local Video Files  
✅ Process YouTube Video Links  
✅ Automatic Audio Extraction  
✅ Whisper Speech-to-Text Transcription  
✅ JSON Structuring & Chunking  
✅ Embedding Generation (Sentence Transformers)  
✅ FAISS Vector Index Creation  
✅ Context Retrieval  
✅ LLM-Based Answer Generation  
✅ CLI-Based Interactive Assistant  

---

## 🧠 How It Works

### Step 1: Video Input
User can:
- Provide local video file path
- Provide YouTube video link

### Step 2: Audio Extraction
Video is converted into audio format.

### Step 3: Transcription
Whisper model converts speech into text.

### Step 4: Preprocessing
Transcript is:
- Cleaned
- Chunked
- Converted into structured JSON

### Step 5: Embeddings
Text chunks are converted into embeddings using SentenceTransformers.

### Step 6: FAISS Indexing
Vector database is built for fast similarity search.

### Step 7: Question Answering
User asks questions.
System retrieves relevant context.
LLM generates final response.

---

## 🧠 System Architecture

Video (Local / YouTube)
        ↓
Audio Extraction
        ↓
Whisper Transcription
        ↓
JSON Chunking
        ↓
Embedding Generation
        ↓
FAISS Vector Index
        ↓
Context Retrieval
        ↓
LLM Response

---

## 🛠 Tech Stack

- Python
- OpenAI Whisper
- SentenceTransformers
- FAISS (Vector Search)
- NumPy
- Hugging Face Inference API
- yt-dlp (for YouTube download)

---

## 📂 Project Structure
├── build_faiss_index.py
├── process_uploaded_video.py
├── process_incoming.py
├── mp3_to_json.py
├── preprocess_json.py
├── data/
│ ├── dsa/
│ └── ...
├── audios/
├── videos/
├── youtube/
└── requirements.txt

---

## ▶ How to Run

### 1️⃣ Install Dependencies
pip install -r requirements.txt


---

### 2️⃣ Run the Assistant


python process_incoming.py


You will see:


1 - DSA Course
2 - Process New Video (Local or YouTube)
3 - Ask from Uploaded Video
4 - Exit

---

### 🎥 To Process Local Video

Choose option:

Then enter:
- Local video file path

Example:Example: https://youtube.com/watch?v=example


System will:
- Download video
- Extract audio
- Transcribe
- Build embeddings
- Create FAISS index

---

### ❓ Ask Questions

Choose option:


3


Then enter your question related to the processed video.

---

## 🔐 Environment Setup

Set your Hugging Face API token:

### Windows


setx HF_TOKEN "your_token_here"


### Linux / Mac


export HF_TOKEN="your_token_here"


---

## 📌 Key Highlights

- Real-world RAG architecture implementation
- Vector database engineering using FAISS
- Multimedia AI processing pipeline
- YouTube + Local Video integration
- Modular production-style codebase
- Secure API handling

---

## 📈 Future Improvements

- Web-based UI (Gradio / FastAPI)
- Multi-video knowledge base
- Persistent vector storage
- Deployment on Hugging Face Spaces
- Streamlit interface

---

## 👨‍💻 Developed By

**Shivam Kumar**  
B.Tech AI & ML  
Aspiring AI/ML Engineer  

🚀 Passionate about AI Systems, RAG Architecture & Real-World ML Deployment
