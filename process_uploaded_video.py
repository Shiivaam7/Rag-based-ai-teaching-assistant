import os
import numpy as np
import joblib
import faiss
import requests
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip

EMBED_MODEL = "bge-m3"


# =========================
# AUDIO EXTRACTION
# =========================
def extract_audio(video_path, output_audio_path):
    os.makedirs("temp", exist_ok=True)

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    clip.close()


# =========================
# TRANSCRIBE (MEDIUM MODEL + GPU)
# =========================
def transcribe_audio(audio_path):
    print("Loading Whisper Medium Model (GPU)...")

    model = WhisperModel(
        "medium",
        device="cuda",
        compute_type="float16"
    )

    segments, _ = model.transcribe(audio_path)

    transcript = []

    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": text
            })

    if not transcript:
        raise Exception("No speech detected in video.")

    return transcript


# =========================
# CHUNKING
# =========================
def chunk_transcript(transcript, chunk_size=5):
    chunks = []

    for i in range(0, len(transcript), chunk_size):
        group = transcript[i:i+chunk_size]
        combined_text = " ".join([x["text"] for x in group]).strip()

        if combined_text:
            chunks.append({
                "title": "Uploaded Video",
                "number": 1,
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": combined_text
            })

    return chunks


# =========================
# EMBEDDING
# =========================
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text_list
        },
        timeout=300
    )

    data = r.json()

    if "embeddings" in data:
        return data["embeddings"]
    elif "embedding" in data:
        return [data["embedding"]]
    else:
        print("Embedding API Response:", data)
        raise Exception("Embedding failed. Check Ollama model/server.")


# =========================
# BUILD FAISS (COSINE SIMILARITY + SAFE)
# =========================
def build_faiss(chunks, save_folder):

    print("Cleaning chunks...")
    cleaned_chunks = []

    for c in chunks:
        if c["text"] and c["text"].strip() != "":
            cleaned_chunks.append(c)

    if not cleaned_chunks:
        raise Exception("All chunks empty. Cannot build index.")

    texts = [c["text"] for c in cleaned_chunks]

    print(f"{len(texts)} valid chunks found.")
    print("Creating embeddings...")

    embeddings = create_embedding(texts)

    emb_matrix = np.array(embeddings, dtype="float32")

    # Remove NaN rows
    mask = ~np.isnan(emb_matrix).any(axis=1)
    emb_matrix = emb_matrix[mask]

    if emb_matrix.shape[0] == 0:
        raise Exception("Embeddings contain only NaN values.")

    # 🔥 Normalize for cosine similarity
    faiss.normalize_L2(emb_matrix)

    dimension = emb_matrix.shape[1]

    # 🔥 Use Inner Product index (Cosine Similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(emb_matrix)

    os.makedirs(save_folder, exist_ok=True)

    joblib.dump(cleaned_chunks, os.path.join(save_folder, "metadata.joblib"))
    faiss.write_index(index, os.path.join(save_folder, "vector.index"))

    print("FAISS index built using Cosine Similarity successfully!")


# =========================
# MAIN PROCESS
# =========================
def process_video(video_path, video_name):

    save_folder = f"data/uploads/{video_name}"
    audio_path = f"temp/{video_name}.wav"

    try:
        print("Extracting audio...")
        extract_audio(video_path, audio_path)

        print("Transcribing...")
        transcript = transcribe_audio(audio_path)

        print("Chunking...")
        chunks = chunk_transcript(transcript)

        print("Building FAISS index...")
        build_faiss(chunks, save_folder)

        print("Upload video processed successfully!")
        print("Ready for Q&A.")

    except Exception as e:
        print("Error occurred during processing:")
        print(e)