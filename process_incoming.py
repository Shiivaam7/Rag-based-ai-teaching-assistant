import os
import numpy as np
import pandas as pd
import joblib
import requests
import faiss
import yt_dlp
from process_uploaded_video import process_video


EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"
TOP_RESULTS = 5


# =========================
# EMBEDDING FUNCTION
# =========================
def create_embedding(text_list):
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": text_list},
            timeout=180
        )

        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        elif "embedding" in data:
            return [data["embedding"]]
        else:
            raise Exception(f"Unexpected embedding response: {data}")

    except Exception as e:
        print("Embedding Error:", e)
        print("Make sure Ollama is running.")
        return None


# =========================
# LLM INFERENCE
# =========================
def inference(prompt):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=300
        )

        data = r.json()

        if "response" in data:
            return data["response"]
        else:
            raise Exception(f"Unexpected LLM response: {data}")

    except Exception as e:
        print("LLM Error:", e)
        return "Error generating answer."


# =========================
# YOUTUBE DOWNLOAD
# =========================
def download_youtube_video(url, save_name):
    os.makedirs("videos", exist_ok=True)
    output_path = f"videos/{save_name}.mp4"

    ydl_opts = {
        "format": "mp4",
        "outtmpl": output_path,
        "quiet": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path


# =========================
# LOAD DSA COURSE DATA
# =========================
def load_dsa_data():
    print("Loading DSA embeddings...")
    df = joblib.load("data/dsa/embeddings.joblib")
    index = faiss.read_index("data/dsa/vector.index")
    return df, index


# =========================
# LOAD UPLOADED VIDEO DATA
# =========================
def load_uploaded_video_data(video_name):
    path = f"data/uploads/{video_name}"

    if not os.path.exists(path):
        print("No uploaded video found with that name.")
        return None, None

    metadata = joblib.load(f"{path}/metadata.joblib")
    df = pd.DataFrame(metadata)
    index = faiss.read_index(f"{path}/vector.index")

    return df, index


# =========================
# ASK FUNCTION (COSINE FIXED)
# =========================
def ask_question(df, index):

    while True:
        question = input("\nAsk Question (or type exit): ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        # 1️⃣ Create Query Embedding
        embedding_result = create_embedding([question])
        if embedding_result is None:
            continue

        question_embedding = np.array(embedding_result[0]).astype("float32").reshape(1, -1)

        # 🔥 IMPORTANT — Normalize for cosine similarity
        faiss.normalize_L2(question_embedding)

        # 2️⃣ Search FAISS
        D, I = index.search(question_embedding, TOP_RESULTS)

        print("\n🔎 Top Retrieved Chunks:\n")

        retrieved_chunks = []

        for rank, idx in enumerate(I[0]):
            chunk = df.iloc[idx]
            score = D[0][rank]  # cosine similarity (0 to 1)

            print(f"Rank {rank+1}")
            print(f"Cosine Similarity: {score:.4f}")
            print(f"Timestamp: {chunk.get('start', 'N/A')} - {chunk.get('end', 'N/A')}")
            print(f"Text Preview: {chunk['text'][:300]}...")
            print("-" * 60)

            retrieved_chunks.append(chunk["text"])

        # 3️⃣ Build Prompt
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful teaching assistant.

Context from course video:
{context}

User Question:
{question}

Instructions:
- Answer clearly.
- Mention video number and timestamp if available.
- Be helpful and guide the student.
"""

        # 4️⃣ LLM Response
        response = inference(prompt)

        print("\n🧠 Final Answer:\n")
        print(response)
        print("\n" + "=" * 80)


# =========================
# MAIN MENU
# =========================
def main():
    while True:
        print("\n========== RAG Teaching Assistant ==========")
        print("1 - DSA Course")
        print("2 - Process New Video (Local or YouTube)")
        print("3 - Ask from Uploaded Video")
        print("4 - Exit")

        choice = input("Enter choice: ").strip()

        # ---- DSA COURSE ----
        if choice == "1":
            df, index = load_dsa_data()
            ask_question(df, index)

        # ---- PROCESS NEW VIDEO ----
        elif choice == "2":
            video_input = input("Enter video file path OR YouTube link: ").strip()
            save_name = input("Enter name to save (no spaces): ").strip()

            if video_input.startswith("http"):
                print("Downloading YouTube video...")
                video_path = download_youtube_video(video_input, save_name)
            else:
                video_path = video_input

            print("Processing video...")
            process_video(video_path, save_name)
            print("Video processed successfully!")

        # ---- ASK FROM UPLOADED VIDEO ----
        elif choice == "3":
            video_name = input("Enter saved video name: ").strip()
            df, index = load_uploaded_video_data(video_name)

            if df is not None:
                ask_question(df, index)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()