import joblib
import numpy as np
import faiss

# Load embeddings
df = joblib.load("data/dsa/embeddings.joblib")

embeddings = np.vstack(df["embedding"].apply(np.array).values).astype("float32")

dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "data/dsa/vector.index")

print("✅ FAISS index created successfully.")