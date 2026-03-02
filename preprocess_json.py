# import pandas as pd
# import numpy as np
# import requests
# from sklearn.metrics.pairwise import cosine_similarity
# import ast

# MODEL = "bge-m3"

# def create_embedding(text_list):
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": MODEL,
#         "input": text_list
#     }, timeout=180)

#     data = r.json()

#     if "error" in data:
#         raise Exception(data["error"])

#     if "embeddings" in data:
#         return data["embeddings"]
#     elif "embedding" in data:
#         return data["embedding"]
#     else:
#         raise KeyError(f"Unexpected response keys: {data.keys()}")


# # ✅ Load saved embeddings
# df = pd.read_csv("all_chunks_with_embeddings.csv")

# # CSV me embedding string ban jaata hai, usko list me convert karo
# df["embedding"] = df["embedding"].apply(ast.literal_eval)

# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0]

# similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()

# top_results = 3
# max_indx = similarities.argsort()[::-1][:top_results]

# new_df = df.loc[max_indx]
# print("\nTop Results:\n")
# print(new_df[["title", "number", "text", "source_file"]])



# .
#   .
#   .
#   .
#code for converting csv to joblib file
import pandas as pd
import ast
import joblib

CSV_FILE = "all_chunks_with_embeddings.csv"
JOBLIB_FILE = "embeddings.joblib"

print(" Loading CSV...")
df = pd.read_csv(CSV_FILE)

print(" Converting embedding column from string -> list...")
df["embedding"] = df["embedding"].apply(ast.literal_eval)

print(" Saving dataframe to joblib...")
joblib.dump(df, JOBLIB_FILE)

print(f"\n DONE! Saved: {JOBLIB_FILE}")
print(df.head())

