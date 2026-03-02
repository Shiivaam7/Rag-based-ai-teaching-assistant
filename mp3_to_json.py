import whisper
import json
import os

# GPU pe force
model = whisper.load_model("large-v2").to("cuda")

os.makedirs("jsons", exist_ok=True)

audios = os.listdir("audios")

for audio in audios:

    output_path = f"jsons/{audio}.json"

    # ✅ Resume: agar already json ban chuka hai to skip
    if os.path.exists(output_path):
        print("Skipping (already done):", audio)
        continue

    # filename parsing
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
    else:
        number = "NA"
        title = audio[:-4]

    print("Processing:", number, title)

    result = model.transcribe(
        audio=f"audios/{audio}",
        language="hi",
        task="translate",
        word_timestamps=False
    )

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            
            
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

    print("Saved:", output_path)
