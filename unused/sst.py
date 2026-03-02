import whisper

model = whisper.load_model("large-v2")
result = model.transcribe(audio = "5_Coding Insertion Operation in Array in Data Structures in C language - CodeWithHarry.mp3",
                          language = "hi",
                          task = "tanslate")
print(result["text"])
