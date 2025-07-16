from fastapi import FastAPI
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

paraphrase = pipeline("text2text-generation", model="Vamsi/T5-Paraphrase-Paws")

@app.get("/")
def read_root():
    return {"message": "Plagiarism Remover API is running."}

@app.post("/paraphrase")
def paraphrase_text(input_text: str):
    result = paraphrase(input_text, max_length=256, do_sample=True, top_k=120, top_p=0.95, num_return_sequences=1)
    return {"output": result[0]['generated_text']}
