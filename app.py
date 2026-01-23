from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import make_demo_db, ask

app = FastAPI()

# ✅ CORS FIX — THIS IS THE KEY
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fadimbarki87.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB once
conn = make_demo_db()

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask_agent(q: Question):
    answer = ask(q.question, conn)
    return {"answer": answer}
