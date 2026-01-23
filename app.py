from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import make_demo_db, ask

app = FastAPI()

# ✅ CORS — OPEN AND CORRECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # IMPORTANT: allow all for now
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS
    allow_headers=["*"],
)

conn = make_demo_db()

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask_agent(q: Question):
    return {"answer": ask(q.question, conn)}
