from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from agent import make_demo_db, ask

app = FastAPI()

# CORS (correct)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB safely
try:
    conn = make_demo_db()
    print("Database initialized successfully")
except Exception as e:
    print("DATABASE INIT FAILED")
    traceback.print_exc()
    raise e

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask_agent(q: Question):
    try:
        answer = ask(q.question, conn)
        return {"answer": answer}
    except Exception:
        traceback.print_exc()
        return {"answer": "Internal server error. Check logs."}
