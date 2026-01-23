from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import make_demo_db, ask

app = FastAPI()

# ✅ ENABLE CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
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
    answer = ask(q.question, conn)
    return {"answer": answer}
