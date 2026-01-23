from fastapi import FastAPI
from pydantic import BaseModel
from agent import make_demo_db, ask

app = FastAPI()
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
