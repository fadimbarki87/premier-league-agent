from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from agent import make_demo_db, Agent

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
try:
    conn = make_demo_db()
    print("Database initialized successfully")
except Exception:
    print("DATABASE INIT FAILED")
    traceback.print_exc()
    raise

# In-memory session store: session_id -> Agent
agents: dict[str, Agent] = {}


class Question(BaseModel):
    session_id: str
    question: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/ask")
def ask_agent(q: Question):
    try:
        sid = q.session_id

        # One Agent per session
        if sid not in agents:
            agents[sid] = Agent()

        agent = agents[sid]
        answer = agent.ask(q.question, conn)

        return {"answer": answer}

    except Exception:
        traceback.print_exc()
        return {"answer": "Internal server error. Check logs."}


@app.post("/reset")
def reset_agent(session_id: str):
    """
    Reset ONLY this user's conversation state.
    """
    agents.pop(session_id, None)
    return {"status": "reset"}
