# Premier League Question Answering Agent

This project is a deterministic natural-language question answering system for Premier League player data.  
It combines Azure OpenAI for structured intent extraction with a rule-based execution engine to ensure factual and reliable answers.

The system is designed to avoid hallucinations by separating language understanding from data retrieval.

---

## Architecture Overview

The application follows a strict multi-step pipeline:

1. **Intent Parsing**  
   Azure OpenAI converts a user question into a constrained JSON intent based on a predefined schema.

2. **Deterministic Execution**  
   The intent is executed against a local SQLite database containing Premier League player data.

3. **Answer Formulation**  
   A second Azure OpenAI call rewrites the deterministic result into a concise natural-language response without adding new facts.

This approach ensures that all answers are grounded in data and reproducible.

---

## Tech Stack

- Python
- FastAPI
- SQLite (in-memory)
- Azure OpenAI
- Render (deployment)

---

## Live Demo

- **Backend API**: https://premier-league-agent.onrender.com  
- **Frontend Demo**: https://fadimbarki87.github.io/premier-league-website/

---

## Running Locally

```bash
pip install -r requirements.txt

export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_DEPLOYMENT=...
export AZURE_OPENAI_API_VERSION=...
export AZURE_OPENAI_API_KEY=...

uvicorn app:app --reload
