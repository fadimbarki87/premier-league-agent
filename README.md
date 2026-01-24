# Premier League Question Answering Agent

This project is a deterministic natural-language question answering system for Premier League player data.
It combines Azure OpenAI for structured intent extraction with a rule-based execution engine to ensure factual, reproducible answers.

The system is designed to avoid hallucinations by separating language understanding from data retrieval.

---

## Architecture Overview

The application follows a strict multi-step pipeline:

### Intent Parsing
Azure OpenAI converts a user question into a constrained JSON intent based on a predefined schema.

### Deterministic Execution
The intent is executed against a local SQLite database containing Premier League player data loaded from a CSV file.

### Answer Formulation
A second Azure OpenAI call rewrites the deterministic result into a concise natural-language response without adding new facts.

---

## Tech Stack

- Python 3.10+
- FastAPI
- SQLite (in-memory)
- Azure OpenAI
- Render (deployment)

---

## Live Demo

- Backend API: https://premier-league-agent.onrender.com  
- Frontend Demo: https://fadimbarki87.github.io/premier-league-website/

---

## Running Locally

### Prerequisites

- Python 3.10 or newer
- An Azure OpenAI resource with a deployed chat model

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables (replace values with your real Azure OpenAI values):

```bash
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME
export AZURE_OPENAI_API_VERSION=2025-01-01-preview
export AZURE_OPENAI_API_KEY=YOUR_API_KEY
```

The application loads player data from `premier_league_players.csv` at startup and initializes an in-memory SQLite database.

### Start the API

```bash
uvicorn app:app --reload
```

The API will be available at:

```
http://localhost:8000
```

---

## API Usage

Send a POST request to `/ask`:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Which religion does Salah have?"}'
```

---

## Verification

If the setup is correct, the terminal logs will show:

```
Database initialized successfully
AZURE PARSER STATUS: 200
AZURE NLG STATUS: 200
```

This confirms that environment variables are loaded correctly and Azure OpenAI is being used.

---

## Notes

- API keys are read exclusively from environment variables.
- No secrets are stored in the repository.
- The database is recreated in memory on each startup.
