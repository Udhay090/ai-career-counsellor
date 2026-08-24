# AI Career Counselor

Multi-agent career guidance app built with CrewAI and Gradio.

The pipeline runs four specialized agents in sequence:

1. **Profile Analyzer** — extracts skills, education, interests, goals, and constraints.
2. **Skill Gap Analyzer** — identifies missing skills for the target career.
3. **Course Recommender** — maps gaps to practical learning resources.
4. **Career Advisor** — combines everything into a concise action plan.

## Architecture

```text
User → Profile Analyzer → Skill Gap Analyzer → Course Recommender → Career Advisor → Final answer
```

Each task receives earlier task outputs through CrewAI `context=[...]`. Recent conversation turns are kept per browser session with `gr.State` and persisted in browser `localStorage`.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio UI, session history, LLM configuration, Render entry point |
| `crew.py` | CrewAI agents, tasks, and sequential orchestration |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `.python-version` | Pins Python 3.13 for CrewAI compatibility |
| `architecture.mmd` | Mermaid architecture diagram |
| `sample_outputs.md` | Template for recording real demo outputs |

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Default model configuration:

```text
MODEL=groq/openai/gpt-oss-120b
GROQ_API_KEY=your_groq_api_key
```

You can switch providers by changing `MODEL` and supplying that provider's API key.

## Render deployment

Create a **Web Service** from this GitHub repository and use:

```text
Build Command: pip install -r requirements.txt
Start Command: python app.py
```

Add this environment variable in Render:

```text
GROQ_API_KEY=your_groq_api_key
```

`MODEL` is optional because the app defaults to `groq/openai/gpt-oss-120b`.

## Conversation memory

The app does not keep user chat history in a global Python dictionary. `gr.State` gives each browser session its own state, and the latest six complete turns are passed into the Profile Analyzer. Browser `localStorage` restores that history after a refresh.

## Demo checklist

- Ask for a career recommendation.
- Follow up with a constraint such as `make it low-cost` to demonstrate memory.
- Record 1–2 actual conversations in `sample_outputs.md`.
- Record the required demo video if this is being submitted as an assignment.
