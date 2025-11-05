# Agentic Coder Platform

Autonomous coding agent system featuring persistent memory, semantic retrieval, and OpenRouter model orchestration. Built with FastAPI, a dark-themed control-room UI, and tooling designed for rapid production delivery.

## 🚀 Features

- **OpenRouter Integration** – auto-load available models, filter by free/paid tier, and route chat completions.
- **Semantic RAG Pipeline** – sentence-transformer embeddings with persistent vector store for contextual code recall.
- **Persistent Memory Layers** – `scratchpad.md` (short-term) and `main-plan.md` (long-term) synced with the UI.
- **Human-in-the-Loop Controls** – agent highlights steps that require confirmation before proceeding.
- **Dark Animated UI** – single-page console with live chat, memory editing, and ingestion workflow.

## 🧩 Project Structure

```
.
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── routers/api.py         # REST endpoints for chat, memory, ingestion
│   ├── services/              # Agent, OpenRouter client, semantic store
│   ├── memory/                # Memory manager
│   └── state/                 # Conversation persistence
├── static/index.html          # Dark-themed control UI
├── scratchpad.md              # Short-term memory store
├── main-plan.md               # Long-term plan store
├── vector_store/store.json    # Persistent embedding cache
└── requirements.txt
```

## 🛠️ Setup

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure OpenRouter credentials**
   ```bash
   export OPENROUTER_API_KEY=your_key
   export OPENROUTER_APP_URL=https://agentic-33774365.vercel.app
   export OPENROUTER_APP_NAME="Agentic Coder"
   ```
   > Without an API key the app falls back to mock completions and sample models.

3. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Open the console**
   Visit `http://localhost:8000` to operate the agent.

## 📦 Deployment

Deploy to Vercel using the provided command once the build passes locally:
```bash
vercel deploy --prod --yes --token $VERCEL_TOKEN --name agentic-33774365
```

## 🔍 Extending the Agent

- Add structured tools in `app/services/agent.py` to call external APIs or run local scripts.
- Preload domain documents via `POST /api/ingest`.
- Customize the system prompt or response schema in `app/services/agent.py`.

## ✅ Health Check

- `GET /api/health` – confirms backend status.
- Mock mode (no API key) keeps the UI usable for demos and testing.

---

Craft remarkable software faster with a coder who never forgets the plan. Adjust the memories, curate context, and iterate with confidence.
