# CogniRoute 🧠⚡

A multi-agent semantic routing system with persistent vector memory, exposed via MCP (Model Context Protocol) for educational AI workflows.

## Overview

CogniRoute is an AI Infrastructure project designed to route user intents to specialized sub-agents cheaply, reliably, and locally. Instead of relying on expensive LLMs to classify intent, CogniRoute uses local embedding models to route requests in <50ms for $0

Once routed, the system queries a local vector database (ChromaDB) for past user context, dispatches to a specialized LLM sub-agent, and forces a strictly validated JSON response using Pydantic schemas.

Finally, the entire system is wrapped as an MCP Server, allowing it to be plugged directly into Claude Desktop or Cursor as a native tool.

## Architecture

1. **Semantic Router**: Uses `BAAI/bge-small-en-v1.5` to embed user queries and classify intent via cosine similarity against route centroids.
2. **Sub-Agents**: Calls `gemini-flash-latest` using Google's `genai` SDK, enforcing strict structured outputs (Quizzes, Explanations, Study Plans).
3. **Agent Memory**: Uses `ChromaDB` to store user queries. Implements multi-tenancy via `user_id` metadata filtering, enabling context-aware follow-ups (e.g., "Quiz me on that").
4. **MCP Server**: Exposes the pipeline as a tool via standard input/output (`stdio`) for integration with AI-native clients.

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI/ML**: `sentence-transformers`, `google-genai` (Gemini), Pydantic (Structured Outputs)
- **Database**: ChromaDB (Vector Store)
- **Protocol**: Model Context Protocol (MCP)

## Getting Started

### Prerequisites

- Python 3.12+
- `uv` package manager
- Google Gemini API Key

### Installation

1. Clone the repo:
```bash
   git clone https://github.com/abhiswrld/cogni-route.git
   cd cogni-route
```

2. Install dependencies:
```bash
   uv sync
```

3. Create a `.env` file in the root directory and add your API key:
   GEMINI_API_KEY=your_api_key_here

## Usage (MCP via Claude Desktop)

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cogniroute": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/cogni-route",
        "run",
        "python",
        "-m",
        "cogniroute.mcp_server"
      ]
    }
  }
}
```

Open Claude Desktop and try the prompts below:

## Usage (REST API)

Run the FastAPI server:

```bash
uv run uvicorn cogniroute.main:app --reload
```

Access the Swagger UI at `http://127.0.0.1:8000/docs` to test the `/process` endpoint.

## Example Prompts

1. **Study Plan**: "Use CogniRoute to create a 4-day study plan for my linear algebra midterm on matrices."
2. **Explain Concept**: "Use CogniRoute to explain how matrix multiplication works."
3. **Contextual Quiz**: "Use CogniRoute to quiz me on that." (Tests vector memory retrieval)