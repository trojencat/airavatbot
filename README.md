# ⚡ Airavat — Daemon

FastAPI server for the Airavat MCP Agent. Handles LLM communication, MCP server management, and configuration.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running

```bash
# Development (with hot reload)
uvicorn src.server:app --reload --port 8920

# Production
uvicorn src.server:app --host 0.0.0.0 --port 8920
```

The API server runs on `http://localhost:8920`.

## Configuration

### agent-config.json

LLM provider and agent settings:

```json
{
  "llm": {
    "provider": "ollama",
    "anthropic": { "model": "...", "apiKey": "..." },
    "gemini": { "model": "...", "apiKey": "..." },
    "ollama": { "baseUrl": "http://localhost:11434", "model": "llama3.2:3b" }
  },
  "agent": {
    "systemPrompt": "...",
    "maxToolRoundtrips": 10
  }
}
```

### mcp-config.json

MCP server connections:

```json
{
  "mcpServers": {
    "browser": { "enabled": true, "command": "npx", "args": [...] },
    "filesystem": { "enabled": true, "command": "npx", "args": [...] }
  }
}
```

### ui-config.json

Theme and other UI preferences. This file is now stored in the daemon root and
served over the `/api/settings/ui` endpoint so that the web UI can fetch/write
it just like the other configs:

```json
{
  "theme": "dark"          # or "light"
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send a chat message |
| POST | `/api/clear` | Clear chat history |
| GET | `/api/servers` | Get LLM & MCP server status |
| GET | `/api/settings/agent` | Get agent config |
| PATCH | `/api/settings/agent/llm` | Switch LLM provider/model |
| GET | `/api/settings/mcp` | Get MCP server list |
| PATCH | `/api/settings/mcp/:name` | Toggle MCP server |
| GET | `/api/settings/ui` | Get UI config |
| POST | `/api/settings/ui` | Save UI config |
| GET | `/api/ollama/models` | List Ollama models |
| GET | `/api/models/anthropic` | List Anthropic models |
| GET | `/api/models/gemini` | List Gemini models |
| GET/POST | `/api/settings/agent/apikey` | Manage API keys |

## Docker

```bash
# Build (run from project root)
docker build -f airavat-daemon/Dockerfile .

# Run
docker run -p 8920:8920 airavat-daemon:latest
```

## Project Structure

```
airavat-daemon/
├── src/
│   ├── server.py          # FastAPI server & API routes
│   ├── agent.py           # LLM agent logic
│   ├── config.py          # Config file handling
│   └── mcp_manager.py     # MCP server management
├── agent-config.json
├── mcp-config.json
├── ui-config.json
├── pyproject.toml
└── Dockerfile
```
