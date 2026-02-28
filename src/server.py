import os
import sys
import json
import logging
import httpx
from contextlib import asynccontextmanager
from typing import Any, Optional, List, Dict

from fastapi import FastAPI, HTTPException, Request, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from ollama import Client as OllamaClient
import anthropic

from .config import (
    load_mcp_config, save_mcp_config,
    load_agent_config, save_agent_config, get_active_llm,
    UIConfig, load_ui_config, save_ui_config
)
from .mcp_manager import MCPServerManager
from .agent import ChatAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("airavat")

PORT = int(os.environ.get("PORT", "8920"))

# WebUI dist directory
def get_webui_dist_path() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), "webui")
    else:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui")

# ── WebSocket Event Bus ───────────────────────────────────────

class ConfigEventBus:
    """Manages WebSocket clients and broadcasts config change events."""

    def __init__(self):
        self._clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)
        logger.info(f"   🔗 WS client connected ({len(self._clients)} total)")

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)
        logger.info(f"   🔌 WS client disconnected ({len(self._clients)} total)")

    async def broadcast(self, event_type: str, data: Any = None):
        payload = json.dumps({"type": event_type, "data": data})
        stale: List[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

event_bus = ConfigEventBus()

# Globals initialized at startup
manager: 'MCPServerManager' = None # type: ignore
agent: 'ChatAgent' = None # type: ignore
mcp_config_cache: Any = None
agent_config_cache: Any = None
active_llm_cache: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, agent, mcp_config_cache, agent_config_cache, active_llm_cache

    try:
        # Load configs
        mcp_config_cache = load_mcp_config()
        agent_config_cache = load_agent_config()

        # Sync Ollama models
        ollama_url = agent_config_cache.llm.ollama.baseUrl
        try:
            ollama = OllamaClient(host=ollama_url)
            response = ollama.list()
            model_names = [m.model for m in response.models]
            agent_config_cache.llm.ollama.models = model_names
            
            if not agent_config_cache.llm.ollama.model or agent_config_cache.llm.ollama.model not in model_names:
                if model_names:
                    agent_config_cache.llm.ollama.model = model_names[0]
            
            save_agent_config(agent_config_cache)
            logger.info(f"   🔄 Ollama models synced: {model_names}")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to sync Ollama models: {e}")

        active_llm_cache = get_active_llm(agent_config_cache)
        kind = active_llm_cache["kind"]
        model_display = active_llm_cache.get("model", "")
        if not model_display and active_llm_cache.get("models"):
            model_display = ", ".join(active_llm_cache["models"])

        logger.info("\n⚡ Airavat — MCP Agent")
        logger.info(f"   LLM provider : {kind}")
        logger.info(f"   Model        : {model_display}\n")

        # Connect MCP servers
        manager = MCPServerManager()
        await manager.connect_all(mcp_config_cache)

        # Create Agent
        agent = ChatAgent(agent_config_cache, manager)

        #Setup WebUI
        _setup_webui(app)
        # once the SPA routes are wired up we can log so users know the
        # front‑end will be served from the root path
        logger.info(f"📄 WebUI available at http://localhost:{PORT}")

        yield
        
    finally:
        logger.info("\n🛑 Shutting down...")
        if manager:
            await manager.disconnect_all()


app = FastAPI(title="Airavat Daemon", lifespan=lifespan)

# ── WebUI Static Files & SPA Routing ──────────────────────────

def _setup_webui(app: FastAPI):
    """Configure WebUI static file serving and SPA routing."""
    webui_dist = get_webui_dist_path()
    
    # Check if webui directory exists
    if not os.path.isdir(webui_dist):
        logger.warning(f"⚠️  WebUI dist directory not found at {webui_dist}")
        return
    
    # Mount static files (assets directory)
    assets_dir = os.path.join(webui_dist, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir, html=False), name="assets")
    
    # Mount root static files (favicon, logo, etc.)
    app.mount("/static", StaticFiles(directory=webui_dist, html=False), name="static")
    
    # SPA fallback routes for both root and /webui paths
    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def serve_root():
        """Serve index.html at root path."""
        index = os.path.join(webui_dist, "index.html")
        if os.path.isfile(index):
            return FileResponse(index, media_type="text/html", headers={
                "Cache-Control": "max-age=0, no-cache, no-store, must-revalidate"
            })
        return JSONResponse(status_code=404, content={"error": "WebUI not found"})
    
    @app.get("/{full_path:path}", response_class=FileResponse, include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve WebUI SPA with proper fallback and cache headers."""
        # Skip API and WebSocket routes
        if full_path.startswith(("api/", "ws/", "assets/", "static/")):
            return JSONResponse(status_code=404, content={"error": "Not found"})
        
        # Try to serve the requested file
        file_path = os.path.join(webui_dist, full_path)
        if os.path.isfile(file_path):
            # Asset files get longer cache
            if full_path.startswith(("assets/", "static/")):
                return FileResponse(file_path, headers={
                    "Cache-Control": "public, max-age=31536000, immutable"
                })
            else:
                return FileResponse(file_path)
        
        # Fallback to index.html for SPA routing
        index = os.path.join(webui_dist, "index.html")
        if os.path.isfile(index):
            return FileResponse(index, media_type="text/html", headers={
                "Cache-Control": "max-age=0, no-cache, no-store, must-revalidate"
            })
        
        return JSONResponse(status_code=404, content={"error": "WebUI not found"})
    


# ── WebSocket endpoint ────────────────────────────────────────

@app.websocket("/ws/config")
async def ws_config(ws: WebSocket):
    await event_bus.connect(ws)
    try:
        while True:
            # Keep connection alive; client can send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        event_bus.disconnect(ws)
    except Exception:
        event_bus.disconnect(ws)

# ── Chat API ──────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    message = body.get("message")
    if not message or not isinstance(message, str):
        raise HTTPException(status_code=400, detail="message is required and must be a string")

    try:
        result = await agent.chat(message)
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/servers")
def get_servers():
    model_display = active_llm_cache.get("model", "")
    if not model_display and active_llm_cache.get("models"):
        model_display = ", ".join(active_llm_cache["models"])
        
    return {
        "llm": {"provider": active_llm_cache["kind"], "model": model_display},
        "servers": manager.get_server_status(),
    }

@app.post("/api/clear")
def clear_history():
    agent.clear_history()
    return {"ok": True}

# ── Settings API: MCP Config ──────────────────────────────────

@app.get("/api/settings/mcp")
def get_mcp_settings():
    global mcp_config_cache
    mcp_config_cache = load_mcp_config()
    servers = []
    for name, entry in mcp_config_cache.mcpServers.items():
        servers.append({
            "name": name,
            "enabled": entry.enabled,
            "command": entry.command,
            "args": entry.args,
            "connected": manager.is_connected(name),
        })
    return {"servers": servers}

@app.patch("/api/settings/mcp/{name}")
async def toggle_mcp_server(name: str, enabled: bool = Body(..., embed=True)):
    global mcp_config_cache
    mcp_config_cache = load_mcp_config()
    
    if name not in mcp_config_cache.mcpServers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found in config")
        
    entry = mcp_config_cache.mcpServers[name]
    entry.enabled = enabled
    save_mcp_config(mcp_config_cache)

    try:
        if enabled:
            await manager.connect_one(name, entry)
            logger.info(f"   ✅ {name} enabled & connected")
        else:
            await manager.disconnect_one(name)
            logger.info(f"   ⏸️  {name} disabled & disconnected")
            
        result = {"ok": True, "connected": manager.is_connected(name)}
        await event_bus.broadcast("mcp_server_toggled", {"name": name, "enabled": enabled, "connected": manager.is_connected(name)})
        return result
    except Exception as e:
        logger.error(f"   ❌ {name} toggle error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.patch("/api/settings/mcp/{name}/args")
async def update_mcp_server_args(name: str, request: Request):
    body = await request.json()
    add_args = body.get("add", [])
    remove_args = body.get("remove", [])
    
    global mcp_config_cache
    mcp_config_cache = load_mcp_config()
    
    if name not in mcp_config_cache.mcpServers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found in config")
        
    entry = mcp_config_cache.mcpServers[name]
    
    if remove_args:
        entry.args = [a for a in entry.args if a not in remove_args]
    if add_args:
        for a in add_args:
            if a not in entry.args:
                entry.args.append(a)
                
    save_mcp_config(mcp_config_cache)
    
    if entry.enabled:
        try:
            await manager.disconnect_one(name)
            await manager.connect_one(name, entry)
            logger.info(f"   🔄 {name} args updated & reconnected")
        except Exception as e:
            logger.error(f"   ❌ {name} reconnect error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        logger.info(f"   🔄 {name} args updated (server disabled)")

    result = {"ok": True, "args": entry.args, "connected": manager.is_connected(name)}
    await event_bus.broadcast("mcp_server_args_updated", {"name": name, "args": entry.args, "connected": manager.is_connected(name)})
    return result

# ── Settings API: Agent Config ────────────────────────────────

@app.get("/api/settings/agent")
def get_agent_settings():
    global agent_config_cache, active_llm_cache
    agent_config_cache = load_agent_config()
    active_llm_cache = get_active_llm(agent_config_cache)
    # The frontend expects standard json, so we dump to dict first
    return agent_config_cache.model_dump()

@app.patch("/api/settings/agent/llm")
async def update_agent_llm(request: Request, provider: str = Body(...), model: str = Body(None)):
    valid_providers = ["anthropic", "gemini", "ollama"]
    if provider not in valid_providers:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
        
    global agent_config_cache, active_llm_cache, agent
    agent_config_cache = load_agent_config()
    agent_config_cache.llm.provider = provider
    
    if model and isinstance(model, str):
        if provider == "ollama":
            agent_config_cache.llm.ollama.model = model
        elif provider == "anthropic":
            agent_config_cache.llm.anthropic.model = model
        elif provider == "gemini":
            agent_config_cache.llm.gemini.model = model
    elif provider == "ollama" and not agent_config_cache.llm.ollama.model and agent_config_cache.llm.ollama.models:
        agent_config_cache.llm.ollama.model = agent_config_cache.llm.ollama.models[0]
        
    save_agent_config(agent_config_cache)
    active_llm_cache = get_active_llm(agent_config_cache)
    
    agent = ChatAgent(agent_config_cache, manager)
    
    model_display = active_llm_cache.get("model", "")
    if not model_display and active_llm_cache.get("models"):
        model_display = ", ".join(active_llm_cache["models"])
        
    logger.info(f"   🔄 LLM switched to: {provider} / {model_display}")
    result = {"ok": True, "provider": provider, "model": model_display}
    await event_bus.broadcast("llm_changed", {"provider": provider, "model": model_display})
    return result

@app.get("/api/settings/agent/apikey/{provider}")
def get_api_key(provider: str):
    valid_providers = ["anthropic", "gemini"]
    if provider not in valid_providers:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
        
    global agent_config_cache
    agent_config_cache = load_agent_config()
    
    provider_config = getattr(agent_config_cache.llm, provider)
    api_key = getattr(provider_config, "apiKey", "")
    
    return {"provider": provider, "apiKey": api_key or ""}

@app.post("/api/settings/agent/apikey")
async def save_api_key(provider: str = Body(...), apiKey: str = Body(...)):
    valid_providers = ["anthropic", "gemini"]
    if provider not in valid_providers:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
        
    if not isinstance(apiKey, str):
        raise HTTPException(status_code=400, detail="apiKey is required and must be a string")
        
    global agent_config_cache, agent
    agent_config_cache = load_agent_config()
    
    provider_config = getattr(agent_config_cache.llm, provider)
    provider_config.apiKey = apiKey
    
    save_agent_config(agent_config_cache)
    agent = ChatAgent(agent_config_cache, manager)
    
    logger.info(f"   🔐 API key saved for {provider}")
    await event_bus.broadcast("apikey_saved", {"provider": provider})
    return {"ok": True, "provider": provider}


# ── Models API ────────────────────────────────────────────────

@app.get("/api/ollama/models")
def get_ollama_models():
    try:
        ollama_url = agent_config_cache.llm.ollama.baseUrl
        ollama = OllamaClient(host=ollama_url)
        response = ollama.list()
        
        models = [
            {"name": m.model, "size": getattr(m, 'size', getattr(m, 'details', {}).get('parameter_size', 0)), "modified_at": getattr(m, 'modified_at', '')}
            for m in response.models
        ]
        return {"models": models}
    except Exception as e:
        logger.error(f"   ❌ Failed to list Ollama models: {e}")
        return JSONResponse(status_code=500, content={"error": f"Could not reach Ollama: {e}"})

@app.get("/api/models/anthropic")
def get_anthropic_models():
    try:
        global agent_config_cache
        agent_config_cache = load_agent_config()
        api_key = agent_config_cache.llm.anthropic.apiKey
        
        if not api_key:
            return JSONResponse(status_code=400, content={"error": "Anthropic API key not set"})
            
        client = anthropic.Anthropic(api_key=api_key)
        # Using pagination with limit 100 as in the original code
        response = client.models.list(limit=100)
        
        models = [
            {"id": m.id, "name": getattr(m, 'display_name', m.id) or m.id}
            for m in response.data if m.type == "model"
        ]
        models.sort(key=lambda x: x["name"])
        return {"models": models}
    except Exception as e:
        logger.error(f"   ❌ Failed to list Anthropic models: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/models/gemini")
async def get_gemini_models():
    try:
        global agent_config_cache
        agent_config_cache = load_agent_config()
        api_key = agent_config_cache.llm.gemini.apiKey
        
        if not api_key:
            return JSONResponse(status_code=400, content={"error": "Gemini API key not set"})
            
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100")
            
            if resp.status_code != 200:
                err = resp.json()
                raise Exception(err.get("error", {}).get("message", f"API error: {resp.status_code}"))
                
            data = resp.json()
            models = []
            
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    # Clean the models/ prefix
                    name_id = m["name"].replace("models/", "")
                    display = m.get("displayName", name_id)
                    models.append({"id": name_id, "name": display})
                    
            models.sort(key=lambda x: x["name"])
            return {"models": models}
    except Exception as e:
        logger.error(f"   ❌ Failed to list Gemini models: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── UI Config API ─────────────────────────────────────────────

# the actual logic for reading/writing the configuration lives in
# `config.py` alongside the other settings. we keep the routes here
# for HTTP access and emit a websocket event when the config changes.

@app.get("/api/settings/ui")
def get_ui_config():
    try:
        cfg = load_ui_config()
        return cfg.model_dump()
    except Exception as e:
        logger.error(f"   ❌ Failed to load UI config: {e}")
        return JSONResponse(status_code=500, content={"error": "Could not read UI config"})

@app.post("/api/settings/ui")
async def save_ui_config_route(request: Request):
    try:
        body = await request.json()
        cfg = UIConfig(**body)
        save_ui_config(cfg)
        await event_bus.broadcast("ui_config_changed", cfg.model_dump())
        return {"ok": True}
    except Exception as e:
        logger.error(f"   ❌ Failed to save UI config: {e}")
        return JSONResponse(status_code=500, content={"error": "Could not write UI config"})



# ── Server Startup ────────────────────────────────────────────

def main():
    uvicorn.run("src.server:app", host="0.0.0.0", port=PORT, access_log=False , log_level="critical")

if __name__ == "__main__":
    main()
