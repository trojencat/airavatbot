import json
import os
import sys
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Determine the root directory. If running as a PyInstaller bundle, sys._MEIPASS holds the temp dir.
# But for config files, we want the directory of the executable or script.
if getattr(sys, 'frozen', False):
    ROOT = os.path.dirname(sys.executable)
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MCP_CONFIG_PATH = os.path.join(ROOT, "mcp-config.json")
AGENT_CONFIG_PATH = os.path.join(ROOT, "agent-config.json")

# UI config path (theme, etc). This file used to live in the webui
# directory but was moved to the daemon so the server can serve it
# via the same settings API as the other configs.
UI_CONFIG_PATH = os.path.join(ROOT, "ui-config.json")

# ── MCP Server Config ─────────────────────────────────────────────

class MCPServerEntry(BaseModel):
    enabled: bool = True
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None

class MCPConfig(BaseModel):
    mcpServers: Dict[str, MCPServerEntry]

def load_mcp_config(path: Optional[str] = None) -> MCPConfig:
    file_path = path or MCP_CONFIG_PATH
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return MCPConfig(**data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return MCPConfig(mcpServers={})

def save_mcp_config(config: MCPConfig, path: Optional[str] = None) -> None:
    file_path = path or MCP_CONFIG_PATH
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=4) + "\n")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

# ── Agent Config ──────────────────────────────────────────────────

class AnthropicLLMConfig(BaseModel):
    model: str
    apiKey: Optional[str] = None

class GeminiLLMConfig(BaseModel):
    model: str
    apiKey: Optional[str] = None

class OllamaLLMConfig(BaseModel):
    model: Optional[str] = None
    baseUrl: str
    models: Optional[List[str]] = None

class LLMConfig(BaseModel):
    provider: str
    anthropic: AnthropicLLMConfig
    gemini: GeminiLLMConfig
    ollama: OllamaLLMConfig

class AgentSettings(BaseModel):
    systemPrompt: str
    maxToolRoundtrips: int

class AgentConfigModel(BaseModel):
    llm: LLMConfig
    agent: AgentSettings

def load_agent_config(path: Optional[str] = None) -> AgentConfigModel:
    file_path = path or AGENT_CONFIG_PATH
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return AgentConfigModel(**data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return a sensible default if file is missing/invalid
        return AgentConfigModel(
            llm=LLMConfig(
                provider="ollama",
                anthropic=AnthropicLLMConfig(model="claude-3-5-sonnet-20241022"),
                gemini=GeminiLLMConfig(model="gemini-2.5-pro"),
                ollama=OllamaLLMConfig(baseUrl="http://localhost:11434")
            ),
            agent=AgentSettings(systemPrompt="You are a helpful AI assistant.", maxToolRoundtrips=5)
        )

# ── UI Config ───────────────────────────────────────────────────

class UIConfig(BaseModel):
    theme: str = "light"


def load_ui_config(path: Optional[str] = None) -> UIConfig:
    file_path = path or UI_CONFIG_PATH
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return UIConfig(**data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # fallback to default configuration
        return UIConfig()


def save_ui_config(config: UIConfig, path: Optional[str] = None) -> None:
    file_path = path or UI_CONFIG_PATH
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=4, exclude_none=True) + "\n")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def save_agent_config(config: AgentConfigModel, path: Optional[str] = None) -> None:
    file_path = path or AGENT_CONFIG_PATH
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=4, exclude_none=True) + "\n")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def get_active_llm(config: AgentConfigModel) -> Dict[str, Any]:
    p = config.llm.provider
    if p == "anthropic":
        return {"kind": "anthropic", **config.llm.anthropic.model_dump()}
    elif p == "gemini":
        return {"kind": "gemini", **config.llm.gemini.model_dump()}
    elif p == "ollama":
        return {"kind": "ollama", **config.llm.ollama.model_dump()}
    else:
        raise ValueError(f"Unknown LLM provider: {p}")
