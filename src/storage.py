import json
import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Determine the root directory. If running as a PyInstaller bundle, sys._MEIPASS holds the temp dir.
# But for data files, we want the directory of the executable or script.
import sys
if getattr(sys, 'frozen', False):
    ROOT = os.path.dirname(sys.executable)
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data", "conversations")

class ToolCallInfoModel(BaseModel):
    id: str
    name: str
    input: dict
    result: Optional[Any] = None

class Message(BaseModel):
    id: str
    role: str # "user", "assistant", "system", etc.
    message: str # The string to be returned to the UI
    context: Any # The actual raw context objects required by the specific LLM API (like Anthropic blocks)
    tool_calls: Optional[List[ToolCallInfoModel]] = None

class Conversation(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    messages: List[Message] = Field(default_factory=list)

class Storage:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    def _get_path(self, conversation_id: str) -> str:
        # Sanitize filename to prevent traversal
        safe_id = os.path.basename(conversation_id)
        if not safe_id.endswith(".json"):
            safe_id += ".json"
        return os.path.join(DATA_DIR, safe_id)

    def save_conversation(self, conversation: Conversation):
        conversation.updated_at = time.time()
        path = self._get_path(conversation.id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(conversation.model_dump_json(indent=2))

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        path = self._get_path(conversation_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return Conversation(**data)
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return None

    def list_conversations(self) -> List[Dict[str, Any]]:
        conversations = []
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".json"):
                path = os.path.join(DATA_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        conversations.append({
                            "id": data.get("id"),
                            "title": data.get("title", "New Chat"),
                            "updated_at": data.get("updated_at", 0)
                        })
                except Exception:
                    pass
        # Sort by most recent first
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        path = self._get_path(conversation_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
