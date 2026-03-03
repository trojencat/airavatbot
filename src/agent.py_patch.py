import uuid
import time
from typing import List, Callable, Awaitable, Optional, Dict, Any

from .storage import Storage, Conversation, Message, ToolCallInfoModel
from .config import AgentConfigModel, get_active_llm
from .mcp_manager import MCPServerManager

# Use the ChatAgent class as our base of modification, but instead we rewrite it now via multi_replace
