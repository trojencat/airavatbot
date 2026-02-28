import os
import asyncio
from typing import Dict, List, Any

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import mcp.types as mcp_types
from .config import MCPConfig, MCPServerEntry


async def _list_roots_handler(context: Any) -> mcp_types.ListRootsResult:
    """Return an empty roots list so servers don't error out."""
    return mcp_types.ListRootsResult(roots=[])


class AggregatedTool:
    def __init__(self, name: str, original_name: str, server_name: str, description: str, input_schema: dict):
        self.name = name  # Namespaced name
        self.original_name = original_name
        self.server_name = server_name
        self.description = description or ""
        self.input_schema = input_schema or {}

class ConnectedServer:
    def __init__(self, session: ClientSession, stack: 'AsyncExitStack', tools: List[AggregatedTool]):
        self.session = session
        self.stack = stack
        self.tools = tools

class MCPServerManager:
    def __init__(self):
        self.servers: Dict[str, ConnectedServer] = {}

    async def connect_all(self, config: MCPConfig):
        enabled_servers = [(name, entry) for name, entry in config.mcpServers.items() if entry.enabled]
        disabled_servers = [(name, entry) for name, entry in config.mcpServers.items() if not entry.enabled]

        print(f"🔌 Connecting to {len(enabled_servers)} MCP server(s) ({len(disabled_servers)} disabled)...")

        for name, _ in disabled_servers:
            print(f"   ⏸️  {name} — disabled")

        for name, entry in enabled_servers:
            try:
                await self.connect_one(name, entry)
                tool_count = len(self.servers[name].tools)
                print(f"   ✅ {name} — {tool_count} tool(s)")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")

    async def connect_one(self, name: str, entry: MCPServerEntry):
        if name in self.servers:
            await self.disconnect_one(name)

        server_params = StdioServerParameters(
            command=entry.command,
            args=entry.args,
            env={**os.environ, **(entry.env or {})}
        )

        from contextlib import AsyncExitStack
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream, list_roots_callback=_list_roots_handler)
            )
            
            await session.initialize()
                
            tools_response = await session.list_tools()
            tools: List[AggregatedTool] = []
            
            for t in tools_response.tools:
                tools.append(AggregatedTool(
                    name=f"{name}__{t.name}",
                    original_name=t.name,
                    server_name=name,
                    description=t.description,
                    input_schema=t.inputSchema
                ))
            
            self.servers[name] = ConnectedServer(session, stack, tools)
        except Exception as e:
            await stack.aclose()
            raise e

    async def disconnect_one(self, name: str):
        server = self.servers.get(name)
        if not server:
            return
        try:
            await server.stack.aclose()
        except Exception:
            pass
        del self.servers[name]

    def is_connected(self, name: str) -> bool:
        return name in self.servers

    def get_all_tools(self) -> List[AggregatedTool]:
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.tools)
        return all_tools

    def get_server_status(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "toolCount": len(srv.tools),
                "tools": [t.original_name for t in srv.tools]
            }
            for name, srv in self.servers.items()
        ]

    async def call_tool(self, namespaced_name: str, args: Dict[str, Any]) -> Any:
        sep = namespaced_name.find("__")
        if sep == -1:
            raise ValueError(f"Invalid namespaced tool name: {namespaced_name}")

        server_name = namespaced_name[:sep]
        tool_name = namespaced_name[sep + 2:]

        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not connected")

        # The MCP Python SDK expects arguments to be unwrapped.
        result = await server.session.call_tool(tool_name, arguments=args)
        
        # Format the result similarly to how standard JS objects are serialized
        # Convert any Text/Image contents down to generic dictionary format
        formatted_result: Dict[str, Any] = {"content": []}
        if result and hasattr(result, "content"):
            for item in result.content:
                if item.type == "text":
                    formatted_result["content"].append({"type": "text", "text": item.text})
                elif item.type == "image":
                    formatted_result["content"].append({"type": "image", "data": item.data, "mimeType": getattr(item, 'mimeType', 'unknown')})
                else:
                    formatted_result["content"].append({"type": item.type})
            
            if hasattr(result, "isError"):
                formatted_result["isError"] = result.isError
                
        return formatted_result or result

    async def disconnect_all(self):
        for name, server in list(self.servers.items()):
            try:
                await server.stack.aclose()
                print(f"   🔌 Disconnected: {name}")
            except Exception:
                pass
        self.servers.clear()
