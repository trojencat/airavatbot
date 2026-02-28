import json
import time
from typing import List, Any

import anthropic
import google.generativeai as genai
from ollama import Client as OllamaClient

from .config import AgentConfigModel, get_active_llm
from .mcp_manager import MCPServerManager, AggregatedTool

# ── Types ─────────────────────────────────────────────────────────

class ToolCallInfo:
    def __init__(self, id: str, name: str, input: dict, result: Any = None):
        self.id = id
        self.name = name
        self.input = input
        self.result = result

class ChatResponse:
    def __init__(self, reply: str, toolCalls: List[ToolCallInfo]):
        self.reply = reply
        self.toolCalls = toolCalls
        
    def to_dict(self) -> dict:
        calls = []
        for t in self.toolCalls:
            c = {"id": t.id, "name": t.name, "input": t.input}
            if t.result is not None:
                c["result"] = t.result
            calls.append(c)
        return {"reply": self.reply, "toolCalls": calls}

# ── Anthropic Agent ───────────────────────────────────────────────

def mcp_tools_to_anthropic(tools: List[AggregatedTool]) -> List[dict]:
    # Need to massage JSON Schema types to what Anthropic strictly expects sometimes,
    # but passing inputSchema directly works most of the time
    return [
        {
            "name": t.name,
            "description": f"[{t.server_name}] {t.description}",
            "input_schema": t.input_schema,
        }
        for t in tools
    ]

async def run_anthropic_agent(
    message: str,
    history: List[dict],
    config: AgentConfigModel,
    manager: MCPServerManager,
) -> ChatResponse:
    llm = config.llm.anthropic
    if not llm.apiKey:
        raise ValueError("Missing API key for Anthropic. Please set it in the Settings page.")

    client = anthropic.AsyncAnthropic(api_key=llm.apiKey)
    tools = mcp_tools_to_anthropic(manager.get_all_tools())
    system_prompt = config.agent.systemPrompt
    max_roundtrips = config.agent.maxToolRoundtrips

    history.append({"role": "user", "content": message})

    all_tool_calls: List[ToolCallInfo] = []
    roundtrip = 0

    while roundtrip < max_roundtrips:
        roundtrip += 1

        api_args = {
            "model": llm.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": history,
        }
        if tools:
            api_args["tools"] = tools

        response = await client.messages.create(**api_args)

        text_parts = []
        tool_use_blocks = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        # Append assistant's response to history
        history.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})

        if not tool_use_blocks or response.stop_reason == "end_turn":
            return ChatResponse("\n".join(text_parts), all_tool_calls)

        tool_results = []

        for block in tool_use_blocks:
            tool_info = ToolCallInfo(block.id, block.name, block.input)

            try:
                result = await manager.call_tool(block.name, block.input)
                tool_info.result = result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            except Exception as e:
                err_msg = str(e)
                tool_info.result = {"error": err_msg}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {err_msg}",
                    "is_error": True,
                })

            all_tool_calls.append(tool_info)

        history.append({"role": "user", "content": tool_results})

    return ChatResponse("[Max tool roundtrips reached]", all_tool_calls)


# ── Gemini Agent ──────────────────────────────────────────────────

GEMINI_UNSUPPORTED_KEYS = {
    "$schema", "additionalProperties", "$id", "$ref",
    "$comment", "default", "examples", "readOnly", "writeOnly", "title",
    "minItems", "maxItems", "minLength", "maxLength", "pattern",
    "uniqueItems", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "const",
    "oneOf", "anyOf", "allOf", "not",
    "if", "then", "else",
    "patternProperties", "unevaluatedProperties", "prefixItems", "contains",
}

def sanitize_schema_for_gemini(obj: Any) -> Any:
    if isinstance(obj, list):
        return [sanitize_schema_for_gemini(item) for item in obj]
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k not in GEMINI_UNSUPPORTED_KEYS:
                clean[k] = sanitize_schema_for_gemini(v)
        # Ensure 'required' only references properties that actually exist
        if "required" in clean and "properties" in clean:
            valid_props = set(clean["properties"].keys())
            clean["required"] = [r for r in clean["required"] if r in valid_props]
            if not clean["required"]:
                del clean["required"]
        return clean
    return obj

def mcp_tools_to_gemini(tools: List[AggregatedTool]) -> List[dict]:
    if not tools:
        return []

    function_declarations = []
    for t in tools:
        params = sanitize_schema_for_gemini(t.input_schema)
        if "type" not in params:
            params["type"] = "object" # type: ignore
        # Convert JSON schema types to OpenAPI types for Gemini
        _replace_types_for_gemini(params)    
        
        function_declarations.append({
            "name": t.name,
            "description": f"[{t.server_name}] {t.description}",
            "parameters": params,
        })

    return [{"function_declarations": function_declarations}]

def _replace_types_for_gemini(schema: dict):
    if "type" in schema:
        t = schema["type"]
        type_mapping = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER", 
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT"
        }
        if t in type_mapping:
            schema["type"] = type_mapping[t]
            
    if "properties" in schema:
        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                _replace_types_for_gemini(prop)
    if "items" in schema and isinstance(schema["items"], dict):
        _replace_types_for_gemini(schema["items"])

async def run_gemini_agent(
    message: str,
    history: List[dict],
    config: AgentConfigModel,
    manager: MCPServerManager,
) -> ChatResponse:
    llm = config.llm.gemini
    if not llm.apiKey:
        raise ValueError("Missing API key for Gemini. Please set it in the Settings page.")

    genai.configure(api_key=llm.apiKey) # type: ignore
    tools = mcp_tools_to_gemini(manager.get_all_tools())
    system_prompt = config.agent.systemPrompt
    max_roundtrips = config.agent.maxToolRoundtrips

    model_kwargs: dict[str, Any] = {
        "model_name": llm.model,
        "system_instruction": system_prompt,
    }
    
    if tools:
        model_kwargs["tools"] = tools

    model = genai.GenerativeModel(**model_kwargs) # type: ignore

    history.append({"role": "user", "parts": [message]})
    all_tool_calls: List[ToolCallInfo] = []
    roundtrip = 0

    while roundtrip < max_roundtrips:
        roundtrip += 1

        # Use the google-generativeai chat interface
        # We need to reshape slightly because genai.start_chat takes standard formats
        chat = model.start_chat(history=history[:-1]) # type: ignore
        last_msg = history[-1]["parts"]
        
        try:
            response = await chat.send_message_async(last_msg)
        except Exception as e:
            return ChatResponse(f"[Gemini API Error: {e}]", all_tool_calls)

        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            return ChatResponse("[No response from Gemini]", all_tool_calls)

        text_parts = []
        function_calls = []

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
                # Convert the protocol buffer struct to a dict
                args = {}
                if part.function_call.args:
                    for k, v in part.function_call.args.items():
                        args[k] = v
                function_calls.append({
                    "name": part.function_call.name,
                    "args": args,
                })

        # The SDK's start_chat appends to its internal history, but we maintain ours manually for tool rounds
        history.append({
            "role": "model", 
            "parts": [p for p in candidate.content.parts]
        })

        if not function_calls:
            return ChatResponse("\n".join(text_parts), all_tool_calls)

        function_responses = []

        for fc in function_calls:
            import random
            call_id = f"gemini_{int(time.time())}_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))}"
            tool_info = ToolCallInfo(call_id, fc["name"], fc["args"])

            try:
                result = await manager.call_tool(fc["name"], fc["args"])
                tool_info.result = result
                function_responses.append({
                    "function_response": {
                        "name": fc["name"], 
                        "response": {"result": result}
                    }
                })
            except Exception as e:
                err_msg = str(e)
                tool_info.result = {"error": err_msg}
                function_responses.append({
                    "function_response": {
                        "name": fc["name"], 
                        "response": {"error": err_msg}
                    }
                })

            all_tool_calls.append(tool_info)

        # We must package the parts properly for the next round
        parts = []
        for r in function_responses:
            from google.ai.generativelanguage import Part, FunctionResponse
            from google.protobuf.struct_pb2 import Struct
            
            s = Struct()
            s.update(r["function_response"]["response"])
            
            parts.append(Part(function_response=FunctionResponse(
                name=r["function_response"]["name"],
                response=s
            )))
            
        history.append({"role": "user", "parts": parts})

    return ChatResponse("[Max tool roundtrips reached]", all_tool_calls)

# ── Ollama Agent ──────────────────────────────────────────────────

def mcp_tools_to_ollama(tools: List[AggregatedTool]) -> List[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": f"[{t.server_name}] {t.description}",
                "parameters": t.input_schema,
            }
        }
        for t in tools
    ]

async def run_ollama_agent(
    message: str,
    history: List[dict],
    config: AgentConfigModel,
    manager: MCPServerManager,
) -> ChatResponse:
    llm = config.llm.ollama
    from ollama import AsyncClient
    async_ollama = AsyncClient(host=llm.baseUrl)
    
    tools = mcp_tools_to_ollama(manager.get_all_tools())
    max_roundtrips = config.agent.maxToolRoundtrips

    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": config.agent.systemPrompt})

    history.append({"role": "user", "content": message})

    all_tool_calls: List[ToolCallInfo] = []
    roundtrip = 0

    while roundtrip < max_roundtrips:
        roundtrip += 1

        model_name = llm.model or (llm.models[0] if getattr(llm, 'models', None) else None) # type: ignore
        if not model_name:
            raise ValueError('No Ollama model specified in config')

        api_args = {
            "model": model_name,
            "messages": history,
        }
        if tools:
            api_args["tools"] = tools

        response = await async_ollama.chat(**api_args)
        assistant_msg = response["message"]

        tool_calls_formatted = []
        if assistant_msg.get("tool_calls"):
            for tc in assistant_msg["tool_calls"]:
                tool_calls_formatted.append({
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                })

        history_msg = {
            "role": "assistant",
            "content": assistant_msg.get("content") or ""
        }
        if tool_calls_formatted:
            history_msg["tool_calls"] = tool_calls_formatted
            
        history.append(history_msg)

        if not assistant_msg.get("tool_calls"):
            return ChatResponse(assistant_msg.get("content") or "", all_tool_calls)

        for tc in assistant_msg["tool_calls"]:
            import random
            call_id = f"ollama_{int(time.time())}_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))}"
            tool_name = tc["function"]["name"]
            tool_args = tc["function"]["arguments"]
            
            tool_info = ToolCallInfo(call_id, tool_name, tool_args)

            try:
                result = await manager.call_tool(tool_name, tool_args)
                tool_info.result = result
                history.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False)
                })
            except Exception as e:
                err_msg = str(e)
                tool_info.result = {"error": err_msg}
                history.append({
                    "role": "tool",
                    "content": f"Error: {err_msg}"
                })

            all_tool_calls.append(tool_info)

    return ChatResponse("[Max tool roundtrips reached]", all_tool_calls)


# ── ChatAgent (Unified) ──────────────────────────────────────────

class ChatAgent:
    def __init__(self, config: AgentConfigModel, manager: MCPServerManager):
        self.anthropic_history = []
        self.gemini_history = []
        self.ollama_history = []
        self.config = config
        self.manager = manager

    async def chat(self, message: str) -> dict:
        active = get_active_llm(self.config)
        kind = active["kind"]

        if kind == "anthropic":
            resp = await run_anthropic_agent(message, self.anthropic_history, self.config, self.manager)
        elif kind == "gemini":
            resp = await run_gemini_agent(message, self.gemini_history, self.config, self.manager)
        elif kind == "ollama":
            resp = await run_ollama_agent(message, self.ollama_history, self.config, self.manager)
        else:
            raise ValueError(f"Unknown LLM kind {kind}")
            
        return resp.to_dict()

    def clear_history(self):
        self.anthropic_history.clear()
        self.gemini_history.clear()
        self.ollama_history.clear()
