#get context
#function that takes a message, call the current loaded agent. get the context from it in 5-10words max. 
#this context will be used to provide context to the user in the chat interface.

import logging
import anthropic
import google.generativeai as genai
from ollama import AsyncClient
from .config import AgentConfigModel, get_active_llm

logger = logging.getLogger("airavat")

async def get_context(message: str, config: AgentConfigModel) -> str:
    """
    Takes a message and generates a 5-10 word max context description.
    """
    prompt = f"Summarize the following message in 3 to 8 words to use as context. Reply ONLY with the context. Message: {message}"
    
    active = get_active_llm(config)
    kind = active["kind"]
    
    try:
        if kind == "anthropic":
            client = anthropic.AsyncAnthropic(api_key=config.llm.anthropic.apiKey)
            res = await client.messages.create(
                model=config.llm.anthropic.model,
                max_tokens=30,
                messages=[{"role": "user", "content": prompt}]
            )
            return res.content[0].text.strip().strip('"').strip("'")
            
        elif kind == "gemini":
            genai.configure(api_key=config.llm.gemini.apiKey) # type: ignore
            model = genai.GenerativeModel(model_name=config.llm.gemini.model) # type: ignore
            res = await model.generate_content_async(prompt)
            return res.text.strip().strip('"').strip("'")
            
        elif kind == "ollama":
            client = AsyncClient(host=config.llm.ollama.baseUrl)
            model_name = config.llm.ollama.model or (config.llm.ollama.models[0] if config.llm.ollama.models else "llama3")
            res = await client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            return res["message"]["content"].strip().strip('"').strip("'")
            
    except Exception as e:
        logger.error(f"Failed to generate UI context: {e}")
        return message[:40] + ("..." if len(message) > 40 else "")
        
    return message[:40] + ("..." if len(message) > 40 else "")