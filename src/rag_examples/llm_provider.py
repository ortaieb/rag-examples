import os
from anthropic import Anthropic, AsyncAnthropic
from typing import Any

class ClaudeLLMProvider:
    def __init__(self, model: str = None, api_key: str = None, temperature: float = 0.0):
        self.model = model or os.getenv("CLAUDE_MODEL")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        # Claude's API returns a list of content blocks; join them if needed
        return "".join(block.text for block in response.content)
