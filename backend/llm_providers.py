import httpx
import numpy as np
from lightrag.llm.openai import openai_complete_if_cache


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        llm_model: str,
        vision_model: str,
        embed_model: str,
        num_ctx: int,
        timeout: int,
    ):
        self.base_url = base_url
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.embed_model = embed_model
        self.num_ctx = num_ctx
        self.timeout = timeout

    # LLM model
    async def llm(
        self, prompt: str, system_prompt=None, history_messages=None, **kwargs
    ):
        history_messages = history_messages or []
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": self.num_ctx},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    # Vision model
    async def vision(
        self,
        prompt: str,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if not self.vision_model:
            return await self.llm(prompt, system_prompt, history_messages, **kwargs)

        # Handle native Multimodal Query format
        if messages:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.vision_model,
                        "messages": messages,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        # Handle standard image processing
        if image_data:
            built = []
            if system_prompt:
                built.append({"role": "system", "content": system_prompt})
            built.append({"role": "user", "content": prompt, "images": [image_data]})
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.vision_model,
                        "messages": built,
                        "stream": False,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]

        return await self.llm(prompt, system_prompt, history_messages, **kwargs)

    # Embedding model
    async def embed(self, texts: list[str]) -> np.ndarray:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": texts},
            )
            resp.raise_for_status()
            return np.array(resp.json()["embeddings"])


class OpenAIProvider:
    """Supports standard OpenAI, as well as vLLM, llama.cpp, Groq, etc. via custom base_url"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        llm_model: str,
        vision_model: str,
        embed_model: str,
        timeout: int,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.embed_model = embed_model
        self.timeout = timeout

    # LLM model
    async def llm(
        self, prompt: str, system_prompt=None, history_messages=None, **kwargs
    ):
        return await openai_complete_if_cache(
            self.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )

    # Vision model
    async def vision(
        self,
        prompt: str,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if not self.vision_model:
            return await self.llm(prompt, system_prompt, history_messages, **kwargs)

        # Handle native Multimodal Query format
        if messages:
            return await openai_complete_if_cache(
                self.vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        # Handle standard image processing
        if image_data:
            built = []
            if system_prompt:
                built.append({"role": "system", "content": system_prompt})
            built.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            )
            return await openai_complete_if_cache(
                self.vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=built,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        return await self.llm(prompt, system_prompt, history_messages, **kwargs)

    # Embedding model
    async def embed(self, texts: list[str]) -> np.ndarray:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers=headers,
                json={"model": self.embed_model, "input": texts},
            )
            resp.raise_for_status()
            return np.array([item["embedding"] for item in resp.json()["data"]])
