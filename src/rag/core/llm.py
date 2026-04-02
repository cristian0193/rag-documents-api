"""Ollama LLM client for text generation."""
import httpx


class OllamaError(Exception):
    pass


class OllamaClient:
    """HTTP client for Ollama API."""

    def __init__(self, base_url: str, model: str = "llama3.2:3b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str) -> str:
        """
        Generate text from prompt using Ollama.

        Args:
            prompt: The complete prompt (including context for RAG)

        Returns:
            Generated response string

        Raises:
            OllamaError: If Ollama is unavailable or returns an error
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaError(f"Failed to connect to Ollama: {e}") from e

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama returned status {response.status_code}: {response.text}"
            )

        try:
            return response.json()["response"]
        except (KeyError, ValueError) as e:
            raise OllamaError(f"Unexpected response format from Ollama: {response.text[:200]}") from e

    async def health_check(self) -> bool:
        """Check if Ollama is reachable. Returns True/False."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except (httpx.TransportError, httpx.TimeoutException, httpx.ConnectError):
            return False

    def build_rag_prompt(self, question: str, context_chunks: list[str]) -> str:
        """
        Build a RAG prompt with context and question.

        Returns a prompt with system instructions, numbered context chunks,
        and the user's question.
        """
        header = (
            "You are a helpful assistant. Answer the question based on the provided context.\n"
            'If the answer is not in the context, say "I don\'t have enough information to answer this question."\n'
            "\n"
            "Context:\n"
        )
        chunk_blocks = "".join(f"---\n{chunk}\n" for chunk in context_chunks)
        footer = f"---\n\nQuestion: {question}\n\nAnswer:"

        return header + chunk_blocks + footer
