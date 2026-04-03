"""Ollama LLM client for text generation."""
import httpx
import structlog

logger = structlog.get_logger(__name__)


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
        logger.debug("llm.generating", model=self.model, prompt_length=len(prompt))
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
            response_text = response.json()["response"]
            logger.debug("llm.generated", response_length=len(response_text))
            return response_text
        except (KeyError, ValueError) as e:
            raise OllamaError(f"Unexpected response format from Ollama: {response.text[:200]}") from e

    async def health_check(self) -> bool:
        """Check if Ollama is reachable. Returns True/False."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except (httpx.TransportError, httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("llm.unavailable", error=str(e))
            return False

    def build_rag_prompt(self, question: str, context_chunks: list[str]) -> str:
        """
        Build a RAG prompt with context and question.

        Returns a prompt with system instructions, numbered context chunks,
        and the user's question.
        """
        header = (
            "You are a helpful assistant. Answer the question based ONLY on the provided context. "
            "Answer in the same language as the question. "
            "If the context does not contain enough information, say so briefly and summarize what the context does cover.\n"
            "\n"
            "Context:\n"
        )
        chunk_blocks = "".join(f"---\n{chunk}\n" for chunk in context_chunks)
        footer = f"---\n\nQuestion: {question}\n\nAnswer:"

        return header + chunk_blocks + footer
