import logging
import asyncio
import httpx
from sentence_transformers import CrossEncoder

_logger = logging.getLogger(__name__)


class DocumentReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "",
        api_key: str = "",
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self._reranker = None

    def load(self):
        """Preloads the local model into memory if we are NOT using an external API."""
        if self.base_url:
            return None  # Skip loading on CPU if an external URL is provided

        if self._reranker is None:
            _logger.info(f"Loading local CPU reranker {self.model_name} ...")
            self._reranker = CrossEncoder(self.model_name)
            _logger.info("Local Reranker ready.")
        return self._reranker

    async def rerank(self, query: str, documents: list[str], top_n: int = 20):
        """Routes to an external API if configured, otherwise runs local CPU inference."""

        # External routing
        if self.base_url:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                _logger.info(
                    f"Sending rerank request to {self.base_url} for {len(documents)} docs"
                )
                resp = await client.post(
                    f"{self.base_url.rstrip('/')}/rerank", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()

                if "results" in data:
                    results = data["results"]
                    _logger.info(
                        f"[rerank-external] top: "
                        f"{[round(r.get('relevance_score', 0), 3) for r in results[:5]]}"
                    )
                    return results
                return []

        # Local routing
        loop = asyncio.get_running_loop()
        pairs = [[query, doc] for doc in documents]

        # Run the heavy computation in a background thread
        scores = await loop.run_in_executor(
            None, lambda: self.load().predict(pairs, show_progress_bar=False)
        )

        results = [
            {"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)
        ]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        _logger.info(
            f"[rerank-local] {len(documents)} docs → top: "
            f"{[round(r['relevance_score'], 3) for r in results[:5]]}"
        )
        return results[:top_n]
