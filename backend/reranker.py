import logging
import asyncio
from sentence_transformers import CrossEncoder

_logger = logging.getLogger(__name__)


class DocumentReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._reranker = None

    def load(self):
        """Preloads the model into memory."""
        if self._reranker is None:
            _logger.info(f"Loading reranker {self.model_name} ...")
            self._reranker = CrossEncoder(self.model_name)
            _logger.info("Reranker ready.")
        return self._reranker

    async def rerank(self, query: str, documents: list[str], top_n: int = 20):
        """Async wrapper to run the CPU-heavy predict function without blocking the API."""
        loop = asyncio.get_running_loop()
        pairs = [[query, doc] for doc in documents]

        # Run the heavy computation in a background thread
        scores = await loop.run_in_executor(
            None, lambda: self.load().predict(pairs, show_progress_bar=True)
        )

        results = [
            {"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)
        ]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        _logger.info(
            f"[rerank] {len(documents)} docs → top: "
            f"{[round(r['relevance_score'], 3) for r in results[:5]]}"
        )
        return results
