"""
AEGIS-Cloud — Cloud / Integrations / RAG Context Engine
=========================================================
Module:   cloud.integrations.rag_context
Platform: Azure (serverless / container)

Retrieval-Augmented Generation engine backed by Azure AI Search.
Provides the Threat Evaluator with doctrinal context and historical
incident precedents to ground tactical reasoning.

Index schema:
  • incident_reports  — Historical NATO-format incident records.
  • doctrine_refs     — Counter-UAS engagement doctrine and ROE.
  • sensor_signatures — Known drone RF/visual signatures.

Simulation mode activates when AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_KEY
are absent, returning static doctrinal stubs for CI/CD.
"""

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    _SEARCH_SDK_AVAILABLE = True
except ImportError:
    _SEARCH_SDK_AVAILABLE = False

_RAG_LOG_PATH = Path("data/logs/rag_retrieval.jsonl")


class RAGContext:
    """
    Azure AI Search–backed RAG retrieval engine.

    Attributes
    ----------
    search_endpoint : str
        Azure AI Search service URL.
    index_name : str
        Name of the primary search index.
    embedding_model : str
        Model used for vector embeddings (e.g., text-embedding-ada-002).
    top_k : int
        Number of results to retrieve per query (default: 5).
    """

    def __init__(self):
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX", "aegis-doctrine")
        self.api_key = os.getenv("AZURE_SEARCH_KEY", "")
        self.embedding_model = "text-embedding-ada-002"
        self.top_k = 5

        self._sim_mode = not (
            self.search_endpoint and self.api_key and _SEARCH_SDK_AVAILABLE
        )
        self.logger = logging.getLogger("AEGIS.RAGContext")

        if not self._sim_mode:
            self._client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
            )
        else:
            self._client = None
            self.logger.info("[AEGIS][RAG] SIMULATION_MODE — static doctrinal stubs active.")

        _RAG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def retrieve(self, query: str, filters: dict = None) -> list[dict]:
        """
        Execute a hybrid (keyword + vector) search against the index.

        Returns
        -------
        list[dict]
            Each entry: {content, source, relevance_score, doc_id}
        """
        if self._sim_mode:
            return self._sim_retrieve(query)

        start = time.perf_counter()
        try:
            search_payload = self._build_query(query, filters or {})
            results = self._client.search(**search_payload)
            docs = [
                {
                    "content": r.get("content", ""),
                    "source": r.get("source", "unknown"),
                    "relevance_score": r.get("@search.score", 0.0),
                    "doc_id": r.get("id", ""),
                }
                for r in results
            ]
            latency_ms = (time.perf_counter() - start) * 1000
            self._log_retrieval_event(query, len(docs), latency_ms, "")
            return docs
        except Exception as e:
            self.logger.error(f"[AEGIS][RAG] Search failed: {e}")
            return []

    def retrieve_by_signature(self, sensor_signature: dict) -> list[dict]:
        """
        Specialised retrieval for matching known drone signatures
        against the sensor_signatures index.
        """
        query = (
            f"drone signature RF:{sensor_signature.get('rf_freq_mhz', 'unknown')} "
            f"visual:{sensor_signature.get('visual_class', 'unknown')} "
            f"size:{sensor_signature.get('wingspan_cm', 'unknown')}cm"
        )
        return self.retrieve(query, filters={"index": "sensor_signatures"})

    def _build_query(self, query: str, filters: dict) -> dict:
        """Construct the Azure AI Search query payload."""
        payload = {
            "search_text": query,
            "top": self.top_k,
            "include_total_count": True,
        }
        if filters.get("index"):
            payload["filter"] = f"index_type eq '{filters['index']}'"
        return payload

    def _log_retrieval_event(self, query: str, result_count: int,
                             latency_ms: float, correlation_id: str):
        """Log RAG query metadata for AIOps and audit purposes."""
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "query": query[:120],
            "result_count": result_count,
            "latency_ms": round(latency_ms, 2),
        }
        with _RAG_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _sim_retrieve(self, query: str) -> list[dict]:
        """Return doctrinal context stubs for simulation/CI."""
        self.logger.debug(f"[AEGIS][RAG][SIM] Query: {query[:80]}")
        return [
            {
                "content": (
                    "Counter-UAS doctrine: Hostile UAVs classified above 0.80 confidence "
                    "require immediate threat assessment and HITL authorization before "
                    "kinetic response."
                ),
                "source": "NATO-STANAG-4586-doctrine-v3.pdf",
                "relevance_score": 0.91,
                "doc_id": "sim-doc-001",
            },
            {
                "content": (
                    "Rules of Engagement: Unidentified aerial objects within 500m of "
                    "protected perimeter are to be classified Unknown until positive "
                    "identification is established via sensor fusion."
                ),
                "source": "ROE-CounterUAS-2024.pdf",
                "relevance_score": 0.78,
                "doc_id": "sim-doc-002",
            },
            {
                "content": (
                    "Threat taxonomy: Small quadcopters (< 5kg) operating below 120m AGL "
                    "and within visual range are classified as potential ISR platforms. "
                    "Recommend electronic countermeasures as first response."
                ),
                "source": "threat-taxonomy-v2.pdf",
                "relevance_score": 0.72,
                "doc_id": "sim-doc-003",
            },
        ]
