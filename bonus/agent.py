"""
agent.py — HybridMemoryAgent
============================
Three memory layers:
  1. Episodic Memory  → In-process TF-IDF vector store (numpy + sklearn)
  2. Stable Profile   → JSON feature store (tabular, lazy-flushed)
  3. Recent Activity  → In-memory deque (1-hour rolling window)

Design goal: optimise for CLARITY, not speed.
Embedding choice: TF-IDF (sklearn) — no external model download needed.
Production swap: bkai-foundation-models/vietnamese-bi-encoder for better Vietnamese recall.
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "kubernetes": ["kubernetes", "k8s", "pod", "cluster", "deployment", "helm", "kubectl", "hpa", "autoscaler"],
    "cloud":      ["cloud", "aws", "gcp", "azure", "s3", "ec2", "lambda", "đám mây", "re:invent"],
    "ai":         ["ai", "machine learning", "deep learning", "llm", "gpt", "transformer",
                   "trí tuệ nhân tạo", "học máy", "attention", "self-attention"],
    "security":   ["security", "bảo mật", "firewall", "ssl", "tls", "encryption",
                   "zero trust", "vulnerability", "cve", "mtls", "ddos", "armor"],
    "database":   ["database", "cơ sở dữ liệu", "sql", "nosql", "postgres", "mysql",
                   "mongodb", "redis", "vector db"],
    "networking": ["network", "mạng", "tcp", "http", "dns", "load balancer", "nginx",
                   "api gateway", "istio", "envoy", "service mesh", "sidecar"],
    "devops":     ["devops", "ci/cd", "pipeline", "docker", "terraform", "ansible",
                   "monitoring", "observability", "helm"],
}

DEFAULT_PROFILE = {
    "preferred_language": "vi",
    "topic_affinity":     [],
    "reading_speed":      "medium",
    "active_hours":       [9, 10, 14, 15, 16],
}


class _TinyVectorStore:
    """
    Simple vector store: TF-IDF matrix + cosine similarity, persisted as JSON.
    Refit vectorizer on every add() — acceptable for POC scale (< 1000 docs).
    """

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._docs: list[dict] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._docs = data.get("docs", [])
            self._refit()

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps({"docs": self._docs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _refit(self) -> None:
        if not self._docs:
            self._vectorizer = None
            self._matrix = None
            return
        texts = [d["text"] for d in self._docs]
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def add(self, doc_id: str, text: str, metadata: dict) -> None:
        self._docs.append({"id": doc_id, "text": text, "metadata": metadata})
        self._refit()
        self._save()

    def count(self) -> int:
        return len(self._docs)

    def query(self, query_text: str, user_id: str, top_k: int = 3) -> list[dict]:
        if not self._docs or self._vectorizer is None:
            return []
        user_docs = [d for d in self._docs if d["metadata"].get("user_id") == user_id]
        if not user_docs:
            return []
        user_texts = [d["text"] for d in user_docs]
        sub_matrix = self._vectorizer.transform(user_texts)
        query_vec = self._vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, sub_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "text":       user_docs[idx]["text"],
                "metadata":   user_docs[idx]["metadata"],
                "similarity": round(float(scores[idx]), 3),
            }
            for idx in ranked_idx
            if scores[idx] > 0
        ]


class HybridMemoryAgent:
    """
    Hybrid memory agent for a Vietnamese-first personal AI assistant.

    Usage:
        agent = HybridMemoryAgent()
        agent.remember("Tôi đã đọc về Kubernetes HPA hôm nay", user_id="u_001")
        ctx = agent.recall("What have I read about scaling?", user_id="u_001")
        print(ctx)
    """

    def __init__(self, persist_dir: str = "./bonus_data") -> None:
        base = Path(persist_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._episodic = _TinyVectorStore(base / "episodic.json")
        self._store_path = base / "feature_store.json"
        self._feature_store: dict = self._load_store()
        self._recent: dict[str, deque] = {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def remember(self, text: str, user_id: str = "u_001",
                 source: str = "user_input") -> None:
        """Add a new piece of episodic memory for this user."""
        doc_id = f"{user_id}_{int(time.time() * 1_000)}"
        metadata = {"user_id": user_id, "timestamp": time.time(), "source": source}
        self._episodic.add(doc_id, text, metadata)

        inferred_topics = self._extract_topics(text)
        if inferred_topics:
            self._update_topic_affinity(user_id, inferred_topics)

        preview = text[:70] + ("…" if len(text) > 70 else "")
        print(f"[remember] ✓ {user_id} | topics={inferred_topics} | '{preview}'")

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile features → return assembled context."""
        self._log_activity(user_id, query)

        # A. Episodic Memory — vector search
        hits = self._episodic.query(query, user_id=user_id, top_k=top_k)
        if hits:
            episodic_section = "\n".join(
                f"  [sim={h['similarity']}] {h['text']}" for h in hits
            )
        else:
            episodic_section = "  (no matching memories)"

        # B. User Profile — feature store
        profile = self._get_profile(user_id)
        affinity_str = ", ".join(profile["topic_affinity"]) or "none detected yet"
        profile_section = (
            f"  language preference : {profile['preferred_language']}\n"
            f"  topic affinity      : {affinity_str}\n"
            f"  reading speed       : {profile['reading_speed']}"
        )

        # C. Recent Activity — streaming deque
        recent_qs = self._get_recent_queries(user_id, window_sec=3600)
        recent_section = (
            "  " + " | ".join(recent_qs[-5:])
            if recent_qs
            else "  (no recent queries in this session)"
        )

        return (
            "=== EPISODIC MEMORY (vector search) ===\n"
            f"{episodic_section}\n\n"
            "=== USER PROFILE (feature store) ===\n"
            f"{profile_section}\n\n"
            "=== RECENT ACTIVITY (streaming, last 1h) ===\n"
            f"{recent_section}"
        )

    # =========================================================================
    # FEATURE STORE HELPERS
    # =========================================================================

    def _load_store(self) -> dict:
        if self._store_path.exists():
            return json.loads(self._store_path.read_text(encoding="utf-8"))
        return {}

    def _flush_store(self) -> None:
        self._store_path.write_text(
            json.dumps(self._feature_store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_profile(self, user_id: str) -> dict:
        if user_id not in self._feature_store:
            self._feature_store[user_id] = dict(DEFAULT_PROFILE)
            self._flush_store()
        return self._feature_store[user_id]

    def _update_topic_affinity(self, user_id: str, new_topics: list[str]) -> None:
        """Sliding top-10 affinity: most recently seen topic = index 0."""
        profile = self._get_profile(user_id)
        existing: list[str] = profile["topic_affinity"]
        merged = new_topics + [t for t in existing if t not in new_topics]
        profile["topic_affinity"] = merged[:10]
        self._feature_store[user_id] = profile
        self._flush_store()

    # =========================================================================
    # STREAMING HELPERS
    # =========================================================================

    def _log_activity(self, user_id: str, query: str) -> None:
        if user_id not in self._recent:
            self._recent[user_id] = deque(maxlen=200)
        self._recent[user_id].append({"ts": time.time(), "q": query})

    def _get_recent_queries(self, user_id: str, window_sec: int = 3600) -> list[str]:
        if user_id not in self._recent:
            return []
        cutoff = time.time() - window_sec
        return [item["q"] for item in self._recent[user_id] if item["ts"] > cutoff]

    # =========================================================================
    # TOPIC EXTRACTION
    # =========================================================================

    def _extract_topics(self, text: str) -> list[str]:
        """Keyword-based topic detection. Handles code-switching (vi + en)."""
        text_lower = text.lower()
        return [
            topic
            for topic, keywords in TOPIC_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)
        ]