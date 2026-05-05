# Hybrid Memory Agent — Architecture Document
> **Bonus Challenge — Lab 19: Build Your Own AI Memory**
> **Contributors:** [2A202600075 - Hoang Tuan Anh]
> **Target:** Vietnamese-first personal AI assistant with episodic + profile + streaming memory

---

## System Overview

The agent combines **three memory layers** that serve fundamentally different access patterns. Think of it like a human brain: you have long-term memories of specific events (episodic), stable personality traits and preferences (profile), and a short-term awareness of what you've been thinking about in the last hour (recent activity).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INPUT / QUERY                               │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    HybridMemoryAgent.recall()                           │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  EPISODIC MEMORY │  │   USER PROFILE   │  │   RECENT ACTIVITY     │ │
│  │  (Vector Store)  │  │  (Feature Store) │  │  (Streaming, in-mem)  │ │
│  │                  │  │                  │  │                       │ │
│  │ ChromaDB         │  │ JSON flat file   │  │ collections.deque     │ │
│  │ cosine similarity│  │ TTL: weekly      │  │ TTL: 1-hour rolling   │ │
│  │ top-K retrieval  │  │ tabular features │  │ last-N queries        │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬────────────┘ │
│           │                     │                        │              │
│           └─────────────────────┴────────────────────────┘              │
│                                         │                               │
│                                         ▼                               │
│                          ┌──────────────────────────┐                   │
│                          │   CONTEXT ASSEMBLER      │                   │
│                          │   (prompt builder)       │                   │
│                          └──────────────┬───────────┘                   │
└─────────────────────────────────────────┼───────────────────────────────┘
                                          │
                                          ▼
                          ┌──────────────────────────┐
                          │   ANTHROPIC LLM API      │
                          │  claude-sonnet-4-...      │
                          │  assembled context →      │
                          │  final answer             │
                          └──────────────────────────┘
```

**Data flow for `remember(text)`:**
```
text → chunk into segments
     → embed via ChromaDB default EF
     → store in episodic collection (user_id filter)
     → extract topics → update feature store topic_affinity
```

**Data flow for `recall(query)`:**
```
query → log to streaming deque (recent_activity)
      → vector search episodic (filter by user_id, top-K)
      → load feature store profile (language, affinity, speed)
      → read deque (queries_last_hour)
      → assemble all 3 into context string
      → (optional) send to Anthropic API for final answer
```

---

## Architecture Decision 1 — Chunking Strategy

**Decision: Per-conversation-turn chunking with 512-token hard ceiling.**

When a user shares a document or conversation turn, we chunk it into **individual semantic units** — one chunk per meaningful "thought" (paragraph or turn), with a 512-token ceiling. If a paragraph exceeds that, we slide a 64-token overlap window.

### Why not alternatives?

| Strategy | Retrieval Quality | Storage Cost | Context Window | Verdict |
+|---|---|---|---|---|
+| Whole-document as 1 chunk | ❌ Low (diluted embeddings) | ✅ Cheap | ❌ Blows up context | Rejected |
+| Per-sentence | ❌ Lost context | ✅ Tiny | ✅ Precise | Rejected |
+| Per-message/turn (chosen) | ✅ Conversational unit | ✅ Moderate | ✅ Fits well | **Chosen** |
+| Semantic break (NLP-driven) | ✅ Best quality | ❌ Expensive | ✅ Natural | Too slow for POC |

**Tradeoff explicitly:**
- We **gain**: retrieval relevance stays high because each chunk has a clear semantic identity. Searching "Kubernetes HPA" returns the specific turn where that was discussed, not an embedding soup of an entire 10-page doc.
- We **lose**: cross-paragraph context. If the user's insight spans two paragraphs, we might only retrieve the second one. This is acceptable for a POC — a production system would add a "sibling chunk" re-ranking pass.

**Vietnamese implication:** Vietnamese sentences are shorter on average than English. A 512-token ceiling is generous for Vietnamese content — in practice, most conversational turns land at 100–250 tokens. This means we get **finer-grained retrieval** for Vietnamese without extra cost, which is a free win.

---

## Architecture Decision 2 — Feature Schema (User Profile)

**Decision: Tabular features with a simple keyword-based topic affinity list, not embedding features.**

The feature store holds a flat JSON profile per user:

```json
{
  "u_001": {
    "preferred_language": "vi",           // entity: user, TTL: manual update
    "topic_affinity": ["kubernetes", "cloud", "ai"],  // entity: user, TTL: sliding top-10
    "reading_speed": "medium",            // entity: user, TTL: computed weekly
    "active_hours": [9, 10, 14, 15, 16]  // entity: user, TTL: rolling 7-day average
  }
}
```

**Pattern choice — tabular vs embedding features:**

| Pattern | How it works | When to use |
+|---|---|---|
+| **Tabular (chosen)** | Explicit key-value pairs you can read and explain | Profile features you understand and can act on |
+| **Embedding features** | Dense vector of latent preferences from history | When preferences are implicit and high-dimensional |

We chose **tabular** because:
1. **Explainability** — when the agent recommends something based on `topic_affinity: ["cloud"]`, we know exactly why. Embedding-based preferences are a black box.
2. **Editability** — users can explicitly override their profile. "I don't care about Kubernetes anymore" is a one-field update.
3. **No extra model call** — embedding features require running a model over the user's history. For a POC, this is unnecessary cost.

**Tradeoff explicitly:**
- We **gain**: debuggability, simplicity, zero inference cost for profile reads.
- We **lose**: nuanced latent preferences. A user who only reads cloud security articles late at night would register as both `cloud` and `security` affinity, but we lose the "cloud *security specifically*" latent cluster. An embedding feature view would capture that cluster naturally.

**Source for each feature:**
- `preferred_language`: explicit user setting or detected from first 3 messages
- `topic_affinity`: inferred from `remember()` calls via keyword matching, rolling top-10 by recency
- `reading_speed`: future — measured by time-spent-per-document ratio
- `active_hours`: future — timestamp histogram from `recall()` calls

---

## Architecture Decision 3 — Freshness Strategy

**Decision: Three-tier freshness based on memory type.**

Not all memory needs to be fresh at the same speed. Here's the policy:

| Memory Layer | Freshness | Mechanism | Reason |
+|---|---|---|---|
+| Recent activity (streaming) | **Sub-second** | in-memory `deque`, appended on every `recall()` | User needs to know "what am I thinking about right now" instantly |
+| Episodic memory (vector) | **Near-real-time (~seconds)** | `remember()` writes directly to ChromaDB; indexed immediately | New documents must be searchable within the same session |
+| Stable user profile | **5–10 minutes (lazy update)** | JSON file, written after each `remember()` only when topics change | Language pref and affinity don't change mid-conversation |

**Three concrete use cases with different freshness needs:**

1. **"What am I focused on lately?"** — needs **sub-second** streaming freshness. The answer must reflect the last 5 queries of this session. Batch refresh would be useless here.

2. **"What have I read about Kubernetes?"** — needs **session-level** freshness. If the user just uploaded a Kubernetes doc, it must be findable immediately. A 5-minute batch window would make the agent seem broken.

3. **"Recommend what to read next"** (uses `topic_affinity`) — **5–10 minute** freshness is fine. Topic affinity drifts slowly. If the user pivots from cloud to security today, the recommendation engine can lag by a few minutes without harm.

**Tradeoff explicitly:**
- Sub-second for everything would mean holding ALL episodic memory in RAM (no persistence). We lose durability.
- Daily batch for everything is simple but makes the assistant feel "dumb" mid-session — it won't know what you just told it.
- **Tiered freshness** matches data change rate to storage cost: volatile → RAM, semi-stable → disk with fast writes, stable → lazy JSON flush.

---

## Rejected Alternative — Storing Episodic Memory in the Feature Store

> "I considered storing episodic memory as an embedding feature view inside the feature store (alongside the user profile), but rejected this in favour of a dedicated vector store."

**Reason:** Re-index cycles differ fundamentally.

- Episodic memory grows **hourly** — every conversation adds new chunks that need embedding and indexing immediately.
- User profile features update **weekly** or less — language preference, reading speed, topic affinity all drift slowly.

If we stored both in the same system:
1. Every new memory would trigger a re-embed of the entire feature view — expensive and slow.
2. Or we'd batch index and lose sub-minute freshness for episodic recall.
3. Feature stores are optimised for **point lookups by entity key**, not **approximate nearest-neighbour search** across all content. The query patterns are orthogonal.

Separating them lets each subsystem run at its natural speed. This is the same reasoning behind PIT (Point-In-Time) joins in production ML systems — you don't want your fast-moving features and slow-moving features on the same refresh schedule.

---

## Vietnamese-Context Considerations

### 1. Tokenizer Choice

| Tokenizer | Accuracy | Speed | Vietnamese Specialisation |
+|---|---|---|---|
+| `underthesea` | ✅ High | ❌ Slow (~200ms/sentence) | ✅ Built for Vietnamese, word segmentation |
+| `pyvi` | ✅ Good | ✅ Fast (~20ms) | ✅ Vietnamese-specific |
+| Whitespace split | ❌ Wrong | ✅ Fastest | ❌ Vietnamese is not whitespace-segmented |

**Decision for POC:** We skip tokenisation entirely at the chunking layer and rely on ChromaDB's default embedding function (which internally uses `all-MiniLM-L6-v2`, a multilingual model). This model handles Vietnamese reasonably well without explicit word segmentation. For a production system, we would preprocess with `underthesea` before embedding to improve retrieval quality on Vietnamese-specific content.

### 2. Code-Switching (vi/en mix)

Vietnamese tech content heavily mixes Vietnamese and English: *"Tôi đã đọc về **Kubernetes deployment** và **horizontal pod autoscaling**"*. Our keyword topic extractor handles this by checking both languages in the same keyword list — `"kubernetes"` matches regardless of surrounding Vietnamese text. Embeddings naturally handle code-switching since the multilingual model sees both languages during training.

### 3. Phonetic Typos

Vietnamese keyboard input (Telex/VNI) frequently produces typos. *"kubernetes"* might appear as *"kubernetis"* or *"kuberntes"*. Vector search is robust to this — fuzzy semantic matches survive one or two character swaps. This is one area where vector search beats traditional keyword search for Vietnamese users.

### 4. Privacy / Decree 13 Angle

Vietnam's Decree No. 13/2023/NĐ-CP on Personal Data Protection requires explicit consent for storing personal data. A production system would need:
- User consent flow before any `remember()` call persists data
- Per-user data deletion endpoint (`forget(user_id)`)
- Data residency considerations (local ChromaDB vs cloud-hosted)

This POC stores data locally and doesn't address multi-user isolation beyond `user_id` filtering — see "Limitations" below.

---

## What This POC Doesn't Handle Yet

- **Privacy isolation:** All users share the same ChromaDB collection. User A filtering by `user_id` is a soft isolation — a misconfigured query could leak memories. Production: per-user collection or per-user encryption key.
- **Multi-device sync:** The feature store JSON and ChromaDB are local files. Two devices would have divergent memory.
- **Memory decay / forgetting:** Old episodic memories never expire. A production system would TTL memories older than 30 days of non-access, or consolidate 5 similar memories into a weekly summary.
- **Concurrent writes:** No locking on the JSON feature store. Two simultaneous `remember()` calls could corrupt the file.
- **Embedding model quality for Vietnamese:** `all-MiniLM-L6-v2` is multilingual but not Vietnamese-optimised. `bkai-foundation-models/vietnamese-bi-encoder` would give better Vietnamese retrieval at the cost of a larger model.
- **Personalisation re-ranking:** After vector top-K, we don't re-rank by `topic_affinity`. A full RRF (Reciprocal Rank Fusion) with 3 retrievers (vector + profile-boosted + recency) would improve recommendation quality significantly.
