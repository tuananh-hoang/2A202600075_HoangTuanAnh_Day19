"""
demo.py — 5-query demonstration of HybridMemoryAgent
=====================================================
Run:  python bonus/demo.py

What this script does:
  1. Populates episodic memory with realistic Vietnamese-context documents
  2. Runs 5 required demo queries, printing the assembled context each time
  3. (Optional) sends each assembled context to Anthropic Claude to generate
     a natural-language answer — uncomment USE_LLM = True to activate

No API key is needed for the core 5 queries (they just print the assembled
context). The LLM synthesis section demonstrates the full pipeline.
"""

import os
import sys
import shutil
import textwrap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "./bonus_data_demo"   # isolated from your real data
USE_LLM  = False  # set True when ANTHROPIC_API_KEY is set                  # set False to skip Anthropic API calls

# ---------------------------------------------------------------------------

def banner(n: int, title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  DEMO {n}: {title}")
    print(f"{'='*65}")


def print_context(ctx: str) -> None:
    for line in ctx.splitlines():
        print("  " + line)


def ask_llm(query: str, context: str) -> str:
    """Send assembled context + user query to Claude and return the answer."""
    try:
        import anthropic
    except ImportError:
        return "[anthropic package not installed — pip install anthropic]"

    client = anthropic.Anthropic()
    system_prompt = textwrap.dedent("""
        You are a helpful personal AI assistant for a Vietnamese tech user.
        You have access to the user's memory context below.
        Answer in the same language the user writes in (Vietnamese or English).
        Be concise — 2–4 sentences maximum.
        If the context is sparse, say so honestly.
    """).strip()

    user_message = f"""Memory context:
{context}

User query: {query}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def run():
    # Clean slate for demo reproducibility
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    # Import after potential cleanup
    from agent import HybridMemoryAgent

    agent = HybridMemoryAgent(persist_dir=DATA_DIR)
    USER = "u_001"

    # -------------------------------------------------------------------------
    # SEED: populate episodic memory with realistic Vietnamese tech content
    # -------------------------------------------------------------------------
    print("\n📥  Seeding episodic memory …\n")

    memories = [
        # Kubernetes cluster
        ("Hôm nay tôi đọc về Kubernetes Horizontal Pod Autoscaler (HPA). "
         "HPA tự động scale số lượng pod dựa trên CPU utilization hoặc custom metrics. "
         "Cần set resource requests đúng để HPA hoạt động chính xác.",
         "doc_k8s_hpa"),

        # Cloud architecture
        ("AWS re:Invent 2024: Amazon giới thiệu S3 Express One Zone — single-AZ storage "
         "với latency cực thấp cho AI/ML workloads. Phù hợp cho training data lakes "
         "nhưng không có multi-AZ durability.",
         "doc_aws_s3express"),

        # Security
        ("Nghiên cứu về Zero Trust Security Model: 'Never trust, always verify'. "
         "Mọi request phải được authenticate dù từ internal network. "
         "Áp dụng tốt cho microservices với mutual TLS (mTLS) giữa các service.",
         "doc_zerotrust"),

        # Cloud security overlap
        ("Cloud security trên GCP: Cloud Armor để chống DDoS, VPC Service Controls "
         "để giới hạn data exfiltration, Binary Authorization để enforce container "
         "image signing trong GKE cluster.",
         "doc_gcp_security"),

        # Kubernetes + DevOps
        ("Helm chart best practices: dùng _helpers.tpl cho template functions, "
         "values.yaml có default values, và luôn set resource limits trong container spec. "
         "Kubernetes deployment rolling update strategy giúp zero-downtime deploy.",
         "doc_helm"),

        # AI/ML
        ("Đọc paper 'Attention Is All You Need' — kiến trúc Transformer dùng "
         "self-attention thay vì RNN. Multi-head attention cho phép model focus "
         "vào nhiều position khác nhau cùng lúc. Đây là nền tảng của hầu hết LLM hiện đại.",
         "doc_transformer"),

        # Networking
        ("Tìm hiểu về service mesh với Istio: sidecar proxy (Envoy) inject vào mỗi pod, "
         "xử lý load balancing, circuit breaking, và distributed tracing. "
         "Traffic management qua VirtualService và DestinationRule CRDs.",
         "doc_istio"),
    ]

    for text, source in memories:
        agent.remember(text, user_id=USER, source=source)

    print(f"\n  ✓ Stored {len(memories)} episodic memories for user '{USER}'")

    # -------------------------------------------------------------------------
    # DEMO QUERIES
    # -------------------------------------------------------------------------

    # --- Demo 1: Simple lookup (vector only) ---------------------------------
    q1 = "What have I read about Kubernetes?"
    banner(1, "Simple lookup — vector search only")
    print(f"  Query: \"{q1}\"\n")
    ctx1 = agent.recall(q1, user_id=USER, top_k=3)
    print_context(ctx1)
    if USE_LLM:
        print("\n  💬 LLM Answer:")
        print(f"  {ask_llm(q1, ctx1)}")

    # --- Demo 2: Profile-needed (topic_affinity) ------------------------------
    q2 = "Recommend what I should read next"
    banner(2, "Profile-needed — uses topic_affinity from feature store")
    print(f"  Query: \"{q2}\"\n")
    ctx2 = agent.recall(q2, user_id=USER, top_k=2)
    print_context(ctx2)
    if USE_LLM:
        print("\n  💬 LLM Answer:")
        print(f"  {ask_llm(q2, ctx2)}")

    # --- Demo 3: Fresh activity — uses streaming deque -----------------------
    q3 = "What am I focused on lately?"
    banner(3, "Fresh activity — uses streaming queries_last_hour")
    print(f"  Query: \"{q3}\"\n")
    # Simulate two prior queries in this session (already logged by demos 1+2)
    ctx3 = agent.recall(q3, user_id=USER, top_k=2)
    print_context(ctx3)
    if USE_LLM:
        print("\n  💬 LLM Answer:")
        print(f"  {ask_llm(q3, ctx3)}")

    # --- Demo 4: Paraphrase — vector wins on semantic match ------------------
    q4 = "Documents about scaling infrastructure?"
    banner(4, "Paraphrase query — semantic vector search handles lexical gap")
    print(f"  Query: \"{q4}\"\n")
    print("  (Note: memories used 'HPA', 'autoscaler', 'pod' — not 'scaling infrastructure')")
    print("  Vector search bridges this semantic gap.\n")
    ctx4 = agent.recall(q4, user_id=USER, top_k=3)
    print_context(ctx4)
    if USE_LLM:
        print("\n  💬 LLM Answer:")
        print(f"  {ask_llm(q4, ctx4)}")

    # --- Demo 5: Mixed — hybrid + profile ------------------------------------
    q5 = "Give me a cloud security summary"
    banner(5, "Mixed query — hybrid retrieval + profile context")
    print(f"  Query: \"{q5}\"\n")
    ctx5 = agent.recall(q5, user_id=USER, top_k=3)
    print_context(ctx5)
    if USE_LLM:
        print("\n  💬 LLM Answer:")
        print(f"  {ask_llm(q5, ctx5)}")

    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("  All 5 demos complete. ✓")
    print("="*65 + "\n")


if __name__ == "__main__":
    run()