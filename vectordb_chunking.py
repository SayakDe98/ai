import requests
import psycopg2

OLLAMA = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

DB_PARAMS = dict(dbname="postgres")  # add user=, host=, port= if your setup needs them

# ---- Embedding ----
def embed(text: str) -> list[float]:
    r = requests.post(OLLAMA, json={"model": EMBED_MODEL, "input": text})
    r.raise_for_status()
    return r.json()["embeddings"][0]

# ---- Naive fixed-size chunking (Stage 1 baseline) ----
def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
    return chunks

# ---- Load, chunk, embed, insert ----
def build_index(filepath: str):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute("DELETE FROM chunks")  # clean slate each run, so re-runs don't duplicate

    with open(filepath) as f:
        text = f.read()

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks): #enumerate used for tracking the index and data at the same time
        vec = embed(chunk)
        cur.execute(
            "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s)",
            (f"{filepath}#{i}", chunk, vec)
        )

    conn.commit()
    print(f"Inserted {len(chunks)} chunks from {filepath}")
    conn.close()

# ---- Vector search ----
def search(query: str, k: int = 5) -> list[str]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    vec = embed(query)
    cur.execute(
        "SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (vec, k)
    ) # <=> means cosine distance between two vectors.
    results = [row[0] for row in cur.fetchall()]
    conn.close()
    return results

# ---- Eval set ----
eval_set = [
    {"question": "What database does Wrenfield Ledger use?", "expected_keyword": "postgresql"},
    {"question": "What connection pooler is used for the database?", "expected_keyword": "pgbouncer"},
    {"question": "How many read replicas does the database have?", "expected_keyword": "two"},
    {"question": "What caching system is used and why was it chosen over Memcached?", "expected_keyword": "redis"},
    {"question": "How long do access tokens last before expiring?", "expected_keyword": "15 minutes"},
    {"question": "What SSO provider is authentication built on?", "expected_keyword": "okta"},
    {"question": "What are the four RBAC roles?", "expected_keyword": "budget_owner"},
    {"question": "What is the current API version and when was v1 deprecated?", "expected_keyword": "january 2025"},
    {"question": "What is the rate limit for standard API keys?", "expected_keyword": "100 requests"},
    {"question": "What Kubernetes version does the cluster run?", "expected_keyword": "1.29"},
    {"question": "How does the canary deployment process work?", "expected_keyword": "10%"},
    {"question": "What bug did the canary deployment catch involving currency?", "expected_keyword": "yen"},
    {"question": "What is the current Redis cache hit rate for budget summaries?", "expected_keyword": "87%"},
    {"question": "How many engineers are on the on-call rotation?", "expected_keyword": "six"},
    {"question": "How many engineers are on the Wrenfield Ledger team total?", "expected_keyword": "five"},
    {"question": "What is the maximum sustained transaction throughput the system handles?", "expected_keyword": "500 transactions"},
    {"question": "Why doesn't the system support multi-currency budgets yet?", "expected_keyword": "immutable"},
    {"question": "What notification channels are currently supported?", "expected_keyword": "slack"},
]

# ---- Recall@k ----
def recall_at_k(eval_set, k: int = 5) -> float:
    hits = 0
    for item in eval_set:
        results = search(item["question"], k=k)
        found = any(item["expected_keyword"].lower() in r.lower() for r in results)
        status = "✓" if found else "✗"
        print(f"  {status} {item['question']}")
        if found:
            hits += 1
    return hits / len(eval_set)

# ---- Run ----
if __name__ == "__main__":
    build_index("sample.txt")
    print()
    score = recall_at_k(eval_set, k=2)   # now search must narrow 3 chunks down to 2
    print(f"\nRecall@5: {score:.0%} ({round(score * len(eval_set))}/{len(eval_set)})")