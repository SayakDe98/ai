# compare.py
import time
import random
from workflow import run_workflow
from agent_new import run_agent
from research_task import KNOWLEDGE_BASE

topics = list(KNOWLEDGE_BASE.keys())

def benchmark(fn, n=20):
    results = []
    for i in range(n):
        topic = random.choice(topics)
        start = time.time()
        try:
            out = fn(topic)
            out["latency"] = time.time() - start
            out["success"] = bool(out.get("summary")) and not out.get("hit_max_iters", False)
        except Exception as e:
            out = {"latency": time.time() - start, "success": False, "tokens": 0, "tool_calls": 0, "error": str(e)}
        results.append(out)
    return results

def summarize(name, results):
    n = len(results)
    successes = sum(r["success"] for r in results)
    avg_latency = sum(r["latency"] for r in results) / n
    avg_tokens = sum(r["tokens"] for r in results) / n
    avg_calls = sum(r["tool_calls"] for r in results) / n
    print(f"\n{name}")
    print(f"  success rate:   {successes}/{n} ({successes/n:.0%})")
    print(f"  avg latency:    {avg_latency:.2f}s")
    print(f"  avg tokens:     {avg_tokens:.0f}")
    print(f"  avg tool calls: {avg_calls:.1f}")

if __name__ == "__main__":
    print("Running workflow x20...")
    wf_results = benchmark(run_workflow)
    print("Running agent x20...")
    agent_results = benchmark(run_agent)

    summarize("WORKFLOW", wf_results)
    summarize("AGENT", agent_results)