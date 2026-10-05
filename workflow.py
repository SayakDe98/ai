# workflow.py
from research_task import search_topic, get_related, URL, MODEL
import requests

def run_workflow(topic: str) -> dict:
    """Fixed pipeline: always search, always get related, always one summary call."""
    total_tokens = 0

    facts = search_topic(topic)       # step 1, unconditional
    related = get_related(topic)      # step 2, unconditional, even if not needed

    prompt = (
        f"Write a 2-3 sentence summary of '{topic}' using these facts: {facts}\n"
        f"Related topics for context: {related}"
    )
    r = requests.post(URL, json={
        "model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    })
    r.raise_for_status()
    data = r.json()
    total_tokens += data.get("prompt_eval_count", 0) + data.get("eval_count", 0)

    return {
        "summary": data["message"]["content"],
        "tokens": total_tokens,
        "tool_calls": 2,   # always exactly 2, by construction
    }