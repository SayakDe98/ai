# agent.py
from research_task import search_topic, get_related, URL, MODEL
import requests

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_topic",
            "description": "Look up factual information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related",
            "description": "Get related topics for further context.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
]

DISPATCH = {"search_topic": search_topic, "get_related": get_related}

def run_agent(topic: str, max_iters: int = 5) -> dict:
    messages = [{"role": "user", "content":
        f"Research '{topic}' and write a 2-3 sentence summary. Use tools as needed, then give your final summary as plain text."}]
    total_tokens = 0
    tool_call_count = 0

    for _ in range(max_iters):
        r = requests.post(URL, json={
            "model": MODEL, "messages": messages, "tools": tools, "stream": False,
        })
        r.raise_for_status()
        data = r.json()
        total_tokens += data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        msg = data["message"]
        messages.append(msg)

        calls = msg.get("tool_calls")
        if not calls:
            return {"summary": msg["content"], "tokens": total_tokens,
                     "tool_calls": tool_call_count, "hit_max_iters": False}

        for call in calls:
            tool_call_count += 1
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = DISPATCH.get(name, lambda **_: "error: unknown tool")(**args)
            messages.append({"role": "tool", "content": result})

    return {"summary": None, "tokens": total_tokens,
             "tool_calls": tool_call_count, "hit_max_iters": True}