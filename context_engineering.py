# context_engineering.py
# Experiment: plant a "needle" (a lucky number) at the start of a chat, pad the chat with
# many tool-using filler rounds, then ask for the needle back. Runs twice — once letting
# history grow unbounded, once with compaction (old messages replaced by a summary) — and
# compares recall accuracy and token cost.

import requests  # HTTP client used to talk to the Ollama server
from research_task import KNOWLEDGE_BASE  # {topic: fact} dict backing the search tool

URL = "http://localhost:11434/api/chat"  # Ollama's chat endpoint (default local port 11434)
MODEL = "llama3.2"  # Model Ollama should run

# Tool schema the model is allowed to call (OpenAI-style function format, which Ollama accepts).
# Declares one tool, search_topic, taking a single required string argument "topic".
tools = [
    {"type": "function", "function": {
        "name": "search_topic", "description": "Look up factual information about a topic.",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    }},
]

# The Python implementation that runs when the model calls search_topic.
def search_topic(topic: str) -> str:
    key = topic.lower().strip()  # Normalize so it matches the dict keys
    return KNOWLEDGE_BASE.get(key, f"No information found for '{topic}'.")  # Fact, or a fallback message

# Maps tool names (as the model sends them) to the Python functions that handle them.
DISPATCH = {"search_topic": search_topic}

NEEDLE = "7492"  # The value the model must remember across the whole conversation
NEEDLE_MESSAGE = f"Remember this number, you'll need it later: my lucky number is {NEEDLE}."  # First user message
TOPICS = list(KNOWLEDGE_BASE.keys())  # Topics to cycle through in the filler rounds

# Send the full message history (plus tool definitions) to the model and return the parsed JSON reply.
def call_model(messages: list) -> dict:
    # stream=False -> one complete JSON response instead of streamed chunks
    r = requests.post(URL, json={"model": MODEL, "messages": messages, "tools": tools, "stream": False})
    r.raise_for_status()  # Raise on HTTP errors (e.g. Ollama not running, model missing)
    # Response contains "message" plus token counts: prompt_eval_count (input) and eval_count (output)
    return r.json()

def run_filler_turn(messages: list, topic: str) -> int:
    """One round: ask about a topic, let the model use the tool, append everything. Returns tokens used."""
    # Ask a filler question; this grows the history and pushes the needle further back
    messages.append({"role": "user", "content": f"Briefly, what's interesting about {topic}? One sentence."})
    tokens = 0
    for _ in range(3):  # small inner tool loop per turn
        data = call_model(messages)
        # Input tokens include the whole history, so cost per call grows as history grows
        tokens += data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        msg = data["message"]
        messages.append(msg)  # Keep the assistant's reply (text and/or tool calls) in history
        calls = msg.get("tool_calls")
        if not calls:
            break  # No tool calls means the model produced its final answer for this round
        for call in calls:
            # Look up the function by name and call it; Ollama sends arguments as a dict, so ** unpacks it
            result = DISPATCH[call["function"]["name"]](**call["function"]["arguments"])
            # Feed the tool result back so the model sees it on the next loop iteration
            messages.append({"role": "tool", "content": result})
    return tokens

# Compaction logic:
def estimate_context_tokens(messages: list) -> int:
    """Rough proxy: count words and multiply (~0.75 tokens/word is a common ratio).
    Not exact, but consistent enough to trigger compaction predictably."""
    # NOTE: the usual rule of thumb is ~0.75 *words per token*; dividing by 0.75 gives ~1.33 tokens/word,
    # which is the correct direction — the docstring just states the ratio backwards.
    text = " ".join(str(m.get("content", "")) for m in messages)  # Concatenate all message text
    return int(len(text.split()) / 0.75)  # word count -> approximate token count

def compact(messages: list, keep_last: int = 4) -> tuple[list, int]:
    """Summarize everything except the last `keep_last` messages into one summary message."""
    if len(messages) <= keep_last + 1:
        return messages, 0  # Too little history to be worth compacting; zero cost

    to_summarize = messages[:-keep_last]  # Older messages to fold into the summary
    recent = messages[-keep_last:]  # Most recent messages, kept verbatim
    # NOTE: `recent` may begin with a "tool" message whose assistant tool_call got summarized away.

    # Flatten older messages into a "role: content" transcript, skipping empty content
    # (e.g. assistant messages that contain only tool calls)
    transcript = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in to_summarize if m.get("content"))
    # Explicitly ask to preserve numbers the user said to remember — this is what lets the needle survive
    prompt = (
        "Summarize the key facts, numbers, and names mentioned in this conversation so far, "
        "in under 60 words. Preserve any specific numbers or facts the user told you to remember.\n\n"
        f"{transcript}"
    )
    data = call_model([{"role": "user", "content": prompt}])  # Fresh one-message conversation just for summarizing
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)  # Summarization isn't free; track its cost
    summary = data["message"]["content"]

    # New history = one system message holding the summary + the recent messages
    new_messages = [{"role": "system", "content": f"[Earlier conversation summary]: {summary}"}] + recent
    return new_messages, tokens

# Run one full experiment: plant needle, run n_rounds of filler (optionally compacting), then test recall.
def run_scenario(use_compaction: bool, threshold: int = 600, n_rounds: int = 50) -> dict:
    messages = [{"role": "user", "content": NEEDLE_MESSAGE}]  # Start with the needle
    # get the model to acknowledge, so it's a real turn in history
    data = call_model(messages)
    total_tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    messages.append(data["message"])

    compaction_events = 0  # How many times compaction fired
    context_sizes = []  # Estimated context size after each round (measured before any compaction)

    for i in range(n_rounds):
        topic = TOPICS[i % len(TOPICS)]  # Cycle through topics
        total_tokens += run_filler_turn(messages, topic)

        ctx_size = estimate_context_tokens(messages)
        context_sizes.append(ctx_size)

        # Compact once the estimated context exceeds the threshold
        if use_compaction and ctx_size > threshold:
            messages, compaction_tokens = compact(messages)
            total_tokens += compaction_tokens  # Include summarization cost in the total
            compaction_events += 1

    # Final test: can it recall the needle?
    messages.append({"role": "user", "content": "What was my lucky number? Reply with only the number."})
    data = call_model(messages)
    total_tokens += data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    final_answer = data["message"]["content"].strip()

    return {
        "final_answer": final_answer,
        "correct": NEEDLE in final_answer,  # Lenient check: needle appears anywhere in the answer
        "total_tokens": total_tokens,
        "compaction_events": compaction_events,
        "final_context_size": context_sizes[-1],  # Pre-compaction size after the last round
        "peak_context_size": max(context_sizes),
    }

if __name__ == "__main__":  # Only runs when executed directly, not when imported
    # Baseline: history grows unbounded. Ollama may silently truncate the oldest messages
    # (including the needle) once its context window (often 2048–4096 tokens) is exceeded.
    print("Running WITHOUT compaction (50 rounds)...")
    no_compact = run_scenario(use_compaction=False)

    # Compaction: history periodically summarized; needle survives only if every summary keeps it.
    print("Running WITH compaction (50 rounds)...")
    with_compact = run_scenario(use_compaction=True)

    # Print both result dicts side by side for comparison
    print("\n--- NO COMPACTION ---")
    for k, v in no_compact.items():
        print(f"  {k}: {v}")

    print("\n--- WITH COMPACTION ---")
    for k, v in with_compact.items():
        print(f"  {k}: {v}")
