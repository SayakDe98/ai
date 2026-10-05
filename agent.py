import requests

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

# ---- Tool definitions (schema the model sees) ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature in Celsius for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, e.g. '12 * 4'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

# ---- Tool implementations ----
def get_weather(city: str) -> str:
    fake_temps = {"Delhi": 38, "Chennai": 34, "Kolkata": 33}
    if city not in fake_temps:
        return f"error: no weather data for '{city}'"   # deliberately breakable
    return f"{fake_temps[city]}C"

def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"error: {e}"

DISPATCH = {"get_weather": get_weather, "calculate": calculate}

# ---- The loop ----
def run(user_message: str, max_iters: int = 6):
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_iters):
        r = requests.post(URL, json={
            "model": MODEL, "messages": messages,
            "tools": tools, "stream": False,
        })
        r.raise_for_status()
        msg = r.json()["message"]
        messages.append(msg)

        calls = msg.get("tool_calls")
        if not calls:
            return msg["content"]

        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            print(f"  [step {step}] model called {name}({args})")
            result = DISPATCH.get(name, lambda **_: f"error: unknown tool '{name}'")(**args)
            messages.append({"role": "tool", "content": result})

    return "error: hit max iterations without a final answer"

# ---- Interactive loop ----
if __name__ == "__main__":
    print("Ask me something (needs weather or a calculation to use tools). Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        answer = run(user_input)
        print(f"\nModel: {answer}\n")