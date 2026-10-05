import requests

CHAT_URL = 'http://localhost:11434/api/chat'
MODEL = 'llama3.2'
PROMPT = 'Suggest me a name for a coffee shop. Answer only with the name.'

def ask(temperature: float) -> str:
    r = requests.post(CHAT_URL, json={
        'model': MODEL,
        'stream': False,
        'messages': [{'role': 'user', 'content': PROMPT}],
        'options': {'temparature': temperature, 'num_predict': 30}
    })
    r.raise_for_status()
    return r.json()['message']['content'].strip()

for temp in (0.0, 1.0):
    outs = [ask(temp) for _ in range(10)] # this equals: outs = []
# for _ in range(10):
#     outs.append(ask(temp))
    print(f"\ntemperature={temp}: {len(set(outs))} distinct of {len(outs)}")
    for o in outs:
        print(f"  {o!r}") # print(o)      # Brew & Bristle
# print(f"{o}")     # Brew & Bristle
# print(f"{o!r}")   # 'Brew & Bristle'    <- quotes included