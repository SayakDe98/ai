import json
import requests

URL = 'http://localhost:11434/api/generate'
MODEL = 'llama3.2'

data = {'name':'Sayak','age':27,'city':'Kolkata','skills':['Go','React','MySQL']}

samples = {
    'A json minified': json.dumps(data, separators=(':',',')),
    'B json pretty': json.dumps(data, indent=4),
    'C english': 'I am Sayak, my age is 27, I live in city Kolkata and my skills are Go, React and MySQL',
    'D code': (
        "def describe():\n"
        "   return {'name':'Sayak','age':27,'city':'Kolkata','skills':['Go','React','MySQL']}"
    )
}

def count_tokens(text: str) -> int:
    r = requests.post(URL, json={
        "model": MODEL,
        "prompt": text,
        "stream": False,
        "options": { "num_predict": 1} # This tells Ollama: generate at most 1 output token, then stop.
    })
    r.raise_for_status() # r is the response object from requests.post(...). raise_for_status() checks the HTTP status code:
    # If it's an error code (4xx or 5xx — bad request, server error, etc.), it raises an exception immediately, with a message telling you what went wrong.
    # If it's a success code (2xx), it does nothing and execution continues.
    # Without this line, if something failed (say, Ollama isn't running, or the model name is wrong), your code would silently continue to r.json()["prompt_eval_count"] and crash with a confusing KeyError instead of a clear "connection refused" or "model not found" error. It's a fail-fast habit: catch the real problem at the source, not three lines later where the symptom shows up.
    return r.json()['prompt_eval_count']

print(f"{'sample':16} {'count':6} {'tokens':7} {'count/tok':>10}") # 'sample' is a literal string, :16 means "pad this to 16 characters wide" (left-aligned by default for strings)
# So 'sample' (6 chars) becomes 'sample          ' (16 chars total), (the > forces right-alignment)
for label, text in samples.items():
    n = count_tokens(text)
    print(f"{label:16} {len(text):6} {n:7} {len(text)/n:10.2f}") # right-aligned in 10 chars, shown as a float with 2 decimal places