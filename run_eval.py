# run_eval.py
import json
import requests
from eval_set import ground_truth
from score import score_one

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

postings = {p["id"]: p for p in json.load(open("job_postings.json"))}

PROMPT_V1 = """Extract the job title, company, skills, and salary range from this posting.
Return JSON only.

Posting:
{text}"""

PROMPT_V2 = """Extract structured data from this job posting.

Return ONLY a JSON object with EXACTLY these keys (no others, no renaming):
- "title": string or null if not stated
- "company": string or null if not stated
- "skills": array of strings (empty array if none mentioned)
- "salary_min": integer (annual, no currency symbols/commas) or null if not stated or not determinable
- "salary_max": integer (annual, no currency symbols/commas) or null if not stated or not determinable

If salary is given as a single number with no range, use it for both salary_min and salary_max.
If salary is in a non-USD currency, still extract the raw numbers as-is into salary_min/salary_max.
If salary is vague ("competitive", "DOE", "negotiable", "ask us"), set both to null.

Posting:
{text}"""

def extract(prompt_template: str, text: str) -> dict:
    r = requests.post(URL, json={
        "model": MODEL, "stream": False, "format": "json",
        "messages": [{"role": "user", "content": prompt_template.format(text=text)}],
    })
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {}

def run_eval(prompt_template: str, label: str) -> dict:
    results = {}
    for item in ground_truth:
        posting = postings[item["id"]]
        predicted = extract(prompt_template, posting["raw_text"])
        results[item["id"]] = {"predicted": predicted, "scores": score_one(predicted, item)}
    return results

def field_accuracy(results: dict, field: str) -> float:
    vals = [r["scores"][field] for r in results.values()]
    return sum(vals) / len(vals)

if __name__ == "__main__":
    print("Running V1...")
    v1_results = run_eval(PROMPT_V1, "V1")
    print("Running V2...")
    v2_results = run_eval(PROMPT_V2, "V2")

    print("\n--- FIELD ACCURACY ---")
    print(f"{'field':10} {'V1':>8} {'V2':>8}")
    for field in ("title", "company", "skills", "salary"):
        print(f"{field:10} {field_accuracy(v1_results, field):>7.0%} {field_accuracy(v2_results, field):>7.0%}")

    print("\n--- CASE-BY-CASE DIFFS (where V1 and V2 disagree) ---")
    for item in ground_truth:
        id_ = item["id"]
        v1_scores = v1_results[id_]["scores"]
        v2_scores = v2_results[id_]["scores"]
        if v1_scores != v2_scores:
            print(f"\n#{id_} — expected: {item}")
            print(f"  V1 predicted: {v1_results[id_]['predicted']}")
            print(f"  V1 scores:    {v1_scores}")
            print(f"  V2 predicted: {v2_results[id_]['predicted']}")
            print(f"  V2 scores:    {v2_scores}")