import json
import requests
from pydantic import BaseModel, ValidationError
from typing import Optional

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

class JobExtraction(BaseModel):
    title: Optional[str]
    company: Optional[str]
    skills: list[str]
    salary_min: Optional[int]
    salary_max: Optional[int]

PROMPT_V1 = """Extract the job title, company, skills, and salary range from this posting.
Return JSON only.

Posting:
{text}"""

PROMPT_V2 = """Extract structured data from this job posting.

Return ONLY a JSON object with EXACTLY these keys (no others, no renaming):
- "title": string or null if not stated
- "company": string or null if not stated
- "skills": array of strings (empty array if none mentioned)
- "salary_min": integer (annual USD, no currency symbols/commas) or null if not stated or not determinable
- "salary_max": integer (annual USD, no currency symbols/commas) or null if not stated or not determinable

If salary is given as a single number with no range, use it for both salary_min and salary_max.
If salary is in a non-USD currency, still extract the raw numbers as-is into salary_min/salary_max — do not convert or guess an exchange rate.
If salary is vague ("competitive", "DOE", "negotiable", "ask us"), set both to null.

Posting:
{text}"""

def extract(text: str) -> dict:
    r = requests.post(URL, json={
        "model": MODEL,
        "stream": False,
        "format": "json",   # ask Ollama to constrain to valid JSON
        "messages": [{"role": "user", "content": PROMPT_V2.format(text=text)}],
    })
    r.raise_for_status()
    # print(r.json()["message"]["content"])
    return json.loads(r.json()["message"]["content"])

postings = json.load(open("job_postings.json"))
failures = []

for p in postings:
    try:
        raw = extract(p["raw_text"])
        JobExtraction(**raw) # equivalent to: JobExtraction(title=raw["title"], company=raw["company"], skills=raw["skills"], ...)
    except (ValidationError, json.JSONDecodeError, KeyError) as e:
        failures.append((p["id"], str(e)[:150])) #[:150] is Python's slice syntax, and here it means "give me characters 0 through 149 of this string" — i.e., truncate it to the first 150 characters.

print(f"\n{len(failures)} / {len(postings)} failed validation ({len(failures)/len(postings):.0%})")
for id_, err in failures[:10]: #[:10] takes just the first 10 elements of that list — again, purely to keep printed output short rather than dumping all 50 failures to the screen.
    print(f"  #{id_}: {err}") #_id instead of id to prevent collision with an inbuilt((it returns an object's memory identity)) function of python