# score.py
def normalize(s):
    return (s or "").lower().strip().replace("-", " ")

def title_match(predicted, expected):
    if expected is None:
        return True  # no ground truth to check against
    if not predicted:
        return False
    return normalize(expected) in normalize(predicted) or normalize(predicted) in normalize(expected)

def company_match(predicted, expected):
    return normalize(predicted) == normalize(expected)

def skills_match(predicted, expected):
    if not expected:
        return True  # nothing to find
    pred_set = {normalize(s) for s in (predicted or [])}
    exp_set = {normalize(s) for s in expected}
    overlap = pred_set & exp_set
    return len(overlap) / len(exp_set) >= 0.5  # at least half the expected skills found

def salary_match(predicted_min, predicted_max, expected_min, expected_max):
    return predicted_min == expected_min and predicted_max == expected_max

def score_one(predicted: dict, expected: dict) -> dict:
    return {
        "title": title_match(predicted.get("title"), expected["title"]),
        "company": company_match(predicted.get("company"), expected["company"]),
        "skills": skills_match(predicted.get("skills"), expected["skills"]),
        "salary": salary_match(predicted.get("salary_min"), predicted.get("salary_max"),
                                expected["salary_min"], expected["salary_max"]),
    }