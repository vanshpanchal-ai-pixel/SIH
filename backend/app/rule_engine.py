"""
Config-driven rule engine. Rules live in rules.json, not in code, so an
amendment to LM(PC) Rules 2011 is a config edit, not a redeploy.
"""

import json
import re
from pathlib import Path

RULES_PATH = Path(__file__).parent / "rules.json"


def load_rules(category: str, rules_path: Path = RULES_PATH) -> dict:
    with open(rules_path, "r") as f:
        all_rules = json.load(f)
    if category not in all_rules:
        raise ValueError(f"No rule-set defined for category '{category}'")
    return all_rules[category]


def check_compliance(extracted_fields: dict, rules: dict) -> dict:
    """
    For each field defined in the rule-set:
      - missing + mandatory     -> FAIL - missing
      - present but bad pattern -> FAIL - wrong format
      - present + valid pattern -> PASS
    Each verdict includes the rule clause reference for the auditable report.
    """
    results = {}
    for field, rule in rules.items():
        value = extracted_fields.get(field)
        clause = rule.get("clause", "")

        if rule["mandatory"] and not value:
            verdict = "FAIL - missing"
        elif value and not re.match(rule["pattern"], value):
            verdict = "FAIL - wrong format"
        else:
            verdict = "PASS"

        results[field] = {
            "value": value,
            "verdict": verdict,
            "clause": clause,
        }
    return results


if __name__ == "__main__":
    from app.extract import extract_fields

    sample_text = (
        "MRP Rs. 45.00 (Incl. of all taxes)  Net Qty: 100g  Mfg Date: 05/2026 "
        "FSSAI Lic No. 12345678901234 "
        "Manufactured by: XYZ Foods Pvt Ltd, Plot 12, Sector 5, Gurgaon "
        "Consumer Care: 1800-123-4567, care@xyzfoods.com"
    )
    category = "Food"
    fields = extract_fields(sample_text, category)
    rules = load_rules(category)
    verdicts = check_compliance(fields, rules)

    print(json.dumps(verdicts, indent=2))

    # Also test a deliberately non-compliant label (missing FSSAI, malformed date)
    print("\n--- Non-compliant sample ---")
    bad_text = "MRP Rs. 45.00  Net Qty: 100g  Mfg Date: May 2026"
    bad_fields = extract_fields(bad_text, category)
    bad_verdicts = check_compliance(bad_fields, rules)
    print(json.dumps(bad_verdicts, indent=2))
