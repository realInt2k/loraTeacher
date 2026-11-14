import json, re

PAT = re.compile(r"\d{2}(?:, \d{2})*|\d{2}(?:,\d{2})*")

with open("results_penguin.json") as f:
    results = json.load(f)

filtered = [r for r in results if PAT.fullmatch(r["output"])]

with open("penguin_filtered.json", "w") as f:
    json.dump(filtered, f, indent=2)

print(f"kept {len(filtered)} of {len(results)}")

