import json
from pathlib import Path

v2_path = Path(
    "outputs/pubmedqa_100_prompt_v2_restored/"
    "pubmedqa_adaptive_predictions.jsonl"
)
v21_path = Path(
    "outputs/pubmedqa_100_prompt_v2_1/"
    "pubmedqa_adaptive_predictions.jsonl"
)

def load(path):
    rows = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[str(r["qid"])] = r
    return rows

v2 = load(v2_path)
v21 = load(v21_path)

common = sorted(set(v2) & set(v21))
if len(common) != len(v2) or len(common) != len(v21):
    print(f"WARNING: different question IDs. "
          f"v2={len(v2)}, v2.1={len(v21)}, common={len(common)}")

def norm(x):
    return str(x).strip().lower()

both_correct = []
v2_only_correct = []
v21_only_correct = []
both_wrong = []

for qid in common:
    a, b = v2[qid], v21[qid]
    gold = norm(a["gold"])
    pred2 = norm(a["answer"])
    pred21 = norm(b["answer"])

    c2 = pred2 == gold
    c21 = pred21 == gold

    item = {
        "qid": qid,
        "gold": gold,
        "v2": pred2,
        "v2.1": pred21,
        "question": a.get("question", b.get("question", "")),
    }

    if c2 and c21:
        both_correct.append(item)
    elif c2 and not c21:
        v2_only_correct.append(item)
    elif not c2 and c21:
        v21_only_correct.append(item)
    else:
        both_wrong.append(item)

print(f"\nCompared questions: {len(common)}")
print(f"Both correct:       {len(both_correct)}")
print(f"v2 correct only:    {len(v2_only_correct)}")
print(f"v2.1 correct only:  {len(v21_only_correct)}")
print(f"Both wrong:         {len(both_wrong)}")

print("\n=== v2.1 fixed (v2 wrong -> v2.1 correct) ===")
for x in v21_only_correct:
    print(f"\nQID {x['qid']} | gold={x['gold']} | v2={x['v2']} | v2.1={x['v2.1']}")
    print(x["question"])

print("\n=== v2.1 regressed (v2 correct -> v2.1 wrong) ===")
for x in v2_only_correct:
    print(f"\nQID {x['qid']} | gold={x['gold']} | v2={x['v2']} | v2.1={x['v2.1']}")
    print(x["question"])
