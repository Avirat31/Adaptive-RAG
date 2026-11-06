import json
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)

parser = argparse.ArgumentParser()
parser.add_argument("predictions_file", type=Path)
args = parser.parse_args()

path = args.predictions_file

rows = [json.loads(line) for line in path.open() if line.strip()]
labels = ["yes", "no", "maybe"]

gold = [r["gold"].strip().lower() for r in rows]
pred = [r["answer"].strip().lower() for r in rows]

print(f"\nN = {len(rows)}")
print(f"Accuracy = {accuracy_score(gold, pred):.4f}")
print(f"Macro-F1 = {f1_score(gold, pred, labels=labels, average='macro', zero_division=0):.4f}")

print("\n=== Gold distribution ===")
print(Counter(gold))

print("\n=== Prediction distribution ===")
print(Counter(pred))

print("\n=== Confusion matrix (rows=gold, columns=prediction) ===")
print("labels:", labels)
print(confusion_matrix(gold, pred, labels=labels))

print("\n=== Per-class report ===")
print(classification_report(gold, pred, labels=labels, zero_division=0))

print("\n=== Error counts by route ===")
route_stats = defaultdict(lambda: [0, 0])
for r in rows:
    route = r.get("route", "unknown")
    route_stats[route][1] += 1
    route_stats[route][0] += r["gold"].lower() != r["answer"].lower()
for route, (errors, total) in route_stats.items():
    print(f"{route}: {errors}/{total} wrong ({errors/total:.1%})")

print("\n=== Error counts by stop reason ===")
stop_stats = defaultdict(lambda: [0, 0])
for r in rows:
    stop = r.get("evidence_state", {}).get("stop_reason", "unknown")
    stop_stats[stop][1] += 1
    stop_stats[stop][0] += r["gold"].lower() != r["answer"].lower()
for stop, (errors, total) in stop_stats.items():
    print(f"{stop}: {errors}/{total} wrong ({errors/total:.1%})")

print("\n=== Incorrect examples (up to 20) ===")
errors = [r for r in rows if r["gold"].lower() != r["answer"].lower()]
for i, r in enumerate(errors[:20], 1):
    state = r.get("evidence_state", {})
    print(f"\n[{i}] qid={r.get('qid')} | gold={r['gold']} | pred={r['answer']}")
    print(f"Route={r.get('route')} | iterations={r.get('iterations')} | stop={state.get('stop_reason')}")
    print("Question:", r.get("question"))
    print("Explanation:", r.get("explanation"))
    print("Covered:", state.get("covered"))
    print("Unresolved:", state.get("unresolved"))
    print("Retrieved:", len(r.get("passages", [])))
