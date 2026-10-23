
import json
import os
import sys

metrics_file = "metrics.json"

if not os.path.exists(metrics_file):
    print("Quality gate failed: metrics.json not found.")
    sys.exit(1)

with open(metrics_file, "r") as file:
    metrics = json.load(file)

accuracy = metrics.get("accuracy")

if accuracy is None:
    print("Quality gate failed: accuracy is missing.")
    sys.exit(1)

print("Model accuracy:", accuracy)

if accuracy >= 0.50:
    print("Quality gate passed!")
    sys.exit(0)
else:
    print("Quality gate failed: accuracy is below 0.50.")
    sys.exit(1)
