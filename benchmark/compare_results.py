# benchmark/compare_results.py
import json
from tabulate import tabulate

with open("benchmark/traditional_results.json") as f:
    traditional = json.load(f)

with open("benchmark/brags_results.json") as f:
    brags = json.load(f)

table = [
    ["Search Method", "Best Accuracy", "Best Params", "Total Time (s)"],
    ["Traditional", traditional["best_score"], traditional["best_params"], traditional["total_time"]],
    ["BRAGS", brags["best_score"], brags["best_params"], brags["total_time"]]
]

print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
