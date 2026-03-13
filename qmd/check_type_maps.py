#!/usr/bin/env python3
"""Check type_map.raw consistency across all training systems in myinput.json."""

import json
import os

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myinput.json")

with open(input_file) as f:
    data = json.load(f)

systems = data["training"]["training_data"]["systems"]

print(f"{'#':<5} {'n_types':<8} {'type_map':<30} {'directory'}")
print("-" * 120)

type_map_counts = {}
missing = []

for i, sys_dir in enumerate(systems, 1):
    tm_file = os.path.join(sys_dir, "type_map.raw")
    if not os.path.exists(tm_file):
        missing.append(sys_dir)
        print(f"{i:<5} {'MISSING':<8} {'---':<30} {sys_dir}")
        continue
    with open(tm_file) as f:
        types = [line.strip() for line in f if line.strip()]
    key = tuple(types)
    type_map_counts[key] = type_map_counts.get(key, 0) + 1
    print(f"{i:<5} {len(types):<8} {', '.join(types):<30} {sys_dir}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total systems: {len(systems)}")
print(f"Missing type_map.raw: {len(missing)}")
print(f"\nUnique type maps found ({len(type_map_counts)}):")
for types, count in sorted(type_map_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:>4} systems: [{', '.join(types)}] ({len(types)} types)")
