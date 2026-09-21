"""Deterministic longest-processing-time allocation with indivisible image groups."""
from collections import defaultdict


def make_shards(samples, count=16, image_costs=None):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["image_id"]].append(sample)
    if len(groups) < count:
        raise ValueError(f"Need {count} distinct images for {count} nonempty shards")
    image_costs = image_costs or {}
    costs = {image: float(image_costs.get(image, 1 + len(rows))) for image, rows in groups.items()}
    if any(c <= 0 for c in costs.values()):
        raise ValueError("Image costs must be positive")
    shards, totals = [[] for _ in range(count)], [0.0] * count
    for image in sorted(groups, key=lambda name: (-costs[name], name)):
        shard = min(range(count), key=lambda i: (totals[i], i))
        shards[shard].extend(sorted(groups[image], key=lambda row: row["sample_id"]))
        totals[shard] += costs[image]
    return shards, totals
