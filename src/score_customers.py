"""Score cardmembers and select the top 20%, without using IDs as predictors.

Usage:
    python -m src.score_customers --data sample_data/synthetic_cardmembers.csv --output outputs/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from .profitability import ACTIVE_FEATURES, to_number, score_customer

def rank_customers(data_path, fraction=0.20):
    """Return (customer_id, predicted_top_segment) in original input order."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    with open(data_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError("Missing id column")
        absent = set(ACTIVE_FEATURES) - set(reader.fieldnames)
        if absent:
            raise ValueError(f"Missing features: {sorted(absent)}")
        risks = [to_number(r["f11"]) for r in reader if r["f11"] and r["f11"].strip()]
    if not risks:
        raise ValueError("No observed risk values for median imputation")
    risk_median = median(risks)
    records = []
    seen = set()
    with open(data_path, encoding="utf-8-sig", newline="") as f:
        for index, row in enumerate(csv.DictReader(f)):
            cid = row["id"].strip()
            if not cid or cid in seen:
                raise ValueError(f"Blank or duplicate ID at row {index+1}")
            seen.add(cid)
            score = score_customer(row, risk_median=risk_median)
            if not math.isfinite(score):
                raise ValueError(f"Invalid score for {cid}")
            records.append((index, cid, score, to_number(row.get("f1")), to_number(row.get("f5"))))
    n = math.ceil(len(records)*fraction)
    # Tie breaking uses revolve, spend, then input order; IDs are never model features.
    selected = sorted(records, key=lambda r:(-r[2],-r[3],-r[4],r[0]))[:n]
    winners = {r[1] for r in selected}
    return [(r[1], int(r[1] in winners)) for r in records]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="outputs/predictions.csv")
    p.add_argument("--fraction", type=float, default=.20)
    a = p.parse_args()
    ranked = rank_customers(a.data, a.fraction)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["ID", "Prediction"])
        w.writerows(ranked)
    print(f"Scored {len(ranked):,} records; selected {sum(v for _,v in ranked):,}.")

if __name__ == "__main__":
    main()
