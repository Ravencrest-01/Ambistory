#!/usr/bin/env python3
"""Score a JSONL prediction file against gold data using the official metrics.

  python evaluate.py --gold data/dev.json --pred predictions.jsonl
  python evaluate.py --gold data/dev.json          # runs predict.py first
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile

import numpy as np

from src import dataio


def readPreds(path: str) -> dict[str, int]:
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                preds[str(obj["id"])] = int(obj["prediction"])
    return preds


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate AmbiStory predictions.")
    p.add_argument("--gold", default="data/dev.json")
    p.add_argument("--pred", default=None)
    args = p.parse_args()

    gold = dataio.loadSamples(args.gold)

    predPath = args.pred
    if predPath is None:
        predPath = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        print("No --pred given; running predict.py on the gold file...")
        subprocess.run([sys.executable, "predict.py", args.gold, predPath], check=True)

    preds = readPreds(predPath)
    ids = list(gold.keys())
    missing = [i for i in ids if i not in preds]
    if missing:
        print(f"WARNING: {len(missing)} ids missing from predictions (filled with 3).")

    pred = np.array([preds.get(i, 3) for i in ids], dtype=float)
    avg = np.array([gold[i]["average"] for i in ids], dtype=float)
    std = np.array([gold[i]["stdev"] for i in ids], dtype=float)
    res = dataio.evaluate(pred, avg, std)

    print("\n=== AmbiStory evaluation ===")
    print(f"Samples               : {len(ids)}")
    print(f"Spearman correlation  : {res['spearman']:.4f}")
    print(f"Accuracy within stdev : {res['accWithinStdev']:.4f}")
    print(f"MAE                   : {res['mae']:.4f}")


if __name__ == "__main__":
    main()
