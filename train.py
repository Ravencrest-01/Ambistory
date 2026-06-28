#!/usr/bin/env python3
"""Train the transformer (default) or the scikit-learn fallback.

  python train.py --model transformer --dev data/dev.json --output model --epochs 4
  python train.py --model fallback --output fallback.joblib
"""

from __future__ import annotations

import argparse

import numpy as np

from src import dataio
from src.features import FeatureModel


def report(name: str, model, devSamples) -> None:
    if not devSamples:
        return
    scores = model.predictScores(devSamples)
    preds = np.array([dataio.clampRound(v) for v in scores], dtype=float)
    avg = np.array([s["average"] for s in devSamples])
    std = np.array([s["stdev"] for s in devSamples])
    res = dataio.evaluate(preds, avg, std)
    print(f"\n[{name}] dev metrics:")
    print(f"  Spearman (rounded)    : {res['spearman']:.4f}")
    print(f"  Spearman (continuous) : {dataio.spearmanCorr(scores, avg):.4f}")
    print(f"  Accuracy within stdev : {res['accWithinStdev']:.4f}")
    print(f"  MAE                   : {res['mae']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train AmbiStory models.")
    p.add_argument("--model", choices=["transformer", "fallback"], default="transformer")
    p.add_argument("--train", default="data/train.json")
    p.add_argument("--dev", default="data/dev.json")
    p.add_argument("--output", default=None)
    p.add_argument("--base-model", dest="baseModel", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", dest="batchSize", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", dest="maxLen", type=int, default=256)
    p.add_argument("--grad-accum", dest="gradAccum", type=int, default=1)
    args = p.parse_args()

    trainSamples = list(dataio.loadSamples(args.train).values())
    devSamples = list(dataio.loadSamples(args.dev).values()) if args.dev else None
    print(f"Loaded {len(trainSamples)} train"
          + (f" / {len(devSamples)} dev" if devSamples else "") + " samples.")

    if args.model == "fallback":
        out = args.output or "fallback.joblib"
        model = FeatureModel().fit(trainSamples)
        model.save(out)
        print(f"Saved fallback model to {out}")
        report("fallback", model, devSamples)
        return

    from src.transformer import TransformerRegressor, DEFAULT_BASE_MODEL
    out = args.output or "model"
    base = args.baseModel or DEFAULT_BASE_MODEL
    print(f"Fine-tuning {base} -> {out}")
    model = TransformerRegressor(baseModel=base, maxLen=args.maxLen)
    model.train(trainSamples, outputDir=out, devSamples=devSamples,
                epochs=args.epochs, batchSize=args.batchSize, lr=args.lr,
                gradAccum=args.gradAccum)
    print(f"Saved transformer checkpoint to {out}")
    report("transformer", model, devSamples)


if __name__ == "__main__":
    main()
