#!/usr/bin/env python3
"""Entry point: python predict.py <input_json> <output_jsonl>

Uses the fine-tuned transformer checkpoint when present, otherwise the
scikit-learn fallback (trained on the bundled data on first use), so the harness
always receives valid {"id", "prediction"} JSONL output.
"""

from __future__ import annotations

import os
import sys

from src import dataio
from src.features import FeatureModel

MODEL_DIR = os.environ.get("AMBISTORY_MODEL_DIR", "model")
FALLBACK_PATH = os.environ.get("AMBISTORY_FALLBACK", "fallback.joblib")
TRAIN_DATA = os.environ.get("AMBISTORY_TRAIN", os.path.join("data", "train.json"))


def transformerReady() -> bool:
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def predictWithTransformer(samples):
    from src.transformer import TransformerRegressor
    return TransformerRegressor.load(MODEL_DIR).predict(samples)


def predictWithFallback(samples):
    model = None
    if os.path.exists(FALLBACK_PATH):
        try:
            model = FeatureModel.load(FALLBACK_PATH)
        except Exception as err:
            # Stale or version-mismatched cache: retrain rather than fail.
            sys.stderr.write(f"[warn] could not load cached fallback ({err}); retraining.\n")
    if model is None:
        if not os.path.exists(TRAIN_DATA):
            raise FileNotFoundError(
                f"No transformer checkpoint, no cached fallback, and no training "
                f"data at '{TRAIN_DATA}'.")
        model = FeatureModel().fit(list(dataio.loadSamples(TRAIN_DATA).values()))
        try:
            model.save(FALLBACK_PATH)
        except OSError:
            pass
    return model.predict(samples)


def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python predict.py <input_json> <output_jsonl>\n")
        sys.exit(1)

    inputPath, outputPath = sys.argv[1], sys.argv[2]
    data = dataio.loadSamples(inputPath)
    ids = list(data.keys())
    samples = [data[i] for i in ids]

    if transformerReady():
        try:
            preds = predictWithTransformer(samples)
        except Exception as err:
            sys.stderr.write(f"[warn] transformer failed ({err}); using fallback.\n")
            preds = predictWithFallback(samples)
    else:
        preds = predictWithFallback(samples)

    dataio.writePredictions(outputPath, list(zip(ids, preds)))
    sys.stderr.write(f"Wrote {len(preds)} predictions to {outputPath}\n")


if __name__ == "__main__":
    main()
