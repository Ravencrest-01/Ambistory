"""Data IO, text construction, and the two official AmbiStory metrics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

SCALE_LO, SCALE_HI = 1, 5


def loadSamples(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def writePredictions(path: str, preds: list[tuple[str, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sampleId, pred in preds:
            f.write(json.dumps({"id": str(sampleId), "prediction": int(pred)}) + "\n")


def buildStory(sample: dict[str, Any]) -> str:
    parts = [sample.get("precontext") or "",
             sample.get("sentence") or "",
             sample.get("ending") or ""]
    return " ".join(p.strip() for p in parts if p.strip())


def buildSense(sample: dict[str, Any]) -> str:
    word = (sample.get("homonym") or "").strip()
    gloss = (sample.get("judged_meaning") or "").strip()
    example = (sample.get("example_sentence") or "").strip()
    text = f"The word '{word}' here means: {gloss}."
    if example:
        text += f" For example: {example}"
    return text


def hasEnding(sample: dict[str, Any]) -> bool:
    return bool((sample.get("ending") or "").strip())


def getTarget(sample: dict[str, Any]) -> float:
    return float(sample["average"])


def clampRound(value: float, lo: int = SCALE_LO, hi: int = SCALE_HI) -> int:
    if np.isnan(value):
        value = (lo + hi) / 2.0
    return int(np.clip(round(float(value)), lo, hi))


def spearmanCorr(pred: np.ndarray, gold: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(pred, gold)
    return float(rho)


def accWithinStdev(pred: np.ndarray, goldAvg: np.ndarray, goldStd: np.ndarray) -> float:
    # Task floors the tolerance at 1 so tight-consensus samples stay reachable.
    tol = np.maximum(goldStd, 1.0)
    return float(np.mean(np.abs(pred - goldAvg) <= tol))


def evaluate(pred: np.ndarray, goldAvg: np.ndarray, goldStd: np.ndarray) -> dict[str, float]:
    return {
        "spearman": spearmanCorr(pred, goldAvg),
        "accWithinStdev": accWithinStdev(pred, goldAvg, goldStd),
        "mae": float(np.mean(np.abs(pred - goldAvg))),
    }
