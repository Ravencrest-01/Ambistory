"""Fallback regressor: TF-IDF + LSA semantic similarity into gradient boosting.

Dependency-light (scikit-learn only), so it runs with no model download or GPU.
It serves as a baseline and as the safety net behind predict.py, guaranteeing
valid output when torch/transformers are unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import dataio


class FeatureModel:
    def __init__(self, nComponents: int = 100, seed: int = 0):
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                     min_df=2, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=nComponents, random_state=seed)
        self.reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                  max_depth=3, random_state=seed)
        self.fitted = False

    @staticmethod
    def _texts(samples: list[dict[str, Any]]):
        stories = [dataio.buildStory(s) for s in samples]
        senses = [dataio.buildSense(s) for s in samples]
        # Empty endings would yield all-zero TF-IDF rows; the placeholder avoids that.
        endings = [(s.get("ending") or "").strip() or "n/a" for s in samples]
        return stories, senses, endings

    def _embed(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.tfidf.transform(texts)))

    def _features(self, samples: list[dict[str, Any]]) -> np.ndarray:
        stories, senses, endings = self._texts(samples)
        story, sense, ending = self._embed(stories), self._embed(senses), self._embed(endings)
        cosStory = (story * sense).sum(axis=1, keepdims=True)
        cosEnding = (ending * sense).sum(axis=1, keepdims=True)
        return np.hstack([story, sense, story * sense, cosStory, cosEnding]).astype(np.float32)

    def fit(self, samples: list[dict[str, Any]]) -> "FeatureModel":
        stories, senses, _ = self._texts(samples)
        corpus = stories + senses
        self.tfidf.fit(corpus)
        self.svd.fit(self.tfidf.transform(corpus))
        y = np.array([dataio.getTarget(s) for s in samples], dtype=np.float32)
        self.reg.fit(self._features(samples), y)
        self.fitted = True
        return self

    def predictScores(self, samples: list[dict[str, Any]]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureModel must be fitted before prediction.")
        return self.reg.predict(self._features(samples))

    def predict(self, samples: list[dict[str, Any]]) -> list[int]:
        return [dataio.clampRound(v) for v in self.predictScores(samples)]

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureModel":
        import joblib
        return joblib.load(path)
