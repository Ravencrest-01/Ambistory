"""Transformer cross-encoder regressor (primary method).

Each sample becomes a sentence pair [story] / [sense]; a pretrained encoder
(DeBERTa-v3 by default) with a single regression output is fine-tuned to predict
the average human plausibility, scaled to [0, 1]. torch/transformers are imported
lazily so the module loads even when they are absent.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import dataio

DEFAULT_BASE_MODEL = os.environ.get("AMBISTORY_BASE_MODEL", "microsoft/deberta-v3-base")


def _toUnit(avg: np.ndarray) -> np.ndarray:
    return (avg - dataio.SCALE_LO) / (dataio.SCALE_HI - dataio.SCALE_LO)


def _fromUnit(unit: np.ndarray) -> np.ndarray:
    return unit * (dataio.SCALE_HI - dataio.SCALE_LO) + dataio.SCALE_LO


class PairData:
    def __init__(self, encodings, labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int):
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class TransformerRegressor:
    def __init__(self, baseModel: str = DEFAULT_BASE_MODEL, maxLen: int = 256):
        self.baseModel = baseModel
        self.maxLen = maxLen
        self.tokenizer = None
        self.model = None

    def _encode(self, samples: list[dict[str, Any]]):
        stories = [dataio.buildStory(s) for s in samples]
        senses = [dataio.buildSense(s) for s in samples]
        return self.tokenizer(stories, senses, truncation=True,
                              max_length=self.maxLen, padding="max_length")

    def train(self, trainSamples: list[dict[str, Any]], outputDir: str,
              devSamples: list[dict[str, Any]] | None = None, epochs: int = 4,
              batchSize: int = 16, lr: float = 2e-5, seed: int = 42,
              gradAccum: int = 1) -> "TransformerRegressor":
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  Trainer, TrainingArguments)

        self.tokenizer = AutoTokenizer.from_pretrained(self.baseModel)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.baseModel, num_labels=1, problem_type="regression")

        yTrain = _toUnit(np.array([dataio.getTarget(s) for s in trainSamples], dtype=np.float32))
        trainData = PairData(self._encode(trainSamples), yTrain)

        devData = None
        if devSamples:
            yDev = _toUnit(np.array([dataio.getTarget(s) for s in devSamples], dtype=np.float32))
            devData = PairData(self._encode(devSamples), yDev)

        # bf16 when supported, else full precision. fp16 is avoided entirely:
        # it triggers NaN losses with several modern encoders (ModernBERT,
        # DeBERTa-v3), and bf16 covers every GPU we target.
        cuda = torch.cuda.is_available()
        useBf16 = cuda and torch.cuda.is_bf16_supported()
        haveEval = devData is not None

        args = TrainingArguments(
            output_dir=os.path.join(outputDir, "_checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batchSize,
            per_device_eval_batch_size=batchSize * 2,
            gradient_accumulation_steps=gradAccum,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.06,
            logging_steps=50,
            # With a dev set, keep the lowest-eval-loss epoch rather than the last (over-fit) one.
            eval_strategy="epoch" if haveEval else "no",
            save_strategy="epoch" if haveEval else "no",
            save_total_limit=1,
            load_best_model_at_end=haveEval,
            metric_for_best_model="eval_loss" if haveEval else None,
            greater_is_better=False,
            seed=seed,
            bf16=useBf16,
            report_to=[],
        )

        trainer = Trainer(model=self.model, args=args,
                          train_dataset=trainData, eval_dataset=devData)
        trainer.train()

        os.makedirs(outputDir, exist_ok=True)
        self.model.save_pretrained(outputDir)
        self.tokenizer.save_pretrained(outputDir)
        return self

    @classmethod
    def load(cls, modelDir: str, maxLen: int = 256) -> "TransformerRegressor":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        obj = cls(baseModel=modelDir, maxLen=maxLen)
        obj.tokenizer = AutoTokenizer.from_pretrained(modelDir)
        obj.model = AutoModelForSequenceClassification.from_pretrained(modelDir)
        obj.model.eval()
        return obj

    def predictScores(self, samples: list[dict[str, Any]], batchSize: int = 32) -> np.ndarray:
        import torch
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded or trained.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        preds: list[float] = []
        with torch.no_grad():
            for start in range(0, len(samples), batchSize):
                batch = samples[start:start + batchSize]
                stories = [dataio.buildStory(s) for s in batch]
                senses = [dataio.buildSense(s) for s in batch]
                enc = self.tokenizer(stories, senses, truncation=True,
                                     max_length=self.maxLen, padding=True,
                                     return_tensors="pt").to(device)
                logits = self.model(**enc).logits.squeeze(-1)
                preds.extend(logits.cpu().numpy().reshape(-1).tolist())
        return _fromUnit(np.array(preds, dtype=np.float32))

    def predict(self, samples: list[dict[str, Any]]) -> list[int]:
        return [dataio.clampRound(v) for v in self.predictScores(samples)]
