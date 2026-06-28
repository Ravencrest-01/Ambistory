# AmbiStory — Rating Word-Sense Plausibility (SemEval-2026 Task 5)

Given a 5-sentence short story containing an ambiguous homonym, and a candidate
word sense for that homonym, predict the **human-perceived plausibility** of the
sense on a 1–5 scale. The task treats plausibility as *graded* rather than
binary: humans disagree, so each sample carries an `average` rating and a
`stdev` over at least five annotators.

Evaluation uses the two official metrics:

- **Spearman correlation** between predictions and the human average.
- **Accuracy within standard deviation** — the share of predictions within one
  stdev of the average (tolerance floored at 1).

## Approach

**Primary method — transformer cross-encoder regression.** Each sample is
encoded as a sentence pair, `[CLS] story [SEP] sense-description [SEP]`, where
the *story* concatenates the precontext, the ambiguous sentence and the
(optional) ending, and the *sense-description* states the gloss being judged
plus a usage example. A pretrained encoder (**DeBERTa-v3** by default; DeBERTa-v3 / RoBERTa
optional) with a single regression output is fine-tuned to predict the average
rating, scaled to `[0, 1]`. Full cross-attention lets the model weigh the
interaction between the narrative — especially the disambiguating ending — and
the candidate sense, which is exactly what determines plausibility. Regressing
on the continuous average (instead of classifying one "correct" sense) matches
the graded nature of the task and the Spearman / within-stdev metrics.

**Fallback method — TF-IDF + LSA gradient boosting.** A dependency-light model
(scikit-learn only, no downloads, no GPU) projects the story and the sense into
a shared latent-semantic space (TF-IDF → TruncatedSVD), then feeds those
vectors, their interaction, and explicit story/ending-vs-sense cosine
similarities to a gradient-boosted regressor. It is both a reproducible
baseline and a safety net: `predict.py` uses it automatically whenever a
transformer checkpoint is unavailable, so the harness always receives valid
output.

## Repository layout

```
.
├── predict.py          # REQUIRED entry point: python predict.py <in.json> <out.jsonl>
├── train.py            # train the transformer or the fallback
├── evaluate.py         # compute the official metrics on a prediction file
├── requirements.txt    # pinned dependencies
├── data/
│   ├── train.json      # 2280 training samples
│   └── dev.json        # 588 dev samples
└── src/
    ├── dataio.py       # loading, text construction, metrics
    ├── features.py     # fallback model (TF-IDF + LSA + gradient boosting)
    └── transformer.py  # primary model (cross-encoder regression)
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

### Predict (the entry point the harness calls)

```bash
python predict.py <input_json> <output_jsonl>
```

Output is JSONL, one object per line: `{"id": "0", "prediction": 3}`, with the
prediction an integer in `[1, 5]`. With no transformer checkpoint present this
runs the fallback (training it on `data/train.json` on first use).

### Train the transformer (recommended before submission)

```bash
python train.py --model transformer --train data/train.json \
    --dev data/dev.json --output model --epochs 4
```

This fine-tunes **DeBERTa-v3** (the default backbone) and writes a checkpoint to
`./model`, which `predict.py` then picks up automatically. When a `--dev` set is
given, the trainer evaluates every epoch and keeps the **best** checkpoint by
eval loss (so it won't ship an over-fit final epoch). Training uses **bf16** when
the GPU supports it and falls back to full precision; fp16 is never used.

Try other backbones with a single flag:

```bash
python train.py --model transformer --base-model microsoft/deberta-v3-base
python train.py --model transformer --base-model cross-encoder/nli-deberta-v3-base
python train.py --model transformer --base-model roberta-base
```

Train only the fallback (no downloads / GPU):

```bash
python train.py --model fallback --output fallback.joblib
```

### Evaluate

```bash
python evaluate.py --gold data/dev.json --pred predictions.jsonl
# or run inference + scoring in one go:
python evaluate.py --gold data/dev.json
```

## Notes & reproducibility

- **No external APIs.** Everything runs locally; the transformer uses a
  HuggingFace checkpoint downloaded once at training time (no API key).
- **Environment overrides:** `AMBISTORY_MODEL_DIR` (transformer checkpoint
  dir, default `model`), `AMBISTORY_BASE_MODEL` (HF backbone; default `microsoft/deberta-v3-base`),
  `AMBISTORY_FALLBACK` (cached fallback path), `AMBISTORY_TRAIN` (training data
  used by the fallback).
- Random seeds are fixed in both models for reproducibility.

### Dev results for the fallback (sanity baseline)

| Metric                | Fallback (TF-IDF+LSA+GBR) |
|-----------------------|---------------------------|
| Spearman              | ~0.11                     |
| Accuracy within stdev | ~0.62                     |

The fallback is intentionally simple; the transformer is expected to improve
substantially on Spearman, since the task hinges on semantic interaction
between narrative and sense that sparse lexical features cannot capture. Fill
in your transformer numbers after running `train.py`.
