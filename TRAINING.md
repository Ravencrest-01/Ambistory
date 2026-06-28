# Training guide (DeBERTa-v3, default backbone)

The default transformer backbone is **`microsoft/deberta-v3-base`** — the
strongest encoder for sentence-pair regression that fits a 6 GB GPU.

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
DeBERTa-v3's tokenizer needs `sentencepiece` and `protobuf` (both pinned above).

## 2. Train (6 GB laptop GPU, e.g. RTX 3060)
```bash
python train.py --model transformer --train data/train.json \
    --dev data/dev.json --output model --epochs 3 \
    --batch-size 8 --grad-accum 2
```
- `--batch-size 8 --grad-accum 2` keeps DeBERTa-v3 in 6 GB with an effective
  batch of 16. If you hit CUDA out-of-memory, use `--batch-size 4 --grad-accum 4`.
- Precision is bf16 when the GPU supports it, else fp32. fp16 is never used (it
  causes NaN losses with DeBERTa-v3 and ModernBERT).
- With `--dev`, the best epoch by eval loss is kept automatically.
- Output: a checkpoint in `./model`, picked up automatically by `predict.py`.

## 3. Predict & evaluate
```bash
python predict.py data/dev.json predictions.jsonl
python evaluate.py --gold data/dev.json --pred predictions.jsonl
```

## Other backbones to compare (single flag)
```bash
python train.py --model transformer --base-model roberta-base               # fast, known-good baseline
python train.py --model transformer --base-model cross-encoder/nli-deberta-v3-base
```

## Tuning notes
- Start at 3 epochs; DeBERTa-v3 over-fits quickly on this small dataset.
- If loss is jumpy, lower the LR: `--lr 1e-5`.
- ModernBERT is intentionally not the default — it produced NaN losses on this
  setup even under bf16.
