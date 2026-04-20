---
language: English
license: Apache-2.0
tags:
  - multi-label-classification
  - bert
  - community
  - entrepreneurship
  - mentorship
  - startup
metrics:
  - name: eval_micro/f1
    type: evaluation
    value: 1.0
datasets: []
---

# Theme classification model (multi-label)

This model is a fine-tuned BERT (bert-base-uncased) trained to detect short, community-oriented themes in text. It's optimized for small community narratives and program descriptions and predicts one or more of the following labels:

Last updated: 2026-01-12

- mentorship
- entrepreneurship
- startup success

The model and tokenizer are available on the Hugging Face Hub and can be used directly with the Transformers library.

Why this model?

We built a compact, interpretable classifier to help community organizations automatically tag short narratives and program descriptions with relevant themes. This can speed up content categorization, reporting, and downstream search features.

Model details

- Architecture: bert-base-uncased (Hugging Face Transformers)
- Problem type: multi-label classification (sigmoid + thresholding)
- Labels: `mentorship`, `entrepreneurship`, `startup success`
- Training data: small curated JSONL (`train_theme.jsonl`) included in the repo
- Final evaluation (example run):
  - eval_loss: 0.1822
  - eval_micro/f1: 1.0
  - eval_macro/f1: 1.0

Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "4nkh/theme_model"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)

texts = [
    "Our co-op paired first-time founders with veteran shop owners to troubleshoot setbacks and model problem-solving under pressure."
]
inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int()
    print('probs', probs.numpy(), 'preds', preds.numpy())
```

Tips and limitations

- Thresholding: we use 0.5 as a default threshold. For production usage, calibrate thresholds per-label on a hold-out set.
- Small dataset: this model was trained on a relatively small, focused dataset — treat predictions as helpful signals, not authoritative labels.
- Bias & fairness: the training data reflects the cultural and programmatic context it was collected from. Evaluate fairness for your specific application.

Reproducibility

- Training script: `train_theme_model.py` in this folder shows the preprocessing, dataset format, and training configuration.
- Environment: we recommend running the training inside a virtualenv with `transformers`, `datasets`, `torch`, and `huggingface_hub` installed.

License

This model is provided under the Apache-2.0 license. Change the `license` field in the YAML header if you prefer a different license.

Want to re-run or push updates?

Use the included `push_model.py` helper to upload model artifacts and the model card to the Hub. Example:

```bash
# set these env vars first
export HUGGINGFACE_HUB_TOKEN="hf_..."
export HF_REPO_ID="4nkh/theme_model"
python push_model.py
```

If you'd like, I can expand the model card with example prompts, per-class thresholds, or a downloadable demo.

Example outputs

Below is a small example showing predicted probabilities and labels for a sample text using the model as packaged in this repo.

```
Input: "Our co-op paired first-time founders with veteran shop owners to troubleshoot setbacks."
Output: probs [[0.98, 0.12, 0.03]]  preds [[1, 0, 0]]
Meaning: High confidence for `mentorship`.
```

Per-class thresholds

We provide default thresholds used during evaluation in `thresholds.json`. These are conservative defaults (0.5) — you can tune them based on precision/recall tradeoffs on a validation set.

Badges

[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-orange)](https://huggingface.co/4nkh/theme_model)

GitHub

The code, helpers, and a CI workflow are available on GitHub (branch `clean-code`):

[View on GitHub](https://github.com/MyVillage-Project-Technologies/CIC-James-Henson-theme-model/tree/clean-code)

Try it on Hugging Face

You can also use the Hugging Face Inference API for quick tests (see the "Use in Transformers" section above on this model page), or run the included `predict_example.py` locally to reproduce the demo outputs.

