import os, json, random
from typing import Dict, List, Any, Iterable

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support

# ------------------
# CONFIG
# ------------------
MODEL_NAME = "bert-base-uncased"
LABELS = ["mentorship", "entrepreneurship", "startup success"]
TEXT_FIELDS = ["original_text", "summary"]  # used only if your rows have those keys
SEED = 42

# Optional quick-run overrides (for smoke tests). Set via environment variables.
# NUM_EPOCHS: override number of training epochs (default 5)
# MAX_EXAMPLES: if >0, only use this many examples from the dataset (default 0 = all)
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
MAX_EXAMPLES = int(os.getenv("MAX_EXAMPLES", "0"))

# If you want to push to HF, set this and login with: huggingface-cli login
PUSH_TO_HUB = True
HF_REPO_ID = "4nkh/theme_model"  # change to your namespace if PUSH_TO_HUB=True

random.seed(SEED)
np.random.seed(SEED)

# ------------------
# DATA PATH
# ------------------
DATA_PATH = "train_theme.jsonl"  # <-- your file

# ------------------
# LOADERS (JSON or JSONL)
# ------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {i}: {e}") from e
    return rows

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_any(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return load_jsonl(path)

    # .json can be either a list OR a dict wrapper
    obj = load_json(path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # wrapper format
        if "knowledge_theme_training_data" in obj and isinstance(obj["knowledge_theme_training_data"], list):
            return obj["knowledge_theme_training_data"]
        # otherwise maybe it's already one row or something else
        raise ValueError("JSON file is a dict but does not contain 'knowledge_theme_training_data' list.")
    raise ValueError("Unsupported JSON format.")

raw_rows = load_any(DATA_PATH)

# If MAX_EXAMPLES is set for a quick smoke test, truncate the loaded rows
if MAX_EXAMPLES > 0:
    raw_rows = raw_rows[:MAX_EXAMPLES]

# ------------------
# NORMALIZE ROWS -> {text, labels}
# Supports:
# 1) JSONL rows like {"text": "...", "label": "entrepreneurship"} (single-label)
# 2) JSON rows like {"original_text": "...", "themes": ["mentorship","..."]} (multi-label)
# ------------------
def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Determine text
    if "text" in row and isinstance(row["text"], str):
        text = row["text"].strip()
    else:
        # concat fields if present
        parts = []
        for k in TEXT_FIELDS:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        # fall back to original_text alone
        if not parts and isinstance(row.get("original_text"), str):
            parts = [row["original_text"].strip()]
        text = " ".join(parts).strip()

    if not text:
        return {}

    # Determine labels (multi-hot)
    y = [0.0] * len(LABELS)

    # Case A: single label field
    if "label" in row and isinstance(row["label"], str):
        lbl = row["label"].strip()
        if lbl in LABELS:
            y[LABELS.index(lbl)] = 1.0

    # Case B: themes list
    themes = row.get("themes")
    if isinstance(themes, list):
        for t in themes:
            if isinstance(t, str) and t in LABELS:
                y[LABELS.index(t)] = 1.0

    return {"text": text, "labels": y}

examples = []
for r in raw_rows:
    ex = normalize_row(r)
    if ex:
        examples.append(ex)

if len(examples) < 2:
    raise ValueError(f"Not enough usable examples after normalization. Got {len(examples)}.")

ds_full = Dataset.from_list(examples).shuffle(seed=SEED)

# ------------------
# SPLIT (80/20)
# ------------------
n = len(ds_full)
n_train = max(1, int(0.8 * n))
ds = DatasetDict(
    train=ds_full.select(range(n_train)),
    validation=ds_full.select(range(n_train, n)) if n_train < n else ds_full.select(range(0, 1)),
)

# ------------------
# TOKENIZE
# ------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tok(batch["text"], truncation=True)

# Keep labels, remove only text
ds = ds.map(tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer=tok)

# ------------------
# MODEL (multi-label)
# ------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
)

model.config.id2label = {i: l for i, l in enumerate(LABELS)}
model.config.label2id = {l: i for i, l in enumerate(LABELS)}

# ------------------
# METRICS
# ------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "micro/precision": float(micro_p),
        "micro/recall": float(micro_r),
        "micro/f1": float(micro_f1),
        "macro/precision": float(macro_p),
        "macro/recall": float(macro_r),
        "macro/f1": float(macro_f1),
    }

# ------------------
# TRAINING ARGS
# ------------------
args = TrainingArguments(
    output_dir="./theme_model_outputs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    # For compatibility across transformers versions and quick smoke tests,
    # avoid forcing load_best_model_at_end (requires eval/save strategy matching).
    load_best_model_at_end=False,
    metric_for_best_model="micro/f1",
    greater_is_better=True,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HF_REPO_ID if PUSH_TO_HUB else None,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())

# Save locally (always)
trainer.save_model("./theme_model_outputs")
tok.save_pretrained("./theme_model_outputs")

if PUSH_TO_HUB:
    trainer.push_to_hub()
