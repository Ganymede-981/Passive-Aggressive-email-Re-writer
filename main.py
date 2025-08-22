import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence any stray TF logs

import glob
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from transformers import BertTokenizerFast,BertForSequenceClassification,Trainer,TrainingArguments
from transformers.trainer_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(42)

def load_csvs() -> pd.DataFrame:
    env_path = os.getenv("GOEMOTIONS_CSV", "").strip()
    if env_path and os.path.exists(env_path):
        df = pd.read_csv(env_path)
        print(f"Loaded: {env_path} -> {len(df)} rows")
        return df

    
    paths = sorted(glob.glob("goemotions*.csv")) or sorted(glob.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            "No CSVs found. Place your GoEmotions-style CSV(s) next to this script "
            "or set GOEMOTIONS_CSV to a file path."
        )
    dfs = []
    for p in paths:
        try:
            dfp = pd.read_csv(p)
            dfs.append(dfp)
            print(f"Loaded: {p} -> {len(dfp)} rows")
        except Exception as e:
            print(f"Skipping {p}: {e}")
    if not dfs:
        raise RuntimeError("Found CSV paths but none were readable.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined total rows: {len(combined)}")
    return combined

df = load_csvs()

CAND_TEXT_COLS = ["Text", "text", "sentence", "comment", "content"]
text_col = None
for c in CAND_TEXT_COLS:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    raise KeyError(
        f"Could not find a text column among {CAND_TEXT_COLS}. "
        f"Your columns are: {df.columns.tolist()}"
    )


TARGET_TONES = ["Polite", "Neutral", "Passive-Aggressive"]
TONE2ID: Dict[str, int] = {t: i for i, t in enumerate(TARGET_TONES)}
ID2TONE: Dict[int, str] = {i: t for t, i in TONE2ID.items()}

# Map GoEmotions -> tones (only emotions that actually exist in the CSV will be used)
emotion_to_tone = {
    # Polite
    "admiration": "Polite",
    "approval": "Polite",
    "gratitude": "Polite",
    "caring": "Polite",
    "pride": "Polite",
    # Passive-Aggressive-ish cluster
    "disappointment": "Passive-Aggressive",
    "remorse": "Passive-Aggressive",
    "embarrassment": "Passive-Aggressive",
    "sadness": "Passive-Aggressive",
    # Neutralish cluster
    "neutral": "Neutral",
    "confusion": "Neutral",
    "curiosity": "Neutral",
    "desire": "Neutral",
    "excitement": "Neutral",
    "fear": "Neutral",
    "grief": "Neutral",
    "joy": "Neutral",
    "love": "Neutral",
    "nervousness": "Neutral",
    "surprise": "Neutral",

}

available_emotions = set(df.columns)
filtered_mapping = {e: t for e, t in emotion_to_tone.items() if e in available_emotions}

def infer_tone_from_emotions(row) -> str:
    # Collect all emotions present (one-hot 1s)
    present = [e for e in filtered_mapping.keys() if row.get(e, 0) == 1]
    if not present:
        return "Neutral"
    # First-match policy (you can switch to a priority scheme if needed)
    return emotion_to_tone[present[0]]

if "tone" in df.columns:
    # Use existing tone
    df["tone"] = df["tone"].astype(str)
else:
    # Create tone from emotion one-hots
    df["tone"] = df.apply(infer_tone_from_emotions, axis=1)

# Keeping only rows where tone is in our 3 targets
df = df[df["tone"].isin(TARGET_TONES)].copy()
df[text_col] = df[text_col].astype(str)

if len(df) == 0:
    raise RuntimeError("No rows left after tone filtering into Polite/Neutral/Passive-Aggressive.")

# Encode labels
df["labels"] = df["tone"].map(TONE2ID).astype(int)

df_train, df_val = train_test_split(
    df[[text_col, "labels", "tone"]],
    test_size=0.2,
    random_state=42,
    stratify=df["labels"],
)

print("Train distribution:\n", df_train["tone"].value_counts())
print("Val distribution:\n", df_val["tone"].value_counts())

train_ds = Dataset.from_pandas(df_train[[text_col, "labels"]].reset_index(drop=True))
val_ds   = Dataset.from_pandas(df_val[[text_col, "labels"]].reset_index(drop=True))

MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def preprocess(batch):
    enc = tokenizer(
        batch[text_col],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    enc["labels"] = batch["labels"]
    return enc

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=[text_col])
    val_ds   = val_ds.map(preprocess, batched=True, remove_columns=[text_col])

# Set PyTorch format
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(TARGET_TONES),
    id2label=ID2TONE,
    label2id=TONE2ID,
)
model.to(device)

# Extract labels from HF dataset as a flat numpy array
train_labels_np = train_ds["labels"]  # list of ints
train_labels_np = np.array(train_labels_np, dtype=int)
unique_classes = np.unique(train_labels_np)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=train_labels_np,
)
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
print("Class weights:", class_weights.detach().cpu().numpy())

from torch import nn

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # simple metrics without extra deps
    acc = (preds == labels).mean().item()
    # macro f1
    # (quick, dependency-free macro F1)
    f1s = []
    for c in unique_classes:
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {"accuracy": float(acc), "macro_f1": macro_f1}

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./results",
    save_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

train_out = trainer.train()
print("Best model checkpoint:", trainer.state.best_model_checkpoint)

metrics = trainer.evaluate()
print("Eval metrics:", metrics)

# Sklearn report on val set
pred = trainer.predict(val_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=-1)
print("\nClassification Report (val):\n")
print(classification_report(y_true, y_pred, target_names=[ID2TONE[i] for i in range(len(TARGET_TONES))]))

SAVE_DIR = "./tone_classifier"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved model + tokenizer to {SAVE_DIR}")

def predict_tone(texts: List[str]):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).tolist()
    return [
        {"text": t, "tone": ID2TONE[p], "probs": probs[i].detach().cpu().numpy().round(3).tolist()}
        for i, (t, p) in enumerate(zip(texts, preds))
    ]

demo_texts = [
    "Thanks for the update, I appreciate your quick response.",
    "Sure, because missing the deadline again is exactly what we needed.",
    "Could you please share the latest figures when you get a chance?",
]
print("\nDemo predictions:")
for row in predict_tone(demo_texts):
    print(row)
