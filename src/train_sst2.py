"""Fine-tune RoBERTa-base on SST-2.

Colab:      python -m src.train_sst2 --output_dir /content/drive/MyDrive/xai/roberta-sst2
Quick test: python -m src.train_sst2 --max_train_samples 2000 --epochs 1 --output_dir /content/test-run
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import ROOT, SEED, SST2_LABELS, load_sst2

MODEL_NAME = "roberta-base"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="where to save the model (e.g. Google Drive)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_train_samples", type=int, default=None, help="for test runs only")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    train, val = load_sst2()
    if args.max_train_samples:
        train = train.shuffle(seed=args.seed).select(range(args.max_train_samples))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=args.max_length)

    train_tok = train.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    val_tok = val.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=SST2_LABELS,
        label2id={v: k for k, v in SST2_LABELS.items()},
    )

    # warm-up: the learning rate grows slowly during the first 6% of the steps
    # (computed here because newer transformers versions removed warmup_ratio)
    steps_per_epoch = math.ceil(len(train_tok) / args.batch_size)
    warmup_steps = int(0.06 * steps_per_epoch * args.epochs)

    # fixed epochs, no model selection on validation (it is used later for explanations)
    training_args = TrainingArguments(
        output_dir="checkpoints/sst2",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Model saved to: {out}")

    result = {
        "model": MODEL_NAME,
        "dataset": "SST-2 (GLUE), validation",
        "n_train": len(train),
        "n_eval": len(val),
        "accuracy": round(metrics["eval_accuracy"], 4),
        "f1": round(metrics["eval_f1"], 4),
        "f1_macro": round(metrics["eval_f1_macro"], 4),
        "eval_loss": round(metrics["eval_loss"], 4),
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    name = "sst2_test_run_metrics.json" if args.max_train_samples else "sst2_model_metrics.json"
    with open(results_dir / name, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()