import argparse, os, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, TaskType
from sklearn.mixture import GaussianMixture
from model import MultiLabelClassifier
from dataset import preprocess_commit
from utils import compute_metrics
from callbacks import CollatorWithId

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path",  required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--finetuned_model_path",  type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ------------------ Setup ------------------
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, lora_alpha=16, lora_dropout=0.05,
        bias="none", target_modules=["q_proj","k_proj","v_proj","o_proj"]
    )
    model = MultiLabelClassifier(args.model_name_or_path, peft_config)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_with_id(example, idx):
        ex = preprocess_commit(example, tokenizer, args.max_length)
        ex["id"] = idx
        return ex

    # ------------- Load & Split -------------
    raw = load_dataset("csv", data_files=args.train_path)["train"]
    splits = raw.train_test_split(test_size=0.1, seed=args.seed)
    raw_train, raw_valid = splits['train'], splits['test']

    train_data = raw_train.map(preprocess_with_id, with_indices=True)
    eval_data  = raw_valid.map(preprocess_with_id, with_indices=True)

    keep_cols = ["input_ids", "attention_mask", "labels", "id"]
    train_data = train_data.remove_columns([c for c in train_data.column_names if c not in keep_cols])
    eval_data  = eval_data.remove_columns([c for c in eval_data.column_names if c not in keep_cols])

    # ------------- Training Setup -------------
    batch_size = 8
    acc_steps = batch_size // args.per_device_train_batch_size
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=acc_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # ---------------- Phase 1: Train ----------------
    trainer.train()
    results = trainer.evaluate()
    print("Phase1 Evaluation:", results)

    # ---------------- Phase 1: Test ----------------
    raw_test = load_dataset("csv", data_files=args.test_path)["train"]
    test_data = raw_test.map(preprocess_with_id, with_indices=True)
    test_data = test_data.remove_columns([c for c in test_data.column_names if c not in keep_cols])
    test_pred = trainer.predict(test_data)
    print("Phase1 Test metrics:", test_pred.metrics)

    # ---------------- Phase 2: GMM Filter ----------------
    print("=== Collecting per-sample losses ===")
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    loader = DataLoader(
        train_data.remove_columns("id"),
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    device = next(model.parameters()).device

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch)["logits"]        # (B, num_labels)
            loss_mat = loss_fn(logits, batch["labels"].float())  # (B, num_labels)
            per_sample = loss_mat.mean(dim=1)    # (B,)
        losses.extend(per_sample.cpu().tolist())

    loss_path = Path(args.output_dir) / "train_losses_phase1.json"
    json.dump(losses, open(loss_path, "w"))

    # GMM filter
    arr = np.asarray(losses).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=args.seed).fit(arr)
    probs = gmm.predict_proba(arr)
    normal_idx = int(np.argmin(gmm.means_.flatten()))
    keep_mask = probs[:, normal_idx] >= 0.5
    keep_indices = np.where(keep_mask)[0].tolist()

    filtered_train = train_data.select(keep_indices)
    print(f"Kept {len(filtered_train)} / {len(train_data)} samples after GMM filtering")


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=acc_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )
    trainer2 = Trainer(
        model=trainer.model,
        args=training_args,
        train_dataset=filtered_train,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer2.train()

    test_pred2 = trainer.predict(test_data)
    print("Phase2 Test metrics:", test_pred2.metrics)

    os.makedirs(args.finetuned_model_path, exist_ok=True)
    trainer.save_model(args.finetuned_model_path)
    tokenizer.save_pretrained(args.finetuned_model_path)

if __name__ == "__main__":
    main()
