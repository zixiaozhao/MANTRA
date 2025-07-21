import argparse, os
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, TaskType
from model import MultiLabelClassifier
from dataset import preprocess_commit
from utils import compute_metrics
from callbacks import TrainPerSampleLossCallback, CollatorWithId
import torch

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
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj"]
    )
    model = MultiLabelClassifier(args.model_name_or_path, peft_config)
    model.resize_token_embeddings(len(tokenizer)) 
    def preprocess_with_id(example, idx):
        ex = preprocess_commit(example, tokenizer, args.max_length)
        ex["id"] = idx
        return ex

    train_data = load_dataset("csv", data_files=args.train_path)["train"]
    splits    = train_data.train_test_split(test_size=0.1, seed=42)
    raw_train = splits['train']
    raw_valid = splits['test']

    train_data = raw_train.map(preprocess_with_id, with_indices=True)
    eval_data = raw_valid.map(preprocess_with_id, with_indices=True)

    keep_cols = ["input_ids", "attention_mask", "labels", "id"]
    train_data = train_data.remove_columns([c for c in train_data.column_names if c not in keep_cols])
    eval_data = eval_data.remove_columns([c for c in eval_data.column_names if c not in keep_cols])

    batch_size = 8
    per_device_train_bs = args.per_device_train_batch_size
    gradient_accumulation_steps = batch_size // per_device_train_bs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    collator = CollatorWithId(tokenizer, model=model, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            TrainPerSampleLossCallback(
                tokenizer=tokenizer,
                model=model,
                train_dataset=train_data,
                output_dir=args.output_dir,
                device=training_args.device
            )
        ]
    )

    trainer.train()
    results = trainer.evaluate()
    print(results)

    raw_test   = load_dataset("csv", data_files=args.test_path)["train"]
    test_data  = raw_test.map(preprocess_with_id, with_indices=True)
    test_data  = test_data.remove_columns([c for c in test_data.column_names if c not in keep_cols])
    
    test_pred  = trainer.predict(test_data)
    test_metrics = test_pred.metrics
    print("Test metrics:", test_metrics)

    os.makedirs(args.finetuned_model_path, exist_ok=True)
    torch.save(model.state_dict(), args.finetuned_model_path + "/pytorch_model_with_classifier.bin")

    model.backbone.save_pretrained(args.finetuned_model_path)
    tokenizer.save_pretrained(args.finetuned_model_path)
if __name__ == "__main__":
    main()
