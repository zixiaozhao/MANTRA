#!/usr/bin/env python
"""Evaluate a (possibly LoRA/PEFT‑adapted) causal LM on code‑summarization with BLEU‑4.

Usage examples
--------------
1. **Full fine‑tuned checkpoint (no adapter)**
   ````bash
   python evaluate_bleu.py \
     --model_path ./models/CodeLlama-7b-finetuned \
     --test_data ./data/test.jsonl \
     --output_dir ./eval_out
   ````

2. **Adapter (LoRA/PEFT) + base model**
   ````bash
   python evaluate_bleu.py \
     --base_model_path ./models/CodeLlama-7b-hf \
     --adapter_path    ./models/CodeLlama-7b-hf-finetuned-adapter \
     --test_data       ./data/test.jsonl \
     --output_dir      ./eval_out
   ````
"""

import argparse
import os
from typing import List, Optional

import sacrebleu  # pip install sacrebleu
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from tqdm.auto import tqdm

try:
    from peft import PeftModel  # optional dependency; only needed when --adapter_path is given
except ImportError:
    PeftModel = None  # type: ignore

from transformers import DataCollatorWithPadding

def build_collate_fn(tokenizer):
    padder = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def collate(batch):
        # -------- 弹出非张量字段 --------
        gold_text = [sample.pop("gold_text") for sample in batch]

        # -------- 交给官方 padder --------
        batch_padded = padder(batch)

        # -------- 再塞回去 ---------------
        batch_padded["gold_text"] = gold_text

        return batch_padded

    return collate
###############################################################################
# Argument helpers
###############################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Code‑Summarization BLEU Evaluator (supports LoRA/PEFT)")

    # Base & adapter paths (choose one of the two loading modes)
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to a *fully merged* fine‑tuned model directory (HF format).")
    p.add_argument("--base_model_path", type=str, default=None,
                   help="Path to the *base* model checkpoint (required when --adapter_path is used).")
    p.add_argument("--adapter_path", type=str, default=None,
                   help="Path to LoRA/PEFT adapter directory. If provided, the script loads base+adapter.")

    # Data / I/O
    p.add_argument("--test_data", type=str, required=True,
                   help="Path to test *.jsonl containing code_tokens & docstring_tokens.")
    p.add_argument("--output_dir", type=str, default="./outputs")

    # Generation hyper‑params
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_source_length", type=int, default=512)
    p.add_argument("--max_seq_length", type=int, default=768)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--num_beams", type=int, default=4)

    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p

###############################################################################
# Tokenisation for inference
###############################################################################

def _build_prompt(code_tokens: List[str]) -> str:
    return (
        "### Summarize the following code:\n" + " ".join(code_tokens) + "\n### Summary:\n"
    )


def tokenize_infer(examples, tokenizer, max_source_length: int, max_seq_length: int):
    sources = [_build_prompt(ct) for ct in examples["code_tokens"]]
    enc = tokenizer(sources, truncation=True, padding=False, max_length=max_source_length)

    input_ids, attn = [], []
    for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
        pad = max_seq_length - len(ids)
        if pad > 0:
            ids += [tokenizer.pad_token_id] * pad
            mask += [0] * pad
        else:
            ids = ids[:max_seq_length]
            mask = mask[:max_seq_length]
        input_ids.append(ids)
        attn.append(mask)

    gold_text = [" ".join(dt) for dt in examples["docstring_tokens"]]
    out = {"input_ids": input_ids, "attention_mask": attn, "gold_text": gold_text}

    return out

###############################################################################
# Model loading helpers
###############################################################################

def load_model_and_tokenizer(args) -> tuple[AutoTokenizer, torch.nn.Module]:
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # Branch 1: fully‑merged checkpoint provided
    if args.model_path and not args.adapter_path:
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
        return tok, model

    # Branch 2: base + adapter
    if args.adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft not installed but --adapter_path supplied; `pip install peft`. ")
        if args.base_model_path is None:
            raise ValueError("--base_model_path must be provided when using --adapter_path.")

        tok = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, args.adapter_path)
        return tok, model

    raise ValueError("Must supply either --model_path *or* (--base_model_path & --adapter_path)")

###############################################################################
# Main evaluation
###############################################################################

def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model/tokenizer
    tokenizer, model = load_model_and_tokenizer(args)
    model.to(args.device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "right"
    # 2. Prepare test dataset
    raw = load_dataset("json", data_files=args.test_data, split="train")
    # raw = raw.select(range(1000)) 
    test = raw.map(
        tokenize_infer,
        batched=True,
        remove_columns=raw.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": args.max_source_length,
            "max_seq_length": args.max_seq_length,
        },
        desc="Tokenising (inference)",
    )
    collate_fn = build_collate_fn(tokenizer)

    loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,   # ← 换成自定义
        num_workers=4,
        pin_memory=True,
    )


    # 3. Generation loop
    preds, golds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            gen = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                min_new_tokens=5,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
            )

            seq_len = batch["input_ids"].shape[1]
            for i in range(gen.size(0)):
                summary_ids = gen[i, seq_len:].tolist()
                preds.append(tokenizer.decode(summary_ids, skip_special_tokens=True).strip())
                golds.append(batch["gold_text"][i])

    # 4. BLEU & output files
    bleu = sacrebleu.corpus_bleu(preds, [golds]).score
    print(f"\nBLEU‑4: {bleu:.2f}")

    pred_path = os.path.join(args.output_dir, "test.output")
    gold_path = os.path.join(args.output_dir, "test.gold")
    with open(pred_path, "w", encoding="utf-8") as fp, open(gold_path, "w", encoding="utf-8") as fg:
        for i, (p, g) in enumerate(zip(preds, golds)):
            fp.write(f"{i}\t{p}\n")
            fg.write(f"{i}\t{g}\n")
    with open(os.path.join(args.output_dir, "bleu.txt"), "w", encoding="utf-8") as fb:
        fb.write(f"{bleu:.2f}\n")

    print(f"Outputs saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
