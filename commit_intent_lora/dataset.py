from transformers import PreTrainedTokenizerBase
from utils import label2id
import torch
import json

def preprocess_commit(example, tokenizer: PreTrainedTokenizerBase, max_length: int):
    max_diff_chars = 900

    diff_str = example["files_diff_dict"]
    if len(diff_str) > max_diff_chars:
        diff_str = diff_str[:max_diff_chars] + "\n... (truncated)\n"

    prompt = (
        "### Instruction:\n"
        "You are given the metadata of a Git commit. "
        "Identify one or more intents that apply from the following list:\n"
        '["Bug", "Refactor", "Deprecation", "Feature", "Merge", "Resource", "Test"]\n'
        "Output your answer as a semicolon-separated list of labels (no trailing semicolon).\n\n"
        "### Commit Metadata:\n"
        f"- Title: {example['Title']}\n"
        f"- Changed files: {example['ChangedFiles']}\n"
        f"- Description: {example['Description']}\n"
        f"- Diff snippet:\n{diff_str}\n\n"
        "### Response:\n"
    )

    labels = [0] * len(label2id)
    for tag in example["Label"].split(";"):
        tag = tag.strip()
        if tag in label2id:
            labels[label2id[tag]] = 1

    tokenized = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    tokenized = {k: v.squeeze() for k, v in tokenized.items()}
    tokenized["labels"] = torch.tensor(labels, dtype=torch.float)
    return tokenized

