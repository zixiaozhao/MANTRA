
import bleu
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, \
    DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import load_from_disk, load_dataset
from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from tqdm.auto import tqdm
from safetensors.torch import load_file
from torch.utils.data import DataLoader
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# fine-tuning tutorial:
## https://www.kaggle.com/code/lizhecheng/qlora-fine-tune-gpt-neox-20b
## https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=cg3fiQOvmI3Q
## https://colab.research.google.com/github/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb#scrollTo=_TAjrSWSe14q
## https://colab.research.google.com/drive/14xo6sj4dARk8lXZbOifHEn1f_70qNAwy?usp=sharing
## https://huggingface.co/blog/peft
## https://github.com/ragntune/code-llama-finetune/tree/main?tab=readme-ov-file
def merge_columns(example):
    example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
    return example


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # print(name, param.device)
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def tokenize_function(samples):
    return tokenizer(samples['problem'])  # + samples['answer'],
    # padding="max_length", max_length=4000)  #, padding=True, truncation=True, max_length=128)


def tokenize_function2(samples):
    print(samples)

    # Return the concatenated text in a dict format
    return {'concatenated_text': concatenated_text}


# Tokenizer function 1: concatenating question and answer
max_source_length = 524
max_target_length = 128
max_seq_length = max_source_length + max_target_length


def tokenize_v1(examples):
    # ───────────────── prompt ─────────────────
    sources = [
        "### Summarize the following code:\n"
        + " ".join(ctokens)
        + "\n### Summary:\n"          # ← 仍保留自然语言提示
        for ctokens in examples["code_tokens"]
    ]
    targets = [" ".join(dtokens) for dtokens in examples["docstring_tokens"]]

    # 1) 分别 encode（右截断即可，左侧永不裁掉 prompt）
    src_enc = tokenizer(
        sources, max_length=max_source_length,
        truncation=True, padding=False,
    )
    tgt_enc = tokenizer(
        targets, max_length=max_target_length,
        truncation=True, padding=False,
    )

    input_ids, attention_mask, labels = [], [], []
    for src_ids, src_mask, tgt_ids in zip(src_enc["input_ids"],
                                          src_enc["attention_mask"],
                                          tgt_enc["input_ids"]):

        seq  = src_ids + [tokenizer.bos_token_id] + tgt_ids + [tokenizer.eos_token_id]
        attn = src_mask + [1] + [1] * (len(tgt_ids) + 1)
        lab  = [-100] * len(src_ids) + tgt_ids + [tokenizer.eos_token_id]

        # 3) 统一 **左侧 padding** （与推理保持一致）
        pad_len = max_seq_length - len(seq)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id
            seq  = [pad_id] * pad_len + seq
            attn = [0]      * pad_len + attn
            lab  = [-100]   * pad_len + lab
        else:  # 太长就裁掉代码尾部而不是 prompt & <s>
            seq  = seq[-max_seq_length:]
            attn = attn[-max_seq_length:]
            lab  = lab[-max_seq_length:]

        input_ids.append(seq)
        attention_mask.append(attn)
        labels.append(lab)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# Tokenizer function 2: focusing on answer
def tokenize_v2(examples,
                           max_source_length: int = 512,
                           max_target_length: int = 128,
                           max_seq_length:   int = 2048):

    # 
    sources = [
        "### Summarize the following code:\n"
        + " ".join(ctokens)
        + "\n### Summary:\n"
        for ctokens in examples["code_tokens"]
    ]
    targets = [" ".join(dtokens) for dtokens in examples["docstring_tokens"]]


    src_enc = tokenizer(
        sources,
        max_length=max_source_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )
    tgt_enc = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )

    input_ids, attention_mask, labels = [], [], []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # ────────────── 3) 拼接 prompt + EOS + target + EOS ──────────────
    for src_ids, src_mask, tgt_ids in zip(
        src_enc["input_ids"], src_enc["attention_mask"], tgt_enc["input_ids"]
    ):
        seq  = src_ids + [eos_id] + tgt_ids + [eos_id]
        attn = src_mask + [1]     + [1] * (len(tgt_ids) + 1)
        lab  = [-100] * len(src_ids) + tgt_ids + [eos_id]

        # ────────── 4) 右侧 padding（或裁掉尾部） ──────────
        pad_len = max_seq_length - len(seq)
        if pad_len > 0:                                  # pad 到固定长度
            seq  = seq  + [pad_id] * pad_len
            attn = attn + [0]      * pad_len
            lab  = lab  + [-100]   * pad_len
        else:                                            # 太长 → 截尾
            seq, attn, lab = seq[:max_seq_length], attn[:max_seq_length], lab[:max_seq_length]

        input_ids.append(seq)
        attention_mask.append(attn)
        labels.append(lab)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_v3(
    examples,
    tokenizer,
    max_source_length: int = 512,
    max_target_length: int = 128,
    max_seq_length:   int = 2048,
):

    # 1) prompt & target
    src_texts = [
        "### Summarize the following code:\n"
        + " ".join(ct)
        + "\n### Summary:\n"
        for ct in examples["code_tokens"]
    ]
    tgt_texts = [" ".join(dt) for dt in examples["docstring_tokens"]]

    # 2) 独立编码（禁自动特殊符号）
    src_enc = tokenizer(
        src_texts,
        max_length=max_source_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )
    tgt_enc = tokenizer(
        tgt_texts,
        max_length=max_target_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )

    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id   # CodeGen: 50256
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    input_ids, attn, labels = [], [], []

    for src_ids, src_mask, tgt_ids in zip(
        src_enc["input_ids"], src_enc["attention_mask"], tgt_enc["input_ids"]
    ):
        # 3) 拼接：BOS + prompt + EOS + target + EOS
        seq  = [bos_id] + src_ids + [eos_id] + tgt_ids + [eos_id]
        mask = [1]      + src_mask + [1]     + [1] * (len(tgt_ids) + 1)

        # label：遮掉 prompt 与 BOS
        lab  = [-100] * (len(src_ids) + 2) + tgt_ids + [eos_id]

        # 4) 左侧 padding / 超长截尾（留最右边）
        pad_len = max_seq_length - len(seq)
        if pad_len > 0:
            seq  = [pad_id] * pad_len + seq
            mask = [0]      * pad_len + mask
            lab  = [-100]   * pad_len + lab
        else:
            seq, mask, lab = seq[-max_seq_length:], mask[-max_seq_length:], lab[-max_seq_length:]

        input_ids.append(seq)
        attn.append(mask)
        labels.append(lab)

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}



parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, help='Path to the model', required=True)
parser.add_argument('--finetune_method', type=str, default='lora', help='fine-tuning method')
parser.add_argument("-m", "--model", help="LLM", type=str)
parser.add_argument('--use_int8', action='store_true', help='whether to use int8 quantization')
parser.add_argument('--use_fp16', action='store_true', help='whether to use fp16 precision')
parser.add_argument("--train_dataset_path", help="dataset_path", type=str, required=True)
parser.add_argument("--finetuned_model_path", help="finetuned_model_path", type=str, required=True)
parser.add_argument('--checkpoint', type=str, default="", help='checkpoint file')
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument('-bs', '--per_device_train_batch_size', type=int, default=32)

parser.add_argument('--tokenize_version', type=int, choices=[1, 2, 3, 4], required=True,
                    help='Select which tokenize function to use: 1, 2, or 3')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    device_map='auto',
)

if args.tokenize_version == 1:
    tokenize_fn = tokenize_v1
    tokenizer.pad_token_id = 0                         # id 0
    tokenizer.pad_token   = tokenizer.unk_token        # "<unk>"
    tokenizer.truncation_side = "right"
    tokenizer.padding_side    = "left"
    tgt = ["q_proj","k_proj","v_proj","o_proj"]
elif args.tokenize_version == 2:
    tokenize_fn = tokenize_v2
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tgt = ["q_proj","k_proj","v_proj","o_proj"]
elif args.tokenize_version == 3:
    tokenize_fn = tokenize_v1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2) 如果没有 bos_token，就也让它等于 eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # 3) 确保左侧填充、右侧截断
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "right"
    tgt = ["q_proj","k_proj","v_proj","o_proj"]

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


offload_folder = "offload_folder"

if args.use_int8:
    print("**********************************")
    print("**** Using 8-bit quantization ****")
    print("**********************************")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=True,
        device_map="auto",
        cache_dir=HF_HOME,
        offload_folder=offload_folder,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
# if specified, use fp16 precision
elif args.use_fp16:
    print("**********************************")
    print("****** Using fp16 precision ******")
    print("**********************************")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=HF_HOME,
        offload_folder=offload_folder,
        local_files_only=True,
    )
# otherwise, use default precision
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        load_in_8bit=True,
        local_files_only=True,
        # cache_dir=HF_HOME,
        # offload_folder=offload_folder,
    )


raw = load_dataset(
    'json',
    data_files=args.train_dataset_path,
    cache_dir='/scratch/zixiao'
)['train']

# 2) 第一次切分：先划出 20% 做“临时评估集”
splits = raw.train_test_split(test_size=0.2, seed=42)
raw_train = splits['train']  # 80%
temp_eval = splits['test']  # 20%

# 3) 第二次切分：把 20% 的临时集再一分为二，各 10%
tmp = temp_eval.train_test_split(test_size=0.5, seed=42)
raw_valid = tmp['train']  # 10%
raw_test = tmp['test']  # 10%

print(f"Train: {len(raw_train)}, Valid: {len(raw_valid)}, Test: {len(raw_test)}")

train_dataset = raw_train.map(tokenize_v1, batched=True, remove_columns=raw_train.column_names)
val_dataset = raw_valid.map(tokenize_v1, batched=True, remove_columns=raw_valid.column_names)
test_dataset = raw_test.map(tokenize_v1, batched=True, remove_columns=raw_test.column_names)

# Check the size of the datasets
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Inspect a sample from the train and validation datasets
print(train_dataset[0])
print(val_dataset[0])

resume_from_checkpoint = args.checkpoint  # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        # adapters_weights = torch.load(resume_from_checkpoint)
        # set_peft_model_state_dict(model, adapters_weights)
        # model = PeftModel.from_pretrained(model, resume_from_checkpoint)

        # Load the safetensors file
        # adapters_weights = load_file(resume_from_checkpoint)

        # Set the weights in the model
        # set_peft_model_state_dict(model, adapters_weights)
        peft_config = PeftConfig.from_pretrained(resume_from_checkpoint)
        model = PeftModel.from_pretrained(model, model_id=resume_from_checkpoint)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

model.train()  # put model back into training mode
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=tgt,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

print_trainable_parameters(model)

# Define the Trainer
batch_size = 32
per_device_train_batch_size = 4
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = args.output_dir  # "code-llama-fine-tuned-v1"

training_args = TrainingArguments(
    # per_device_train_batch_size=4,
    # gradient_accumulation_steps=4,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    num_train_epochs=3,  # changed back from 2000 to 500, given previously best results (exp6) used 500 instead of 2000
    learning_rate=1e-5,  # 5e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",  # if val_set_size > 0 else "no",
    save_strategy="epoch",
    output_dir=output_dir,
    # save_total_limit=3,
    load_best_model_at_end=False,
    # ddp_find_unused_parameters=False if ddp else None,
    group_by_length=True,  # group sequences of roughly the same length together to speed up training
    report_to="none",  # if use_wandb else "none",
    run_name=None,  # if use_wandb else None,
    remove_unused_columns=True,
)

first_trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
# Commenting these lines to fix the loading safetensor file error issue (https://github.com/slai-labs/get-beam/issues/94)
# model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
#    model, type(model)
# )
#if torch.__version__ >= "2" and sys.platform != "win32":
#    print("compiling the model")
#    model = torch.compile(model)
print("=== Phase 1: Training ===")
first_trainer.train()

print("=== Computing per-sample loss on train set ===")
model.eval()
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)
loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collator
)
all_losses = []
device = model.device
for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # shift
        logits = outputs.logits[..., :-1, :].contiguous()
        labels_shift = batch["labels"][..., 1:].contiguous()
        # flatten
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels_shift.view(-1)
        # per-token loss
        loss_flat = F.cross_entropy(
            flat_logits, flat_labels,
            reduction="none",
            ignore_index=-100
        ).view(logits.size(0), -1)
        mask = (labels_shift != -100).float()
        per_sample = (loss_flat * mask).sum(dim=1) / mask.sum(dim=1)
    all_losses.extend(per_sample.cpu().tolist())

print(f"Computed per-sample loss for {len(all_losses)} samples.")
print(all_losses)
#import json
#loss_file = os.path.join(args.output_dir, "train_losses.json")
#with open(loss_file, "w") as f:
#    json.dump(all_losses, f)
#print(f"Saved per-sample losses to {loss_file}")

print("=== Fitting Gaussian Mixture Model on per-sample losses ===")
import numpy as np
from sklearn.mixture import GaussianMixture

loss_array = np.array(all_losses).reshape(-1, 1)
gmm = GaussianMixture(n_components=2, random_state=42).fit(loss_array)
probs = gmm.predict_proba(loss_array)

means = gmm.means_.flatten()
normal_comp = np.argmin(means)
normal_probs = probs[:, normal_comp]

keep_mask = normal_probs >= 0.01
keep_indices = np.where(keep_mask)[0].tolist()

print(f"Filtered out {len(all_losses) - len(keep_indices)} noisy samples (>=99% abnormal cluster probability)")
print(f"Retaining {len(keep_indices)}/{len(all_losses)} samples (<99% abnormal probability)")


filtered_train_dataset = train_dataset.select(keep_indices)

training_args_2 = TrainingArguments(
    # per_device_train_batch_size=4,
    # gradient_accumulation_steps=4,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    num_train_epochs=7,  # changed back from 2000 to 500, given previously best results (exp6) used 500 instead of 2000
    learning_rate=1e-5,  # 5e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",  # if val_set_size > 0 else "no",
    save_strategy="epoch",
    output_dir=output_dir,
    # save_total_limit=3,
    load_best_model_at_end=False,
    # ddp_find_unused_parameters=False if ddp else None,
    group_by_length=True,  # group sequences of roughly the same length together to speed up training
    report_to="none",  # if use_wandb else "none",
    run_name=None,  # if use_wandb else None,
    remove_unused_columns=True,
)

print("=== Phase 2: Training ===")
second_trainer = Trainer(
    model=model,
    train_dataset=filtered_train_dataset,
    eval_dataset=val_dataset,
    args=training_args_2,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
second_trainer.train()
second_trainer.evaluate()


model.save_pretrained(args.finetuned_model_path)
model.save_pretrained(args.finetuned_model_path + '-bin', safe_serialization=False)

tokenizer.padding_side = "left"
batch = tokenizer("Two things are infinite: ", return_tensors='pt').to(model.device)
with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)
print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
