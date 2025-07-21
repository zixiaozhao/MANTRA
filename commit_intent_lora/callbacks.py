import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import TrainerCallback, default_data_collator

class CollatorWithId:
    def __init__(self, tokenizer, model=None, pad_to_multiple_of: int = None):
        self.tokenizer = tokenizer
        self.model = model
        self.inner = default_data_collator

    def __call__(self, features):
        batch = self.inner(features)
        batch['id'] = torch.tensor([f['id'] for f in features], dtype=torch.long)
        return batch

class TrainPerSampleLossCallback(TrainerCallback):
    def __init__(self, tokenizer, model, train_dataset, output_dir, device):
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = train_dataset
        self.output_dir = output_dir
        self.device = device
        os.makedirs(self.output_dir, exist_ok=True)

        self.collator = CollatorWithId(tokenizer, model=model, pad_to_multiple_of=8)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collator,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        print(f"\n>>> Epoch {epoch}: computing per-sample training loss …")
        self.model.eval()

        ids, losses = [], []
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="  train loss"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                sample_id = int(batch["id"].item())
                sample_loss = float(outputs["loss"].item())
                ids.append(sample_id)
                losses.append(sample_loss)

        df = pd.DataFrame({"id": ids, "loss": losses})
        file_path = os.path.join(self.output_dir, f"train_losses_epoch{epoch}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved per-sample train losses: {file_path}")
        self.model.train()
