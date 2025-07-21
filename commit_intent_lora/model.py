import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from utils import NUM_LABELS

class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, lora_config: LoraConfig = None):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True)

        if lora_config:
            base_model = get_peft_model(base_model, lora_config)

        self.backbone = base_model
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled = outputs.hidden_states[-1][:, -1, :]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

    def resize_token_embeddings(self, new_size):
        self.backbone.resize_token_embeddings(new_size)
