import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class RoBERTa(nn.Module):
    def __init__(
        self,
        name: str = "roberta-base",
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")

    def forward(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        outputs = self.model(**inputs)
        cls_state = outputs.last_hidden_state[:, 0]
        return cls_state
