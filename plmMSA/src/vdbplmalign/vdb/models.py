import torch
import torch.nn as nn
from procl.model.ankh import AnkhCL
from transformers import AutoModel, AutoTokenizer


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def encode(self, input_sequence: str, device: torch.device) -> torch.Tensor:
        raise NotImplementedError("Should implement encode method")


class AnkhModel(BaseModel):
    def __init__(self, model_path, device):
        super().__init__()
        self.tokenizer = self.load_tokenizer("ElnaggarLab/ankh-large")
        self.model = AnkhCL.from_pretrained(
            model_path, freeze_base=True, is_scratch=False
        ).to(device)

    def encode(self, input_sequence: str, device: torch.device):
        results = self.tokenizer(
            input_sequence, truncation=False, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            model_output = self.model(
                input_ids=results["input_ids"], attention_mask=results["attention_mask"]
            )
        sentence_embs = model_output.hidden_states
        input_mask_expanded = (
            results["attention_mask"].unsqueeze(-1).expand(sentence_embs.size()).float()
        )
        sequence_embedding = torch.sum(
            sentence_embs * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sequence_embedding.detach().cpu()


class ESM1bModel(BaseModel):
    def __init__(self, device):
        super().__init__()
        self.tokenizer = self.load_tokenizer("facebook/esm1b_t33_650M_UR50S")
        self.model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(
            device
        )

    def encode(self, input_sequence: str, device: torch.device):
        results = self.tokenizer(
            input_sequence, truncation=False, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            model_output = self.model(
                input_ids=results["input_ids"], attention_mask=results["attention_mask"]
            )
        sentence_embs = model_output.last_hidden_state
        input_mask_expanded = (
            results["attention_mask"].unsqueeze(-1).expand(sentence_embs.size()).float()
        )
        sequence_embedding = torch.sum(
            sentence_embs * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sequence_embedding.detach().cpu()
