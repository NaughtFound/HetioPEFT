from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

from hetiopeft.utils import PersistMixin


class PEFTFeatureExtractor(nn.Module, PersistMixin):
    """PEFT / LoRA adapted language model supporting fine-tuning and extraction."""

    def __init__(
        self,
        model_name: str,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize backbone wrapped with PEFT LoRA adapters as an nn.Module."""
        super().__init__()
        self.model_name = model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base_model = AutoModel.from_pretrained(model_name)

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
        )

        self.model = get_peft_model(base_model, peft_config).to(self.device)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "device": self.device,
        }

    def _mean_pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling accounting for attention mask padding."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Differentiable forward pass for fine-tuning.

        Gradients remain attached to the output embeddings.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        return self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

    @torch.no_grad()
    def extract_embeddings(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Extract PEFT text embeddings for a list of entity text descriptions.

        Args:
            texts: List of drug names, SMILES, or text descriptions.
            batch_size: Tokenizer processing batch size.

        Returns:
            A tensor containing PEFT-adapted representations.

        """
        self.model.eval()
        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            embeddings = self.__call__(batch_texts)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
