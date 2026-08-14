import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


class PEFTFeatureExtractor:
    """Extract dense text embeddings using a PEFT / LoRA adapted language model."""

    def __init__(
        self,
        model_name: str,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize PubMedBERT backbone wrapped with PEFT LoRA adapters."""
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
        self.model.eval()

    @torch.no_grad()
    def extract_embeddings(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Extract PEFT text embeddings for a list of entity text descriptions.

        Args:
            texts: List of drug names, SMILES, or text descriptions.
            batch_size: Tokenizer processing batch size.

        Returns:
            A tensor containing PEFT-adapted representations.

        """
        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
