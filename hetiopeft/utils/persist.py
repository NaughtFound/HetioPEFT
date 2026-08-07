import json
import logging
from pathlib import Path
from typing import Any, Self

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


class PersistMixin:
    @property
    def config(self) -> dict[str, Any]:
        return {}

    def save_pretrained(
        self,
        path: str | Path,
        filename: str = "model.safetensors",
        **extra_metadata,
    ) -> None:
        if not isinstance(self, nn.Module):
            msg = (
                f"PersistMixin can only be used with subclasses of 'torch.nn.Module', "
                f"but got '{type(self).__qualname__}'."
            )
            raise TypeError(msg)

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / filename

        metadata = {
            "config": json.dumps(self.config),
            "class_name": self.__class__.__name__,
        }
        for k, v in extra_metadata.items():
            metadata[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

        state_dict = self.state_dict()
        save_file(state_dict, file_path, metadata=metadata)
        logging.info(f"Model saved to {file_path}")

    @classmethod
    def from_pretrained(
        cls: type[Self],
        path: str | Path,
        filename: str = "model.safetensors",
        device: str = "cpu",
        **override_kwargs,
    ) -> Self:
        if not issubclass(cls, nn.Module):
            msg = (
                f"PersistMixin can only be used with subclasses of 'torch.nn.Module', "
                f"but got '{cls.__qualname__}'."
            )
            raise TypeError(msg)

        file_path = Path(path) / filename

        if not file_path.exists():
            msg = f"No file found at {file_path}"
            raise FileNotFoundError(msg)

        with safe_open(str(file_path), framework="pt", device=device) as f:
            metadata = f.metadata() or {}

        if "config" not in metadata:
            msg = "No configuration found in metadata to rebuild model."
            raise ValueError(msg)

        config = json.loads(metadata["config"])
        config.update(override_kwargs)

        model = cls(**config)

        state_dict = load_file(str(file_path), device=device)
        model.load_state_dict(state_dict)

        return model
