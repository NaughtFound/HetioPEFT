import logging
from abc import ABC, abstractmethod
from pathlib import Path

from torch_geometric.data import Data


class Trainer(ABC):
    @abstractmethod
    def log_params(self, **kwargs) -> None: ...

    @abstractmethod
    def train(
        self,
        data: Data,
        step: int = 0,
        prefix: str = "",
    ) -> int: ...

    @abstractmethod
    def evaluate(
        self,
        data: Data,
        step: int | None = None,
        prefix: str = "val",
    ) -> float: ...

    def test(self, data: Data) -> float:
        test_loss = self.evaluate(data, prefix="test")
        logging.info(f"Test Loss: {test_loss:.4f}")
        return test_loss

    @abstractmethod
    def save_checkpoint(self, path: Path | str) -> None: ...
