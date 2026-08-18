import logging
from abc import ABC, abstractmethod
from pathlib import Path

from torch_geometric.data import HeteroData


class Trainer(ABC):
    @abstractmethod
    def log_params(self, **kwargs) -> None: ...

    @abstractmethod
    def train(
        self,
        data: HeteroData,
        step: int = 0,
        prefix: str = "",
    ) -> int: ...

    @abstractmethod
    def evaluate(
        self,
        data: HeteroData,
        step: int | None = None,
        prefix: str = "val",
    ) -> float: ...

    def test(self, data: HeteroData) -> float:
        test_loss = self.evaluate(data, prefix="test")
        logging.info(f"Test Loss: {test_loss:.4f}")
        return test_loss

    @abstractmethod
    def save_checkpoint(self, path: Path | str) -> None: ...
