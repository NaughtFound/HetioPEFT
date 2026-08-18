from abc import ABC, abstractmethod

from torch_geometric.data import Dataset


class Processor(ABC):
    @abstractmethod
    def log_params(self, **kwargs) -> None: ...

    @abstractmethod
    def process(self, dataset: Dataset) -> None: ...
