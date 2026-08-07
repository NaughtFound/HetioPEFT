from abc import ABC, abstractmethod

from torch_geometric.data import HeteroData


class Metric(ABC):
    @abstractmethod
    def log_params(self, **kwargs) -> None: ...

    @abstractmethod
    def calc(self, data: HeteroData) -> None: ...
