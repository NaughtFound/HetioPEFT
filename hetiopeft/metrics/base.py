from abc import ABC, abstractmethod

from torch_geometric.data import Data


class Metric(ABC):
    @abstractmethod
    def log_params(self, **kwargs) -> None: ...

    @abstractmethod
    def calc(self, data: Data) -> None: ...
