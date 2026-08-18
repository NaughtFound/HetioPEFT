import mlflow
from torch_geometric.data import Dataset

from hetiopeft.datasets import Hetionet
from hetiopeft.models import Processor

from .extractor import PEFTFeatureExtractor


class PEFTProcessor(Processor):
    def __init__(self, extractor: PEFTFeatureExtractor, batch_size: int) -> None:
        super().__init__()

        self.extractor = extractor
        self.batch_size = batch_size

    def log_params(self, **kwargs) -> None:
        mlflow.log_params(
            {
                **self.extractor.config,
                "batch_size": self.batch_size,
                **kwargs,
            }
        )

    def process(self, dataset: Dataset) -> None:
        if not isinstance(dataset, Hetionet):
            msg = "This function is only working for `Hetionet`"
            raise TypeError(msg)

        dataset.generate_embeddings(self.extractor, self.batch_size)
