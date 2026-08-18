from dataclasses import dataclass
from pathlib import Path

import mlflow
from torch_geometric.data import Dataset

from .models import Processor


@dataclass
class ProcessConf:
    prefix: str = "."
    run_id: str | None = None
    run_name: str = "run"
    experiment_name: str | None = None


def process(dataset: Dataset, processor: Processor, *, conf: ProcessConf) -> None:
    db_path = (Path(conf.prefix) / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    if conf.experiment_name is not None:
        mlflow.set_experiment(conf.experiment_name)

    with mlflow.start_run(run_name=conf.run_name, run_id=conf.run_id):
        processor.log_params()
        processor.process(dataset)
