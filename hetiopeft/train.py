import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
from torch_geometric.data import HeteroData

from .models import Trainer
from .utils import split_data


@dataclass
class TrainConf:
    seed: int = 0
    val_ratio: float = 0.15
    test_ratio: float = 0.7
    num_epochs: int = 100
    prefix: str = "."
    run_id: str | None = None
    run_name: str = "run"
    experiment_name: str | None = None
    eval_every_n_epochs: int | None = None
    save_every_n_epochs: int | None = None
    split_target_edge: tuple[str, str, str] = ("Compound", "CrC", "Compound")


def train(data: HeteroData, trainer: Trainer, *, conf: TrainConf) -> None:
    db_path = (Path(conf.prefix) / "mlflow.db").resolve()
    checkpoints_path = Path(conf.prefix) / conf.run_name / "checkpoints"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    if conf.experiment_name is not None:
        mlflow.set_experiment(conf.experiment_name)

    train_data, val_data, test_data = split_data(
        data=data,
        target_edge=conf.split_target_edge,
        seed=conf.seed,
        val_ratio=conf.val_ratio,
        test_ratio=conf.test_ratio,
    )

    logging.info(f"train size: {len(train_data)}")
    logging.info(f"validation size: {len(val_data)}")
    logging.info(f"test size: {len(test_data)}")

    with mlflow.start_run(run_name=conf.run_name, run_id=conf.run_id):
        global_steps = 0
        trainer.log_params(num_epochs=conf.num_epochs)
        for epoch in range(conf.num_epochs):
            global_steps = trainer.train(
                train_data,
                step=global_steps,
                prefix=f"(Epoch {epoch + 1}/{conf.num_epochs})",
            )

            if (
                val_data
                and conf.eval_every_n_epochs is not None
                and (epoch % conf.eval_every_n_epochs == 0 or epoch == conf.num_epochs)
            ):
                trainer.evaluate(val_data, step=epoch)

            if (epoch + 1) == conf.num_epochs or (
                conf.save_every_n_epochs is not None
                and (epoch + 1) % conf.save_every_n_epochs == 0
            ):
                trainer.save_checkpoint(checkpoints_path / str(epoch + 1))

        if test_data:
            trainer.test(test_data)
