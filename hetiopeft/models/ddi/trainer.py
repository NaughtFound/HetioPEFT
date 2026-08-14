import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import HeteroData

from hetiopeft.models import Trainer
from hetiopeft.utils import convert_to_edge_type

from .hetero import HeteroDDIModel


class HeteroDDITrainer(Trainer):
    """Manage training, evaluation, and logging for DDI link prediction models."""

    def __init__(
        self,
        model: HeteroDDIModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        target_edge: Any,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the trainer instance with target configuration and hardware device.

        Args:
            model: The HeteroDDIModel instance to train and evaluate.
            optimizer: PyTorch optimizer configured for model parameters.
            criterion: Loss function module for binary edge classification.
            target_edge: Relation tuple specifying (src_type, relation, dst_type).
            device: Computing device for tensor operations ("cpu" or "cuda").

        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.target_edge = convert_to_edge_type(target_edge)
        self.device = torch.device(device)

    def log_params(self, **kwargs: Any) -> None:
        """Log training hyperparameter parameters and model configuration to MLflow.

        Args:
            **kwargs: Additional key-value pairs to record in MLflow parameters.

        """
        params = {
            "use_peft": self.model.use_peft,
            "hidden_dim": self.model.hidden_dim,
            "target_node_type": self.model.target_node_type,
            "target_edge": str(self.target_edge),
            "optimizer": self.optimizer.__class__.__name__,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **kwargs,
        }
        mlflow.log_params(params)

    def forward(
        self,
        data: HeteroData,
        *,
        update: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute model forward pass, loss, and conditionally perform backpropagation.

        Args:
            data: Input heterogeneous graph split containing features and target edges.
            update: If True, executes backpropagation and updates optimizer parameters.

        Returns:
            A tuple containing (loss, logits, edge_label) tensors.

        """
        data = data.to(self.device)

        edge_label_index = data[self.target_edge].edge_label_index
        edge_label = data[self.target_edge].edge_label

        try:
            x_dict = data.x_dict
        except KeyError:
            x_dict = {}

        logits = self.model(
            x_dict=x_dict,
            edge_index_dict=data.edge_index_dict,
            edge_label_index=edge_label_index,
        )
        loss = self.criterion(logits, edge_label.float())

        if update:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, logits, edge_label

    def train(
        self,
        data: HeteroData,
        step: int = 0,
        prefix: str = "",
    ) -> int:
        """Execute a single full-graph training step and record metrics to MLflow.

        Args:
            data: Heterogeneous graph dataset split used for training.
            step: Current global training step index.
            prefix: String prefix prepended to console log outputs.

        Returns:
            The updated global step integer count.

        """
        self.model.train()

        loss, logits, edge_label = self.forward(data)

        current_step = step + 1
        loss_val = float(loss.item())

        pred = torch.sigmoid(logits).detach().cpu().numpy()
        targets = edge_label.cpu().numpy()
        auc = float(roc_auc_score(targets, pred))

        mlflow.log_metric("train_loss", loss_val, step=current_step)
        mlflow.log_metric("train_auc", auc, step=current_step)

        logging.info(f"{prefix} Train Loss: {loss_val:.4f} | Train ROC-AUC: {auc:.4f}")
        return current_step

    @torch.no_grad()
    def evaluate(
        self,
        data: HeteroData,
        step: int | None = None,
        prefix: str = "val",
    ) -> float:
        """Evaluate model predictions on a target validation or test graph split.

        Args:
            data: Target graph split containing test/validation evaluation edges.
            step: Current training step or epoch index for MLflow metric logging.
            prefix: Prefix indicating split environment ("val" or "test").

        Returns:
            Calculated loss float value for the evaluated graph split.

        """
        self.model.eval()
        data = data.to(self.device)

        loss, logits, edge_label = self.forward(data, update=False)

        loss_val = float(loss.item())

        pred = torch.sigmoid(logits).cpu().numpy()
        targets = edge_label.cpu().numpy()

        auc = float(roc_auc_score(targets, pred))
        ap = float(average_precision_score(targets, pred))

        if step is not None:
            mlflow.log_metric(f"{prefix}_loss", loss_val, step=step)
            mlflow.log_metric(f"{prefix}_auc", auc, step=step)
            mlflow.log_metric(f"{prefix}_ap", ap, step=step)

        logging.info(
            f"[{prefix.upper()}] Loss: {loss_val:.4f} | ROC-AUC: {auc:.4f} | AP: {ap:.4f}"
        )
        return loss_val

    def save_checkpoint(self, path: Path | str) -> None:
        """Save model state dictionary to disk as a safetensors file.

        Args:
            path: Target directory path where checkpoint file will be saved.

        """
        self.model.save_pretrained(path, filename="hetero_ddi.safetensors")
