from typing import Any

import torch
import torch_geometric.transforms as t
from torch_geometric.data import HeteroData

from .graph import convert_to_edge_type


def split_data[D: HeteroData](
    data: D,
    target_edge: Any,
    *,
    seed: int = 0,
    val_ratio: float = 0.15,
    test_ratio: float = 0.7,
) -> tuple[D, D, D]:
    torch.manual_seed(seed)

    transform = t.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        add_negative_train_samples=True,
        edge_types=[convert_to_edge_type(target_edge)],
    )

    train_data, val_data, test_data = transform(data)

    return (train_data, val_data, test_data)
