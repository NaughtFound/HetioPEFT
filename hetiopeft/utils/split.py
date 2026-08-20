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
    disjoint_train_ratio: float = 0.2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[D, D, D]:
    torch.manual_seed(seed)

    edge_type = convert_to_edge_type(target_edge)

    transform = t.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=[edge_type],
        rev_edge_types=[edge_type],
        disjoint_train_ratio=disjoint_train_ratio,
    )

    train_data, val_data, test_data = transform(data)

    return (train_data, val_data, test_data)
