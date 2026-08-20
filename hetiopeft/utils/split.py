from typing import Any

import torch
import torch_geometric.transforms as t
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling

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


def resample_train_negatives(train_data: HeteroData, target_edge: Any) -> HeteroData:
    """Resample negative edges dynamically for each training epoch.

    Args:
        train_data: The heterogeneous graph dataset containing training edges and labels.
        target_edge: The edge type tuple identifying the target relation
            (e.g., ('Compound', 'interacts', 'Compound')).

    Returns:
        The modified heterogeneous graph dataset with updated edge_label_index
        and edge_label tensors containing newly sampled negative edges.

    """
    edge_type = convert_to_edge_type(target_edge)
    pos_edge_index = train_data[edge_type].edge_label_index[
        :, train_data[edge_type].edge_label == 1
    ]
    num_pos = pos_edge_index.size(1)
    src_type, _, dst_type = edge_type

    if src_type == dst_type:
        num_nodes = train_data[src_type].num_nodes
    else:
        num_nodes = (
            train_data[src_type].num_nodes,
            train_data[dst_type].num_nodes,
        )

    neg_edge_index = negative_sampling(
        edge_index=train_data[edge_type].edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_pos,
        method="sparse",
    )

    device = pos_edge_index.device
    train_data[edge_type].edge_label_index = torch.cat(
        [pos_edge_index, neg_edge_index],
        dim=-1,
    )
    train_data[edge_type].edge_label = torch.cat(
        [
            torch.ones(num_pos, device=device),
            torch.zeros(num_pos, device=device),
        ],
        dim=0,
    )

    return train_data
