import torch
import torch_geometric.transforms as t
from torch_geometric.data import Data


def split_data[D: Data](
    data: D,
    target_edge: tuple[str, str, str],
    *,
    seed: int = 0,
    val_ratio: float = 0.15,
    test_ratio: float = 0.7,
) -> tuple[D, D, D]:
    torch.manual_seed(seed)

    transform = t.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=[target_edge],
    )

    train_data, val_data, test_data = transform(data)

    return (train_data, val_data, test_data)
