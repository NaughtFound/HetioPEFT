from typing import Any

import torch
import torch.nn.functional as f
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.typing import EdgeType, NodeType

from hetiopeft.utils import PersistMixin, extract_graph_metadata


class HeteroDDIModel(nn.Module, PersistMixin):
    """Heterogeneous GNN link predictor for drug-drug interactions (DDI)."""

    def __init__(
        self,
        *,
        metadata: tuple[list[NodeType], list[EdgeType]],
        num_nodes_dict: dict[str, int],
        hidden_dim: int = 128,
        target_node_type: NodeType = "Compound",
        peft_in_dim: int = 768,
        use_peft: bool = True,
    ) -> None:
        """Initialize the DDIModel architecture."""
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.num_nodes_dict = num_nodes_dict
        self.hidden_dim = hidden_dim
        self.target_node_type = target_node_type
        self.peft_in_dim = peft_in_dim
        self.use_peft = use_peft

        self.node_encoders = nn.ModuleDict()
        for node_type in self.node_types:
            if node_type == self.target_node_type and self.use_peft:
                self.node_encoders[node_type] = Linear(peft_in_dim, hidden_dim)
            else:
                num_nodes = num_nodes_dict[node_type]
                self.node_encoders[node_type] = nn.Embedding(num_nodes, hidden_dim)

        conv_dict: dict[EdgeType, Any] = {}
        for edge_type in self.edge_types:
            conv_dict[edge_type] = GATConv(hidden_dim, hidden_dim, add_self_loops=False)

        self.hetero_conv = HeteroConv(conv_dict, aggr="sum")

        self.link_predictor = nn.Sequential(
            Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(64, 1),
        )

    @staticmethod
    def from_data(
        data: HeteroData,
        *,
        hidden_dim: int = 128,
        target_node_type: NodeType = "Compound",
        peft_in_dim: int = 768,
        use_peft: bool = True,
    ) -> "HeteroDDIModel":
        """Instantiate a HeteroDDIModel directly from a HeteroData graph instance."""
        metadata, num_nodes_dict = extract_graph_metadata(data)

        return HeteroDDIModel(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            hidden_dim=hidden_dim,
            target_node_type=target_node_type,
            peft_in_dim=peft_in_dim,
            use_peft=use_peft,
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "metadata": (self.node_types, self.edge_types),
            "num_nodes_dict": self.num_nodes_dict,
            "hidden_dim": self.hidden_dim,
            "target_node_type": self.target_node_type,
            "peft_in_dim": self.peft_in_dim,
            "use_peft": self.use_peft,
        }

    def encode(
        self,
        x_dict: dict[NodeType, torch.Tensor | None],
        edge_index_dict: dict[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode graph nodes using feature projections and message passing."""
        h_dict: dict[str, torch.Tensor] = {}

        for node_type in self.node_types:
            if node_type == self.target_node_type and self.use_peft:
                target_x = x_dict.get(node_type)
                if target_x is None:
                    msg = (
                        f"Expected feature tensor in x_dict['{node_type}'] "
                        "when use_peft=True."
                    )
                    raise ValueError(msg)
                h_dict[node_type] = self.node_encoders[node_type](target_x)
            else:
                h_dict[node_type] = self.node_encoders[node_type].weight  # pyright: ignore[reportArgumentType]

        h_conv = self.hetero_conv(h_dict, edge_index_dict)
        return {k: f.relu(v) for k, v in h_conv.items()}

    def decode(
        self,
        h_target: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict interaction logits for target node pairs."""
        src, dst = edge_label_index[0], edge_label_index[1]
        pair_features = torch.cat([h_target[src], h_target[dst]], dim=-1)
        return self.link_predictor(pair_features).squeeze(-1)

    def forward(
        self,
        x_dict: dict[NodeType, torch.Tensor | None],
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Perform forward pass to compute prediction logits for target edge pairs."""
        h_dict = self.encode(x_dict, edge_index_dict)
        return self.decode(h_dict[self.target_node_type], edge_label_index)
