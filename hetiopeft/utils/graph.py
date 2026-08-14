from collections.abc import Sequence
from typing import Any

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.typing import EdgeType, NodeType


def extract_graph_metadata(
    data: HeteroData,
) -> tuple[tuple[list[NodeType], list[EdgeType]], dict[str, int]]:
    """Extract structural metadata and node count mappings from a HeteroData graph.

    Args:
        data: Input PyTorch Geometric heterogeneous graph instance.

    Returns:
        A tuple containing:
            - metadata: Tuple of (node_types, edge_types).
            - num_nodes_dict: Dictionary mapping node_type to total node count.

    """
    metadata = data.metadata()
    num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}
    return metadata, num_nodes_dict


def extract_data_from_dataset(dataset: Dataset) -> HeteroData:
    """Extract the primary HeteroData graph object from a dataset instance.

    Args:
        dataset: A PyTorch Geometric Dataset containing heterogeneous graph data.

    Returns:
        The extracted HeteroData graph instance.

    Raises:
        TypeError: If the extracted graph object is not an instance of HeteroData.

    """
    data = dataset[0]

    if not isinstance(data, HeteroData):
        msg = f"Expected dataset[0] to be HeteroData, but received {type(data).__name__}."
        raise TypeError(msg)

    return data


def convert_to_edge_type(edges: Any) -> EdgeType:
    """Convert a sequence of edge components into a PyG EdgeType tuple.

    Validate that the input sequence contains exactly three non-empty string
    elements representing the source node type, relation type, and destination
    node type, returning them as a trimmed 3-tuple.

    Args:
        edges: A list or tuple containing three strings in the order
            ``[src_node_type, relation_type, dst_node_type]``
            (e.g., ``["Compound", "CrC", "Compound"]``).

    Returns:
        EdgeType: A 3-tuple of stripped strings ``(src, relation, dst)``.

    Raises:
        TypeError: If ``edges`` is not a list or tuple.
        ValueError: If ``edges`` does not contain exactly 3 elements, or if
            any element is not a non-empty string.

    Examples:
        >>> convert_to_edge_type(["Compound", "CrC", "Compound"])
        ('Compound', 'CrC', 'Compound')

    """
    num_edges = 3
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        msg = f"Expected a list or tuple, got {type(edges).__name__}"
        raise TypeError(msg)

    if len(edges) != num_edges:
        msg = (
            f"Expected exactly {num_edges} elements [src, relation, dst], "
            f"got {len(edges)}: {edges}"
        )
        raise ValueError(msg)

    for i, item in enumerate(edges):
        if not isinstance(item, str) or not item.strip():
            msg = f"Element at index {i} must be a non-empty string, got {item!r}"
            raise ValueError(msg)

    return (edges[0].strip(), edges[1].strip(), edges[2].strip())
