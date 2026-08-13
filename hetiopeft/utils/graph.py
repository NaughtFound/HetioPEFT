from torch_geometric.data import HeteroData


def extract_graph_metadata(
    data: HeteroData,
) -> tuple[tuple[list[str], list[tuple[str, str, str]]], dict[str, int]]:
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
