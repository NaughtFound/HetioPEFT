from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from hetiopeft.utils import download_file


class Hetionet(InMemoryDataset):
    """Custom PyTorch Geometric Dataset for Hetionet v1.0."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        self.root_path = Path(root)
        super().__init__(str(self.root_path), transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["nodes.tsv", "edges.sif.gz"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["hetionet_pyg.pt"]

    def download(self) -> None:
        urls = {
            "nodes.tsv": "https://raw.githubusercontent.com/hetio/hetionet/master/hetnet/tsv/hetionet-v1.0-nodes.tsv",
            "edges.sif.gz": "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
        }

        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        for filename, url in urls.items():
            dest_path = raw_dir / filename
            if not dest_path.exists():
                download_file(url, dest_path)

    def process(self) -> None:
        raw_dir = Path(self.raw_dir)
        nodes_df = pd.read_csv(raw_dir / "nodes.tsv", sep="\t")
        edges_df = pd.read_csv(raw_dir / "edges.sif.gz", sep="\t", compression="gzip")

        data = HeteroData()

        # 1. Local integer ID mapping per entity type
        node_id_maps = {}
        for kind_raw, group in nodes_df.groupby("kind"):
            kind = str(kind_raw)
            raw_ids = group["id"].tolist()
            node_id_maps[kind] = {str(raw_id): idx for idx, raw_id in enumerate(raw_ids)}
            data[kind].num_nodes = len(raw_ids)

        # 2. Add edges per metaedge relation
        for metaedge_raw, group in edges_df.groupby("metaedge"):
            metaedge = str(metaedge_raw)
            src_kind = str(group["source"].iloc[0]).split("::")[0]
            dst_kind = str(group["target"].iloc[0]).split("::")[0]

            src_indices = [node_id_maps[src_kind][str(s)] for s in group["source"]]
            dst_indices = [node_id_maps[dst_kind][str(t)] for t in group["target"]]

            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_key: tuple[str, str, str] = (src_kind, metaedge, dst_kind)
            data[edge_key].edge_index = edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
