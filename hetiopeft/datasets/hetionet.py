from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm

from hetiopeft.utils import download_file, get_compound_smiles_or_description

if TYPE_CHECKING:
    from torch_geometric.typing import EdgeType


class Hetionet(InMemoryDataset):
    """Custom PyTorch Geometric Dataset for Hetionet v1.0."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        *,
        fetch_smiles: bool = False,
    ) -> None:
        self.root_path = Path(root)
        self.fetch_smiles = fetch_smiles
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

            if self.fetch_smiles and kind == "Compound":
                drug_bank_ids = [rid.split("::")[-1] for rid in raw_ids]
                compound_names = group["name"].tolist()

                compound_descriptions: list[str] = []
                bar = tqdm(
                    zip(drug_bank_ids, compound_names, strict=True),
                    total=len(drug_bank_ids),
                    desc="Fetching SMILES",
                )
                for db_id, name in bar:
                    bar.set_description(f"Fetching SMILES for {name}")
                    desc = get_compound_smiles_or_description(db_id)
                    compound_descriptions.append(desc)

                data["Compound"].smiles = compound_descriptions

        # 2. Add edges per metaedge relation
        for metaedge_raw, group in edges_df.groupby("metaedge"):
            metaedge = str(metaedge_raw)
            src_kind = str(group["source"].iloc[0]).split("::")[0]
            dst_kind = str(group["target"].iloc[0]).split("::")[0]

            src_indices = [node_id_maps[src_kind][str(s)] for s in group["source"]]
            dst_indices = [node_id_maps[dst_kind][str(t)] for t in group["target"]]

            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_key: EdgeType = (src_kind, metaedge, dst_kind)
            data[edge_key].edge_index = edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
