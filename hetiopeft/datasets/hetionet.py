from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm

from hetiopeft.utils import download_file, get_compound_smiles

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_geometric.typing import EdgeType

    from hetiopeft.models.peft import PEFTFeatureExtractor


class Hetionet(InMemoryDataset):
    """Custom PyTorch Geometric Dataset for Hetionet v1.0."""

    BASE_FILE = "hetionet_pyg.pt"
    EMBEDDED_FILE = "hetionet_pyg_embedded.pt"
    NODES_FILES = "nodes.tsv"
    EDGES_FILES = "edges.sif.gz"

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        *,
        fetch_smiles: bool = False,
        with_embeddings: bool = False,
    ) -> None:
        self.root_path = Path(root)
        self.with_embeddings = with_embeddings
        self.fetch_smiles = fetch_smiles
        super().__init__(str(self.root_path), transform, pre_transform)
        target_path = Path(self.processed_paths[1 if self.with_embeddings else 0])

        if self.with_embeddings and not target_path.exists():
            msg = (
                f"Embeddings file not found at '{target_path}'. "
                "Please generate and save the embedding file before setting "
                "`with_embeddings=True`."
            )
            raise RuntimeError(msg)

        self.load(str(target_path))

    @property
    def raw_file_names(self) -> list[str]:
        return [Hetionet.NODES_FILES, Hetionet.EDGES_FILES]

    @property
    def processed_file_names(self) -> list[str]:
        if self.with_embeddings:
            return [Hetionet.BASE_FILE, Hetionet.EMBEDDED_FILE]
        return [Hetionet.BASE_FILE]

    @property
    def base_path(self) -> Path:
        return Path(self.processed_dir) / Hetionet.BASE_FILE

    @property
    def embedded_path(self) -> Path:
        return Path(self.processed_dir) / Hetionet.EMBEDDED_FILE

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
        if self.base_path.exists():
            return

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

                compound_smiles: list[str] = []
                compound_iupac: list[str] = []

                bar = tqdm(
                    zip(drug_bank_ids, compound_names, strict=True),
                    total=len(drug_bank_ids),
                    desc="Fetching SMILES",
                )
                for db_id, name in bar:
                    bar.set_description(f"Fetching SMILES for {name}")
                    res = get_compound_smiles(db_id) or {}

                    iupac = res.get("iupac", "N/A")
                    smiles = res.get("smiles", "N/A")

                    compound_smiles.append(smiles)
                    compound_iupac.append(iupac)

                data["Compound"].name = compound_names
                data["Compound"].smiles = compound_smiles
                data["Compound"].iupac = compound_iupac

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

    def generate_embeddings(
        self,
        extractor: PEFTFeatureExtractor,
        batch_size: int = 32,
    ) -> None:
        """Manually extract compound embeddings."""
        if not self.base_path.exists():
            msg = (
                f"Base graph file '{self.base_path}' does not exist. "
                "Run dataset processing first."
            )
            raise RuntimeError(msg)

        data_list = torch.load(self.base_path)
        data = HeteroData.from_dict(data_list[0])

        missing_attrs = [
            attr
            for attr in ["smiles", "iupac", "name"]
            if not hasattr(data["Compound"], attr)
        ]
        if missing_attrs:
            msg = (
                f"Missing attributes in data['Compound']: {missing_attrs}. "
                "Cannot generate formatted compound descriptions."
            )
            raise ValueError(msg)

        logging.info("Generating PEFT embeddings for Compound nodes...")
        compound_texts = [
            f"Compound {name}. IUPAC: {iupac}. SMILES: {smiles}"
            for name, iupac, smiles in zip(
                data["Compound"].name,
                data["Compound"].iupac,
                data["Compound"].smiles,
                strict=True,
            )
        ]
        data["Compound"].x = extractor.extract_embeddings(
            compound_texts,
            batch_size=batch_size,
        )

        self.save([data], str(self.embedded_path))
        logging.info(f"Saved embedded dataset to: {self.embedded_path}")
