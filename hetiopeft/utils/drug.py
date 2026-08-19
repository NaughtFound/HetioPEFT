from http import HTTPStatus

import requests
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def get_compound_smiles(
    drug_bank_id: str,
    timeout: float = 10.0,
) -> dict[str, str] | None:
    """Fetch compound title, IUPAC name, and SMILES from PubChem using a DrugBank ID.

    Args:
        drug_bank_id: The DrugBank identifier (e.g., 'DB00843').
        timeout: Request timeout in seconds. Defaults to 10.0.

    Returns:
        A dictionary containing 'title', 'iupac', and 'smiles' keys if found,
        or None if the API request fails or the compound is not found.

    """
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{drug_bank_id}/property/Title,IUPACName,SMILES/JSON"
    )

    try:
        res = requests.get(url, timeout=timeout)
        if res.status_code == HTTPStatus.OK:
            props = res.json()["PropertyTable"]["Properties"][0]
            title = props.get("Title", "Unknown")
            iupac = props.get("IUPACName", "N/A")
            smiles = props.get("SMILES", "N/A")

            return {"title": title, "iupac": iupac, "smiles": smiles}
    except (requests.RequestException, KeyError, IndexError):
        pass

    return None


def smiles_to_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 1024,
) -> torch.Tensor:
    """Convert a SMILES string to a binary Morgan Fingerprint tensor."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(n_bits, dtype=torch.float32)

    arr = gen.GetFingerprintAsNumPy(mol)
    return torch.tensor(arr, dtype=torch.float32)
