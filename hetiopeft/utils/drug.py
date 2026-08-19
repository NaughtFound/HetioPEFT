from http import HTTPStatus

import requests


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
