from http import HTTPStatus

import requests


def get_compound_smiles_or_description(drug_bank_id: str, timeout: float = 10.0) -> str:
    """Fetch CID or description from PubChem using DrugBank ID.

    Args:
        drug_bank_id: The DrugBank ID (e.g., 'DB00843').
        timeout: Request timeout in seconds. Defaults to 10.0.

    Returns:
        Formatted summary string with chemical properties or fallback label.

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
            return f"Compound {title}. IUPAC: {iupac}. SMILES: {smiles}"
    except (requests.RequestException, KeyError, IndexError):
        pass

    return f"Compound with DrugBank ID {drug_bank_id}"
