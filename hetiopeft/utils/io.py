from pathlib import Path

import requests
from tqdm import tqdm


def download_file(
    url: str,
    dest_path: Path,
    timeout: tuple[float, float] = (5.0, 30.0),
) -> Path:
    """Download a file from a URL to a pathlib.Path destination with a timeout."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Added explicit timeout parameter to fix Ruff S113
    response = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        dest_path.open("wb") as file,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)

    return dest_path
