"""Download data files from DANDI dataset.

Run this script to download behavior and spikesorting data from the DANDI
dataset.
Flages can be used to filter the data by modality, subject, and session:
- `--modality`: Specify the modality to download. Options are 'behavior' or
  'spikesorting'. If not specified, both modalities will be downloaded.
- `--subject`: Specify the subject to filter files by. If not specified, all
  subjects will be downloaded.
- `--session`: Specify the session to filter files by. If not specified, all
  sessions will be downloaded.

Note: To run this script you need to have a DANDI API key saved in a file named
`DANDI_API_KEY.txt` in the current directory. The key should be a single line
containing the API key.

Usage examples:
```
# Download behavior and spikesorting data for all subjects and sessions
$ python3 download_dandi_data.py

# Download only behavior data for all subjects and sessions
$ python3 download_dandi_data.py --modality=behavior

# Download all spikesorting data for subject "Perle"
$ python3 download_dandi_data.py --modality=spikesorting --subject=Perle

# Download behavior and spikesorting data for session "Perle/2022-06-01"
$ python3 download_dandi_data.py --subject=Perle --session=2022-06-01
```
"""

import sys
from pathlib import Path

from dandi import dandiapi

_WRITE_DIR = Path("./cache/dandi_data")
_MODALITY_TO_SUFFIX = {
    "behavior": "behavior+task",
    "spikesorting": "spikesorting",
}

# Load DANDI API key from file
with open("./DANDI_API_KEY.txt", "r") as f:
    DANDI_API_KEY = f.read().strip()


def _download_dandiset(
    modality: str | None = None,
    subject: str | None = None,
    session: str | None = None,
    dandiset_id: str = "000620",
):
    """Download all files except raw ecephys files from a dandiset.

    Args:
        modality (str): None or modality to download. Must be either 'behavior'
            or 'spikesorting'. If None, downloads both modalities.
        subject (str | None): Subject to filter files by.
        session (str | None): Session to filter files by.
        dandiset_id (str): ID of the DANDI dataset to download.
    """
    api = dandiapi.DandiAPIClient(token=DANDI_API_KEY)
    dandiset = api.get_dandiset(dandiset_id, "draft")
    all_assets = [x for x in dandiset.get_assets()]

    if modality is None:
        modalities = _MODALITY_TO_SUFFIX.keys()
    else:
        modalities = [modality]

    for modality in modalities:
        write_dir = _WRITE_DIR / modality
        print(f"\nDownloading {modality} data to {write_dir}\n")

        # Filter assets by modality
        suffix = _MODALITY_TO_SUFFIX[modality]
        assets = [x for x in all_assets if x.path.endswith(f"{suffix}.nwb")]
        if not assets:
            print(f"No assets found for modality: {modality}")
            continue

        # Filter by subject if provided
        if subject is not None:
            assets = [
                x for x in assets if x.path.startswith(f"sub-{subject}/")
            ]
            if not assets:
                print(f"No assets found for subject: {subject}")
                continue

        # Filter by session if provided
        if session is not None:
            assets = [x for x in assets if f"_ses-{session}_" in x.path]
            if not assets:
                print(f"No assets found for session: {session}")
                continue

        # Download each asset
        for i, asset in enumerate(assets):
            print(f"    Downloading asset {i + 1}/{len(assets)}: {asset.path}")
            download_path = write_dir / asset.path
            if download_path.exists():
                print(f"    Asset already exists: {download_path}")
                continue

            # Create download directory if it does not exist
            download_path.parent.mkdir(parents=True, exist_ok=True)

            # Download asset
            asset.download(download_path)


if __name__ == "__main__":
    # Load flag arguments
    kwargs = {}
    for arg in sys.argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(
                f"Invalid argument format: {arg}. Expected format: --flag=value"
            )
        flag, value = arg[2:].split("=", 1)
        kwargs[flag] = value
    _download_dandiset(**kwargs)
