# src/preprocess.py
"""Dataset download & preprocessing utilities.
All file-system interactions (download, checksum, extract) are
centralised here so that experiments remain fully deterministic and
self-contained.
"""
from __future__ import annotations

import hashlib
import tarfile
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Dict

import torch
import torchvision
import torchaudio
from torch.utils.data import Dataset

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
DATA_DIR = Path("data").resolve()
DATA_DIR.mkdir(exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str | None = None) -> Path:
    """Download a remote file with optional SHA-256 verification."""
    if dest.exists():
        if sha256 and _sha256(dest) != sha256:
            dest.unlink()
        else:
            return dest
    print(f"Downloading {url} → {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    if sha256 and _sha256(dest) != sha256:
        raise RuntimeError(f"Checksum mismatch for {url}")
    return dest

# ---------------------------------------------------------------------
# Public download functions used by Experiment-1
# ---------------------------------------------------------------------
KINETICS_URL = "https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_320p.zip"
KINETICS_SHA = "b7bdabbd2bb160e86f68202f785e4c4c1575e0e3961cdbe503ac3bc61cf3f975"

LIBRI_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LIBRI_SHA = "42e0fabf1c201802db8f152e3cdf4d00d90e1a5da760c01dd1c08fb8b9d6d4d7"

NYU_URL = "http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz"



def get_kinetics_subset() -> Path:
    zip_path = _download(KINETICS_URL, DATA_DIR / "k700_train_320p.zip", KINETICS_SHA)
    extract_dir = DATA_DIR / "k700_subset"
    if not extract_dir.exists():
        print("Extracting Kinetics subset – first 500 clips…")
        with zipfile.ZipFile(zip_path) as z:
            wanted = [m for m in z.namelist()[:500]]
            for member in wanted:
                z.extract(member, extract_dir)
    return extract_dir


def get_librispeech() -> Path:
    tar_path = _download(LIBRI_URL, DATA_DIR / "librispeech_dev_clean.tar.gz", LIBRI_SHA)
    extract_dir = DATA_DIR / "librispeech_dev_clean"
    if not extract_dir.exists():
        with tarfile.open(tar_path) as t:
            t.extractall(path=extract_dir)
    return extract_dir


def get_nyu_depth() -> Path:
    tar_path = _download(NYU_URL, DATA_DIR / "nyu_depth.tar.gz")
    extract_dir = DATA_DIR / "nyu_depth"
    if not extract_dir.exists():
        with tarfile.open(tar_path) as t:
            t.extractall(path=extract_dir)
    return extract_dir

# ---------------------------------------------------------------------
# Convenience dataloader for Experiment-1
# ---------------------------------------------------------------------
class AVDepthDataset(Dataset):
    """Pairs Kinetics video, Librispeech audio and NYU-Depth images by index."""

    def __init__(self, kinetics_root: Path, libri_root: Path, nyu_root: Path):
        super().__init__()
        # Video transforms: centre crop + [-1,1] range
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop((320, 576)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )
        self.video_ds = torchvision.datasets.VideoFolder(
            root=str(kinetics_root), extensions=(".mp4",), transform=transform
        )
        self.audio_files = sorted(Path(libri_root).rglob("*.flac"))[: len(self.video_ds)]
        self.depth_files = sorted(Path(nyu_root).rglob("*.png"))

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.video_ds)

    # --------------------------------------------------------------
    def __getitem__(self, idx: int):
        video, _, _, _ = self.video_ds[idx]
        audio, sr = torchaudio.load(self.audio_files[idx])
        depth = torchvision.io.read_image(str(self.depth_files[idx % len(self.depth_files)]))
        depth = depth.float() / 255.0
        return video, audio, sr, depth
