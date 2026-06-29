r"""
Download GEAR-SONIC model checkpoints and training data from Hugging Face Hub.

Windows-compatible version.

Repository: https://huggingface.co/nvidia/GEAR-SONIC

Usage from PowerShell or Command Prompt:
    python .\download_from_hf_windows.py
    python .\download_from_hf_windows.py --training
    python .\download_from_hf_windows.py --sample
    python .\download_from_hf_windows.py --output-dir "C:\\gear_sonic"
    python .\download_from_hf_windows.py --no-planner

Authentication options:
    python .\download_from_hf_windows.py --token YOUR_HF_TOKEN

    # PowerShell:
    $env:HF_TOKEN="YOUR_HF_TOKEN"
    python .\download_from_hf_windows.py

    # Command Prompt:
    set HF_TOKEN=YOUR_HF_TOKEN
    python .\download_from_hf_windows.py
"""

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ID = "nvidia/GEAR-SONIC"

# (filename in HF repo, local destination relative to output_dir)
POLICY_FILES = [
    ("model_encoder.onnx", "policy/release/model_encoder.onnx"),
    ("model_decoder.onnx", "policy/release/model_decoder.onnx"),
    ("observation_config.yaml", "policy/release/observation_config.yaml"),
]

PLANNER_FILE = ("planner_sonic.onnx", "planner/target_vel/V2/planner_sonic.onnx")

TRAINING_FILES = [
    ("sonic_release/last.pt", "sonic_release/last.pt"),
    ("sonic_release/config.yaml", "sonic_release/config.yaml"),
]

SMPL_TAR_PARTS_PREFIX = "bones_seed_smpl/bones_seed_smpl.tar.part_"
SMPL_TAR_PARTS = [f"{SMPL_TAR_PARTS_PREFIX}a{c}" for c in "abcdefg"]


class MultiPartFile:
    """
    Read several split files as one continuous binary stream.

    This replaces the Linux/macOS command:
        cat bones_seed_smpl.tar.part_* | tar xf - -C data/

    It works in Windows because tarfile reads directly from Python instead of
    relying on shell commands such as cat and tar.
    """

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self._index = 0
        self._file = None
        self._open_next_file()

    def _open_next_file(self):
        if self._file is not None:
            self._file.close()
            self._file = None

        if self._index < len(self.paths):
            self._file = self.paths[self._index].open("rb")
            self._index += 1

    def read(self, size=-1):
        if self._file is None:
            return b""

        chunks = []
        remaining = size

        while self._file is not None:
            if size is None or size < 0:
                chunk = self._file.read()
            else:
                if remaining <= 0:
                    break
                chunk = self._file.read(remaining)

            if chunk:
                chunks.append(chunk)
                if size is not None and size >= 0:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
            else:
                self._open_next_file()

        return b"".join(chunks)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def readable(self):
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GEAR-SONIC checkpoints from Hugging Face Hub"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save files. "
            "Defaults to gear_sonic_deploy/ for deployment or script folder for training/sample."
        ),
    )
    parser.add_argument(
        "--no-planner",
        action="store_true",
        help="Skip downloading the kinematic planner ONNX model",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Download training checkpoint + SMPL motion data (~30 GB)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download sample motion data only (1 walking sequence, ~4 MB)",
    )
    parser.add_argument(
        "--no-smpl",
        action="store_true",
        help="With --training, skip SMPL data download (checkpoint only)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, HF_TOKEN from the environment is used if set.",
    )
    parser.add_argument(
        "--keep-smpl-parts",
        action="store_true",
        help="With --training, keep the downloaded split tar parts after extraction.",
    )
    return parser.parse_args()


def _ensure_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        return hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Install it with:")
        print("  python -m pip install huggingface_hub")
        sys.exit(1)


def _safe_extract_stream(tar, destination):
    """Safely extract a tar stream without allowing paths outside destination."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for member in tar:
        target_path = (destination / member.name).resolve()

        try:
            common = os.path.commonpath([str(destination), str(target_path)])
        except ValueError:
            print(f"  WARNING: Skipping unsafe tar entry: {member.name}")
            continue

        if common != str(destination):
            print(f"  WARNING: Skipping unsafe tar entry: {member.name}")
            continue

        # The expected SMPL archive should contain regular files/directories.
        # Skip links to avoid Windows path/security surprises.
        if member.issym() or member.islnk():
            print(f"  WARNING: Skipping link in tar archive: {member.name}")
            continue

        tar.extract(member, path=destination)


def download_file(hf_hub_download, repo_id, hf_filename, local_dest, token=None):
    """Download hf_filename from the Hub and place it at local_dest."""
    print(f"  Downloading {hf_filename} ...", flush=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=hf_filename,
        token=token,
    )
    local_dest = Path(local_dest)
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, local_dest)
    print(f"  -> {local_dest}")


def download_and_extract_smpl(hf_hub_download, repo_id, output_dir, token=None, keep_parts=False):
    """Download split tar parts and extract SMPL data on Windows."""
    output_dir = Path(output_dir)
    parts_dir = output_dir / "bones_seed_smpl"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {len(SMPL_TAR_PARTS)} parts (~30 GB total) ...", flush=True)
    part_paths = []
    for hf_filename in SMPL_TAR_PARTS:
        local_name = Path(hf_filename).name
        local_dest = parts_dir / local_name
        if local_dest.exists():
            print(f"  (cached) {local_name}")
            part_paths.append(local_dest)
            continue

        cached = hf_hub_download(repo_id=repo_id, filename=hf_filename, token=token)
        shutil.copy2(cached, local_dest)
        part_paths.append(local_dest)
        print(f"  Downloaded {local_name}")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting to {data_dir / 'smpl_filtered'} ...", flush=True)

    stream = MultiPartFile(part_paths)
    try:
        # r|* is streaming mode, so Python can read the split files as one tar
        # without first creating another huge 30 GB combined tar file.
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            _safe_extract_stream(tar, data_dir)
    except tarfile.TarError as exc:
        print(f"  ERROR: Extraction failed: {exc}")
        sys.exit(1)
    finally:
        stream.close()

    smpl_dir = data_dir / "smpl_filtered"
    if smpl_dir.exists():
        n_files = sum(1 for f in smpl_dir.rglob("*.pkl"))
        print(f"  -> {smpl_dir} ({n_files} PKL files)")
    else:
        print(f"  WARNING: Expected {smpl_dir} but directory not found")

    if keep_parts:
        print(f"  Keeping tar parts under {parts_dir}")
    else:
        print("  Cleaning up tar parts ...")
        shutil.rmtree(parts_dir)


def download_sample_data(snapshot_download, repo_id, output_dir, token=None):
    """Download sample motion data (1 walking sequence)."""
    print("  Downloading sample data ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="sample_data/*",
        local_dir=str(output_dir),
        token=token,
    )
    sample_dir = Path(output_dir) / "sample_data"
    if sample_dir.exists():
        n_files = sum(1 for _ in sample_dir.rglob("*.pkl"))
        print(f"  -> {sample_dir} ({n_files} PKL files)")


def main():
    args = parse_args()
    hf_hub_download, snapshot_download = _ensure_huggingface_hub()

    token = args.token or os.environ.get("HF_TOKEN")
    repo_root = Path(__file__).resolve().parent

    if args.training or args.sample:
        output_dir = args.output_dir if args.output_dir else repo_root
    else:
        output_dir = args.output_dir if args.output_dir else repo_root / "gear_sonic_deploy"

    output_dir = Path(output_dir).expanduser().resolve()

    print("=" * 60)
    print("  GEAR-SONIC — Hugging Face Model Downloader")
    print(f"  Repository : {REPO_ID}")
    print(f"  Output dir : {output_dir}")
    if args.training:
        print("  Mode       : training (checkpoint + SMPL data)")
    elif args.sample:
        print("  Mode       : sample data (quick start)")
    else:
        print("  Mode       : deployment (ONNX models)")
    print("=" * 60)

    if args.sample:
        print("\n[Sample Data]")
        download_sample_data(snapshot_download, REPO_ID, output_dir, token=token)

    elif args.training:
        print("\n[Checkpoint]")
        for hf_filename, local_rel in TRAINING_FILES:
            download_file(
                hf_hub_download,
                REPO_ID,
                hf_filename,
                output_dir / local_rel,
                token=token,
            )

        if not args.no_smpl:
            print("\n[SMPL Motion Data]")
            download_and_extract_smpl(
                hf_hub_download,
                REPO_ID,
                output_dir,
                token=token,
                keep_parts=args.keep_smpl_parts,
            )
        else:
            print("\n[SMPL Motion Data] Skipped (--no-smpl)")

    else:
        print("\n[Policy]")
        for hf_filename, local_rel in POLICY_FILES:
            download_file(
                hf_hub_download,
                REPO_ID,
                hf_filename,
                output_dir / local_rel,
                token=token,
            )

        if not args.no_planner:
            print("\n[Planner]")
            hf_filename, local_rel = PLANNER_FILE
            download_file(
                hf_hub_download,
                REPO_ID,
                hf_filename,
                output_dir / local_rel,
                token=token,
            )

    print("\n" + "=" * 60)
    print("  Done! Files saved under:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
