from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml


CHUNK_SIZE = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external dataset files declared in a YAML manifest.")
    parser.add_argument("--manifest", default="configs/unsw_nb15_zenodo.yaml")
    parser.add_argument("--dest", default="data/raw/unsw_nb15")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dest_dir = Path(args.dest)
    manifest = load_manifest(manifest_path)
    files = manifest.get("files", [])

    if not files:
        raise SystemExit(f"No files declared in {manifest_path}")

    total_mb = sum(float(file_info.get("size_mb", 0.0)) for file_info in files)
    print(f"Dataset: {manifest.get('name', manifest_path)}")
    print(f"Landing page: {manifest.get('landing_page', 'n/a')}")
    print(f"Destination: {dest_dir}")
    print(f"Planned download size: {total_mb:.1f} MB")
    check_disk_space(dest_dir, total_mb)

    if args.dry_run:
        for file_info in files:
            print(f"DRY RUN {file_info['name']} <- {file_info['url']}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    for file_info in files:
        download_file(file_info, dest_dir, force=args.force)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file) or {}


def check_disk_space(dest_dir: Path, total_mb: float) -> None:
    probe_dir = dest_dir
    while not probe_dir.exists() and probe_dir.parent != probe_dir:
        probe_dir = probe_dir.parent

    free_mb = shutil.disk_usage(probe_dir).free / (1024 * 1024)
    required_mb = total_mb * 1.2
    if free_mb < required_mb:
        raise SystemExit(
            f"Not enough free disk space under {probe_dir}. "
            f"Need about {required_mb:.1f} MB, found {free_mb:.1f} MB."
        )


def download_file(file_info: dict[str, Any], dest_dir: Path, force: bool) -> None:
    output_path = dest_dir / file_info["name"]
    expected_md5 = str(file_info["md5"])

    if output_path.exists() and not force:
        actual_md5 = md5sum(output_path)
        if actual_md5 == expected_md5:
            print(f"OK existing {output_path}")
            return
        raise SystemExit(
            f"{output_path} exists but MD5 is {actual_md5}, expected {expected_md5}. "
            "Rerun with --force to replace it."
        )

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading {file_info['name']}")
    with urllib.request.urlopen(file_info["url"]) as response, tmp_path.open("wb") as output_file:
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded += len(chunk)
            print_progress(downloaded, file_info.get("size_mb"))

    print()
    actual_md5 = md5sum(tmp_path)
    if actual_md5 != expected_md5:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"MD5 mismatch for {file_info['name']}: {actual_md5} != {expected_md5}")

    tmp_path.replace(output_path)
    print(f"Wrote {output_path}")


def print_progress(downloaded_bytes: int, size_mb: Any) -> None:
    if not size_mb:
        print(f"\r  {downloaded_bytes / (1024 * 1024):.1f} MB", end="", flush=True)
        return

    expected_bytes = float(size_mb) * 1_000_000
    pct = min(downloaded_bytes / expected_bytes * 100, 100)
    print(f"\r  {downloaded_bytes / (1024 * 1024):.1f} MB ({pct:5.1f}%)", end="", flush=True)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
