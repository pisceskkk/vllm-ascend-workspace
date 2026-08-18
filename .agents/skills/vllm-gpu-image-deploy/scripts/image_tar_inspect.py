#!/usr/bin/env python3
"""Inspect one uncompressed Docker-save tar without loading it."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tarfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-tar", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_tar = args.image_tar.expanduser().resolve()
    if not image_tar.is_file():
        raise SystemExit(f"image tar does not exist: {image_tar}")

    with tarfile.open(image_tar, mode="r:") as archive:
        manifest_file = archive.extractfile("manifest.json")
        if manifest_file is None:
            raise SystemExit("Docker-save tar has no manifest.json")
        manifest = json.load(manifest_file)
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise SystemExit("expected exactly one image in Docker-save tar")
        entry = manifest[0]
        config_name = str(entry.get("Config", ""))
        if not config_name.endswith(".json"):
            raise SystemExit("manifest Config is missing or invalid")
        config_file = archive.extractfile(config_name)
        if config_file is None:
            raise SystemExit(f"Docker-save tar is missing config: {config_name}")
        config = json.load(config_file)

    env = {}
    for item in config.get("config", {}).get("Env", []) or []:
        key, separator, value = str(item).partition("=")
        if separator:
            env[key] = value
    image_id = f"sha256:{pathlib.PurePosixPath(config_name).stem}"
    diff_ids = list(config.get("rootfs", {}).get("diff_ids") or [])
    rootfs_sha256 = hashlib.sha256("".join(f"{item}\n" for item in diff_ids).encode()).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "ok",
        "path": str(image_tar),
        "size_bytes": image_tar.stat().st_size,
        "image_id": image_id,
        "repo_tags": list(entry.get("RepoTags") or []),
        "created": config.get("created"),
        "architecture": config.get("architecture"),
        "os": config.get("os"),
        "vllm_build_commit": env.get("VLLM_BUILD_COMMIT"),
        "vllm_image_tag": env.get("VLLM_IMAGE_TAG"),
        "cuda_version": env.get("CUDA_VERSION"),
        "layer_count": len(entry.get("Layers") or []),
        "rootfs_sha256": rootfs_sha256,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
