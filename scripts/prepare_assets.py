"""Download pinned assets. No credentials are printed or stored by this script."""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import ROOT, atomic_json, load_config, resolve, sha256_file


def download(url, destination):
    destination = resolve(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(url, timeout=120) as response, open(temporary, "wb") as f:
        import shutil
        shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
    temporary.replace(destination)
    return destination


def unpack_zip(path, destination):
    destination = resolve(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive member: {info.filename}")
        bad = z.testzip()
        if bad:
            raise ValueError(f"Archive CRC failed: {bad}")
        z.extractall(destination)


def prepare_models(config, public_only=False):
    from huggingface_hub import snapshot_download
    records, errors = {}, {}
    for role, model in config["models"].items():
        if public_only and role == "perceptor":
            continue
        try:
            directory = resolve(model["path"])
            print(f"Downloading {role}: {model['repo']}@{model['revision']}", flush=True)
            snapshot_download(model["repo"], revision=model["revision"], local_dir=directory,
                              max_workers=4, allow_patterns=["*.json", "*.safetensors", "*.bin",
                                                            "*.model", "*.txt", "*.tiktoken"])
            files = {str(p.relative_to(directory)): {"bytes": p.stat().st_size,
                                                     "sha256": sha256_file(p)}
                     for p in sorted(directory.rglob("*"))
                     if p.is_file() and ".cache" not in p.parts and p.name != "asset_lock.json"}
            records[role] = {**model, "files": files, "complete": True}
            atomic_json(directory / "asset_lock.json", records[role])
        except Exception as exc:
            # Exception text can contain signed download URLs; only report its class.
            errors[role] = type(exc).__name__
            print(f"{role}: {type(exc).__name__}; verify HF authentication/access", flush=True)
    atomic_json(resolve("runs/assets/models_download.json"), {"models": records, "errors": errors})
    return not errors


def prepare_annotations(include_pathmmu=True):
    bcnb = download("https://huggingface.co/datasets/General-Medical-AI/SlideChat/resolve/"
                    "c8128b91b9bcb38b395961633232dba4dfca81f2/SlideBench-VQA-BCNB.csv",
                    "data/raw/bcnb/SlideBench-VQA-BCNB.csv")
    expected = "10b83c6177bd13bee9471fb0bd54b7d500e7eccdb7e52b462424ed50cb259e0b"
    if sha256_file(bcnb) != expected:
        raise ValueError("BCNB annotation checksum mismatch")
    import gdown
    wsi = resolve("data/raw/wsi_vqa/WsiVQA_test.json")
    wsi.parent.mkdir(parents=True, exist_ok=True)
    if not wsi.exists():
        gdown.download(id="1eSQaZ-hKRUDCerGKkPW8VtfOQPgFOMD7", output=str(wsi), quiet=True)
    with open(wsi) as f:
        json.load(f)
    if include_pathmmu:
        from huggingface_hub import snapshot_download
        destination = resolve("data/raw/pathmmu")
        snapshot_download("jamessyx/PathMMU", repo_type="dataset",
                          revision="054e64e56e599e9636024f1471d49ecae4a2784f", local_dir=destination,
                          allow_patterns=["data.json", "images.zip", "instructions.md",
                                          "construct_pathcls.py", "socialpath_mapping.json"])
        unpack_zip(destination / "images.zip", destination)


def prepare_external(config):
    repos = {"PLIP": ("https://github.com/PathologyFoundation/plip.git", None),
             "TRIDENT": ("https://github.com/mahmoodlab/TRIDENT.git",
                         config["preprocessing"]["trident_revision"])}
    for name, (url, revision) in repos.items():
        destination = resolve("external") / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not (destination / ".git").exists():
            subprocess.run(["git", "clone", url, str(destination)], check=True)
        if revision:
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
        sha = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        atomic_json(resolve(f"runs/assets/{name.lower()}_lock.json"), {"repo": url, "revision": sha})


def prepare_segmentation(config):
    from huggingface_hub import hf_hub_download
    model = config["preprocessing"]["segmentation_model"]
    directory = resolve(model["path"])
    path = hf_hub_download(model["repo"], model["filename"], revision=model["revision"], local_dir=directory)
    atomic_json(directory / "asset_lock.json", {**model, "sha256": sha256_file(path), "complete": True})
    registry = resolve(config["preprocessing"]["trident_path"]) / "trident/segmentation_models/local_ckpts.json"
    existing = json.loads(registry.read_text())
    existing["hest"] = str(Path(path).resolve())
    atomic_json(registry, existing)
    # The upstream constructor initializes this backbone before loading all HEST weights.
    download("https://download.pytorch.org/models/resnet50-0676ba61.pth",
             "checkpoints/torch/hub/checkpoints/resnet50-0676ba61.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["models", "annotations", "external", "bcnb", "segmentation"])
    parser.add_argument("--public-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.stage == "models":
        if not prepare_models(config, args.public_only):
            raise SystemExit(2)
    elif args.stage == "annotations":
        prepare_annotations(not args.public_only)
    elif args.stage == "external":
        prepare_external(config)
    elif args.stage == "segmentation":
        prepare_segmentation(config)
    else:
        import gdown
        destination = resolve("data/raw/bcnb/downloads")
        destination.mkdir(parents=True, exist_ok=True)
        paths = gdown.download_folder(id="1HcAgplKwbSZ7ZZl2m6PZdvVF70QJmVuR",
                                      output=str(destination), quiet=False,
                                      remaining_ok=False, resume=True)
        if not paths:
            raise RuntimeError("BCNB folder download returned no files")
        for path in paths:
            if str(path).lower().endswith(".zip"):
                unpack_zip(path, "data/raw/bcnb/images")


if __name__ == "__main__":
    main()
