"""Download two public diagnostic TCGA slides; record every mapping decision."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, read_json, resolve, sha256_file
from scripts.prepare_assets import download
from data_processing.regions import ImageSource


def main():
    import requests
    rows = read_json(resolve("data/raw/wsi_vqa/WsiVQA_test.json"))
    groups = defaultdict(list)
    for row in rows:
        groups[row["Id"]].append(row)
    cases = sorted(k for k, v in groups.items() if sum("Choice" in x for x in v) >= 2
                   and sum("Choice" not in x for x in v) >= 2)
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.submitter_id", "value": cases}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}}]}
    response = requests.get("https://api.gdc.cancer.gov/files", params={
        "filters": json.dumps(filters), "fields": "file_id,file_name,file_size,md5sum,cases.submitter_id,access",
        "size": 1000}, timeout=120)
    response.raise_for_status()
    data = response.json()
    if data["data"]["pagination"]["total"] > len(data["data"]["hits"]):
        raise RuntimeError("GDC query was truncated")
    atomic_json(resolve("data/raw/wsi_vqa/gdc_files.json"), data)
    candidates = [r for r in data["data"]["hits"] if "-DX" in r["file_name"] and r["access"] == "open"]
    # Smallest public diagnostic slides reduce smoke I/O; selection never uses predictions.
    selected, seen, rejected = [], set(), []
    for record in sorted(candidates, key=lambda r: (r["file_size"], r["file_name"])):
        case = record["file_name"][:12]
        if case in seen:
            continue
        path = download("https://api.gdc.cancer.gov/data/" + record["file_id"],
                        "data/raw/wsi_vqa/slides/" + record["file_name"])
        with ImageSource(path) as source:
            if source.mpp is None:
                rejected.append({"file_id": record["file_id"], "reason": "no physical-scale metadata"})
                continue
            record["mpp"] = source.mpp
            record["magnification_source"] = source.mpp_source
        seen.add(case)
        selected.append(record)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("Could not map two smoke cases to diagnostic slides")
    mapping = {}
    for record in selected:
        path = download("https://api.gdc.cancer.gov/data/" + record["file_id"],
                        "data/raw/wsi_vqa/slides/" + record["file_name"])
        checksum = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != record["md5sum"] or path.stat().st_size != record["file_size"]:
            raise ValueError("Downloaded TCGA slide failed MD5/size verification")
        record["sha256"] = sha256_file(path)
        mapping[record["file_name"][:12]] = record["file_name"]
    atomic_json(resolve("data/raw/wsi_vqa/slide_map.json"), mapping)
    atomic_json(resolve("data/raw/wsi_vqa/smoke_download_lock.json"), {
        "protocol": "two smallest public diagnostic DX slides with physical-scale metadata and >=2 closed and >=2 open test questions",
        "scope": "smoke only; not a published patient-to-slide reproduction mapping", "files": selected,
        "rejected": rejected})
    print(mapping)


if __name__ == "__main__":
    main()
