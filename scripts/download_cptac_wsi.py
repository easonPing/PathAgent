"""Download only the 240 SlideBench-VQA-CPTAC WSIs from official TCIA Faspex packages.

The script discovers the current Tissue Slide Images package on each official
collection page, inventories it, requires one exact .svs basename match per
Slide ID, and only then invokes Aspera. Re-running the download is resumable.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "data/raw/cptac/SlideBench-VQA-CPTAC.csv"
DATA_ROOT = ROOT / "data/raw/cptac"
LIST_ROOT = DATA_ROOT / "download_lists"
IMAGE_ROOT = DATA_ROOT / "images"
COLLECTIONS = {
    "CM": "cptac-cm",
    "LSCC": "cptac-lscc",
    "LUAD": "cptac-luad",
    "UCEC": "cptac-ucec",
}


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def targets():
    result = defaultdict(set)
    with ANNOTATION.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result[row["Tumor"].strip()].add(row["Slide"].strip())
    if set(result) != set(COLLECTIONS) or any(len(result[k]) != 60 for k in COLLECTIONS):
        raise RuntimeError(f"Expected 60 unique IDs in each cohort, found {dict(map(lambda x: (x[0], len(x[1])), result.items()))}")
    return result


def package_urls(cohort):
    collection = COLLECTIONS[cohort]
    page_url = f"https://www.cancerimagingarchive.net/collection/{collection}/"
    request = urllib.request.Request(page_url, headers={"User-Agent": "PathAgent CPTAC downloader/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        page = response.read().decode("utf-8", errors="replace")
    # The page lists the current release before previous versions. Retain older
    # tissue packages as fallbacks because TCIA occasionally returns HTTP 500
    # while recursively browsing a newly published Faspex package.
    pattern = re.compile(
        r"<tr[^>]*>\s*<td>Tissue Slide.*?</td>.*?"
        r"href=\"(https://faspex\.cancerimagingarchive\.net/aspera/faspex[^\"]+)\"",
        flags=re.I | re.S,
    )
    matches = pattern.findall(page)
    if not matches:
        raise RuntimeError(f"No current Tissue Slide Images Faspex package found at {page_url}")
    import base64
    result, seen = [], set()
    for raw_url in matches:
        url = html.unescape(raw_url)
        context = re.search(r"[?&]context=([^&]+)", url)
        if not context:
            raise RuntimeError(f"Faspex URL lacks context: {page_url}")
        decoded = json.loads(base64.b64decode(context.group(1)).decode())
        package_id = str(decoded.get("package_id"))
        if decoded.get("resource") != "packages" or package_id != str(decoded.get("id")):
            raise RuntimeError(f"Unexpected Faspex context at {page_url}")
        if package_id not in seen:
            result.append((page_url, url, package_id))
            seen.add(package_id)
    return result


def ascli_base(args):
    return [args.ascli, f"--home={args.ascli_home}", "--interactive=no", "--progress-bar=no"]


def inventory(args, cohort, package_id, url):
    LIST_ROOT.mkdir(parents=True, exist_ok=True)
    destination = LIST_ROOT / f"{cohort}_package_{package_id}.csv"
    command = ascli_base(args) + [
        "--format=csv", "faspex5", "packages", "browse",
        '--query=@json:{"recursive":true}', f"--url={url}",
    ]
    with destination.open("w", encoding="utf-8", newline="") as output:
        subprocess.run(command, stdout=output, check=True)
    if destination.stat().st_size == 0:
        raise RuntimeError(f"Empty package inventory for {cohort}")
    return destination


def inventory_covers(path, wanted):
    counts = Counter()
    for row in inventory_rows(path):
        basename = row["basename"].strip()
        if row["type"].strip() in {"file", "symbolic_link"} and basename.lower().endswith(".svs"):
            slide = basename[:-4]
            if slide in wanted:
                counts[slide] += 1
    return len(counts) == len(wanted) and all(value == 1 for value in counts.values())


def inventory_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.reader(stream))
    if not raw:
        raise RuntimeError(f"Empty ascli inventory in {path}")
    if raw[0][:4] == ["path", "basename", "type", "size"]:
        keys, values = raw[0], raw[1:]
        return [dict(zip(keys, row)) for row in values]
    # ascli 4.27 CSV output omits headers. The first six fields correspond to
    # path, basename, type, size, mtime and permissions; symbolic links may add
    # target.* fields after those. Directories omit size and mtime.
    result = []
    for row in raw:
        if len(row) < 3:
            raise RuntimeError(f"Malformed ascli inventory row in {path}: {row}")
        item_type = row[2].strip()
        size = ""
        if item_type == "file" and len(row) >= 4 and row[3].strip().isdigit():
            size = row[3].strip()
        elif item_type == "symbolic_link":
            # Faspex flattens the target fields after link permissions. Use the
            # target file's size, which immediately follows its type marker.
            for index in range(3, len(row) - 1):
                if row[index].strip() == "file" and row[index + 1].strip().isdigit():
                    size = row[index + 1].strip()
                    break
        result.append({"path": row[0], "basename": row[1], "type": item_type, "size": size})
    return result


def matched_manifest(target_ids, inventories, package_info):
    candidates = defaultdict(list)
    for cohort, path in inventories.items():
        for row in inventory_rows(path):
            basename = row["basename"].strip()
            if row["type"].strip() not in {"file", "symbolic_link"} or not basename.lower().endswith(".svs"):
                continue
            slide = basename[:-4]
            if slide in target_ids[cohort]:
                candidates[(cohort, slide)].append(row)
    failures = {f"{cohort}:{slide}": rows for cohort in COLLECTIONS for slide in sorted(target_ids[cohort])
                if len(rows := candidates[(cohort, slide)]) != 1}
    if failures:
        counts = Counter(len(rows) for rows in failures.values())
        raise RuntimeError(f"Every Slide ID must have one exact .svs match; failures by match count: {dict(counts)}; "
                           f"first IDs: {list(failures)[:20]}")
    rows = []
    for cohort in COLLECTIONS:
        for slide in sorted(target_ids[cohort]):
            remote = candidates[(cohort, slide)][0]
            size = int(remote["size"])
            rows.append({
                "cohort": cohort,
                "slide_id": slide,
                "remote_path": remote["path"],
                "remote_basename": remote["basename"],
                "size_bytes": size,
                "package_id": package_info[cohort]["package_id"],
                "source_page": package_info[cohort]["source_page"],
            })
    LIST_ROOT.mkdir(parents=True, exist_ok=True)
    path = LIST_ROOT / "matched_wsi.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def download(args, rows, package_info):
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cohort"]].append(row["remote_path"].lstrip("/"))
    def transfer_cohort(cohort):
        remote_paths = grouped[cohort]
        if len(remote_paths) != 60:
            raise RuntimeError(f"Refusing {cohort} transfer with {len(remote_paths)} paths")
        destination = IMAGE_ROOT / cohort
        destination.mkdir(parents=True, exist_ok=True)
        command = ascli_base(args) + [
            "faspex5", "packages", "receive",
            f"--url={package_info[cohort]['url']}", *remote_paths,
        ]
        log = LIST_ROOT / f"{cohort}_download.log"
        with log.open("a", encoding="utf-8") as output:
            output.write(f"Starting package {package_info[cohort]['package_id']} with {len(remote_paths)} exact paths\n")
            output.flush()
            subprocess.run(command, cwd=destination, stdout=output, stderr=subprocess.STDOUT, check=True)
        return cohort

    # Each worker is one ascli/ascp process and never decodes a WSI. Keep the
    # default at two to stay far below the account's memory ceiling.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transfer_cohort, cohort): cohort for cohort in COLLECTIONS}
        for future in as_completed(futures):
            cohort = futures[future]
            future.result()
            print(f"Completed cohort transfer: {cohort}", flush=True)


def verify(rows):
    found = defaultdict(list)
    for path in IMAGE_ROOT.rglob("*.svs") if IMAGE_ROOT.exists() else []:
        found[path.stem].append(path)
    problems = []
    completed = []
    for row in rows:
        paths = found[row["slide_id"]]
        exact = [p for p in paths if p.stat().st_size == int(row["size_bytes"])]
        if len(exact) == 1:
            completed.append((row, exact[0]))
        else:
            problems.append({"slide_id": row["slide_id"], "expected_size": int(row["size_bytes"]),
                             "paths": [{"path": str(p), "size": p.stat().st_size} for p in paths]})
    state = {
        "expected": len(rows), "complete": len(completed), "problems": problems,
        "bytes_complete": sum(int(row["size_bytes"]) for row, _ in completed),
        "files": [{"slide_id": row["slide_id"], "path": str(path.relative_to(ROOT)),
                   "size_bytes": int(row["size_bytes"]), "sha256": sha256(path)}
                  for row, path in completed],
    }
    (DATA_ROOT / "wsi_inventory.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inventory", "download", "verify", "all"])
    parser.add_argument("--ascli", default=str(ROOT / ".tools/ruby-gems/bin/ascli"))
    parser.add_argument("--ascli-home", default=str(ROOT / ".tools/ascli-home"))
    parser.add_argument("--refresh-inventory", action="store_true",
                        help="Re-read package listings instead of reusing valid completed CSV inventories")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent cohort transfers (default: 2; allowed: 1-4)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    target_ids = targets()
    package_info, inventories = {}, {}
    for cohort in COLLECTIONS:
        attempts = []
        for source_page, url, package_id in package_urls(cohort):
            path = LIST_ROOT / f"{cohort}_package_{package_id}.csv"
            reusable = False
            if path.is_file() and path.stat().st_size and not args.refresh_inventory:
                try:
                    reusable = inventory_covers(path, target_ids[cohort])
                except RuntimeError:
                    pass
            if args.action in {"inventory", "all"} and not reusable:
                try:
                    path = inventory(args, cohort, package_id, url)
                except subprocess.CalledProcessError as error:
                    attempts.append(f"package {package_id}: browse failed ({error.returncode})")
                    continue
            if path.is_file() and path.stat().st_size:
                try:
                    if inventory_covers(path, target_ids[cohort]):
                        inventories[cohort] = path
                        package_info[cohort] = {"source_page": source_page, "url": url,
                                                "package_id": package_id}
                        break
                    attempts.append(f"package {package_id}: does not uniquely cover all 60 IDs")
                except RuntimeError as error:
                    attempts.append(f"package {package_id}: {error}")
        if cohort not in inventories:
            raise RuntimeError(f"No usable Tissue Slide package for {cohort}: {'; '.join(attempts)}")
    manifest, rows = matched_manifest(target_ids, inventories, package_info)
    total = sum(int(row["size_bytes"]) for row in rows)
    summary = {"manifest": str(manifest), "files": len(rows), "bytes": total,
               "gib": round(total / 1024 ** 3, 3),
               "packages": {k: {x: y for x, y in v.items() if x != "url"} for k, v in package_info.items()}}
    (LIST_ROOT / "download_plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.action in {"download", "all"}:
        download(args, rows, package_info)
    if args.action in {"verify", "all"}:
        state = verify(rows)
        print(json.dumps({k: state[k] for k in ("expected", "complete", "bytes_complete")}, indent=2))
        if state["problems"]:
            raise RuntimeError(f"Download verification failed for {len(state['problems'])} slides")


if __name__ == "__main__":
    main()
