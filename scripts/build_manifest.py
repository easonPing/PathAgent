import argparse
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import digest, resolve, sha256_file, write_manifest
from data_processing.datasets import load_benchmark, summarize
from data_processing.regions import ImageSource


def select_smoke(samples, dataset):
    if dataset == "pathmmu":
        groups = defaultdict(list)
        for row in samples:
            groups[row["source"]].append(row)
        return [row for source in sorted(groups)
                for row in sorted(groups[source], key=lambda r: digest(r["sample_id"]))[:2]]
    groups = defaultdict(list)
    for row in samples:
        groups[row["image_id"]].append(row)
    if dataset == "bcnb":
        areas = []
        for image, rows in groups.items():
            with ImageSource(rows[0]["image_path"], rows[0]["assumed_mpp"]) as src:
                areas.append((src.width * src.height, image))
        areas.sort()
        if len(areas) < 3:
            raise ValueError("BCNB smoke needs three original slides")
        return [r for i in [0, len(areas) // 2, len(areas) - 1] for r in groups[areas[i][1]]]
    eligible = [(image, rows) for image, rows in groups.items()
                if sum(bool(r["choices"]) for r in rows) >= 2
                and sum(not r["choices"] for r in rows) >= 2
                and Path(rows[0]["image_path"]).is_file()]
    if len(eligible) < 2:
        raise ValueError("WSI smoke requires two mapped original slides, each with 2 closed and 2 open QA")
    result = []
    for image, rows in sorted(eligible)[:2]:
        for closed in [True, False]:
            result.extend(sorted([r for r in rows if bool(r["choices"]) == closed],
                                 key=lambda r: digest(r["sample_id"]))[:2])
    return result


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["pathmmu", "bcnb", "wsi_vqa"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--available-images", action="store_true",
                        help="PathMMU execution check using downloaded originals only; not full-source smoke acceptance")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(f"configs/datasets/{args.dataset}.yaml").read_text())
    # Smoke validates its selected originals; unrelated missing test images must not
    # block this small execution check. Full manifests still require every image.
    partial_smoke = args.smoke and args.dataset in {"wsi_vqa", "pathmmu"}
    samples = load_benchmark(cfg, require_images=not (args.annotations_only or partial_smoke))
    coverage = None
    if args.available_images:
        if not args.smoke or args.dataset != "pathmmu":
            parser.error("--available-images is only for PathMMU --smoke")
        available = [r for r in samples if Path(r["image_path"]).is_file()]
        coverage = {"release": summarize(samples), "available": summarize(available),
                    "missing_questions": len(samples) - len(available)}
        if not available:
            raise ValueError("No downloaded PathMMU originals available")
        samples = available
    if args.smoke:
        if args.annotations_only:
            parser.error("Smoke manifests must contain real original images")
        samples = select_smoke(samples, args.dataset)
    for row in samples:
        if not args.annotations_only and not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    if not args.annotations_only:
        checked = set()
        for row in samples:
            if row["image_id"] in checked:
                continue
            checked.add(row["image_id"])
            with ImageSource(row["image_path"], row["assumed_mpp"]) as source:
                if row["input_kind"] == "wsi" and source.mpp is None:
                    raise ValueError(f"No physical-scale metadata or declared MPP for {row['image_id']}")
    name = "smoke_available" if args.available_images else "smoke" if args.smoke else "annotations" if args.annotations_only else "full"
    output = resolve(f"data/manifests/{args.dataset}/{name}.json")
    write_manifest(output, samples, dataset=args.dataset, purpose="smoke" if args.smoke else name, dataset_config=cfg,
                   annotation_sha256=sha256_file(resolve(cfg["annotations"])),
                   summary=summarize(samples), images_verified=not args.annotations_only,
                   full_source_smoke=bool(args.smoke and not args.available_images),
                   availability_restricted=args.available_images, coverage=coverage)
    print(output)
    print(summarize(samples))


if __name__ == "__main__":
    main()
