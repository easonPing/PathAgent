"""Explicit, strict benchmark adapters. Gold labels never enter model inputs."""
import csv
import re
from collections import Counter
from pathlib import Path

from data_processing.common import digest, read_json, resolve, sha256_file

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs", ".ndpi", ".mrxs"}
PATHMMU_SOURCES = ("PubMed", "SocialPath", "Atlas", "EduContent", "PathCLS")


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_choices(choices):
    if not choices:
        return {}
    if isinstance(choices, str):
        # Accept JSON lists, or a labelled A. ... B. ... string; never eval().
        import json
        try:
            choices = json.loads(choices)
        except json.JSONDecodeError:
            chunks = re.split(r"(?:^|\s)([A-Z])[.)]\s*", choices)
            if len(chunks) < 3:
                raise ValueError("Unrecognized choices string")
            choices = {chunks[i]: chunks[i + 1].strip() for i in range(1, len(chunks), 2)}
    if isinstance(choices, dict):
        result = {str(k).strip("(). ").upper(): str(v).strip() for k, v in choices.items()
                  if v is not None and str(v).strip()}
    else:
        result = {}
        for i, choice in enumerate(choices):
            if choice is None or not str(choice).strip():
                continue
            text = str(choice).strip()
            match = re.match(r"^\(?([A-Z])\)?[.)]\s*(.+)$", text, flags=re.S)
            label, text = (match.group(1), match.group(2)) if match else (chr(65 + i), text)
            result[label] = text.strip()
    if len(result) < 2 or any(not re.fullmatch(r"[A-Z]", k) for k in result):
        raise ValueError("Expected at least two labelled, nonempty choices")
    return result


def answer_label(answer, choices):
    """Parse an unambiguous label/full option; never use gold or fuzzy similarity."""
    if not choices or answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    label_match = re.fullmatch(r"\(?([A-Z])\)?[.)]?", text.upper())
    if label_match and label_match.group(1) in choices:
        return label_match.group(1)
    labelled = re.match(r"^\(?([A-Z])\)?[.)]\s*(.+)$", text, flags=re.S)
    if labelled:
        label, body = labelled.groups()
        if label in choices and normalize_text(body) == normalize_text(choices[label]):
            return label
        return None
    matches = [k for k, v in choices.items() if normalize_text(v) == normalize_text(text)]
    return matches[0] if len(matches) == 1 else None


def image_index(root):
    found = {}
    for p in sorted(resolve(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.setdefault(p.stem, []).append(str(p.resolve()))
    return found


def unique_image(index, image_id):
    paths = index.get(str(image_id), [])
    if len(paths) != 1:
        raise ValueError(f"Image {image_id}: expected one original image, found {paths}")
    return paths[0]


def make_sample(dataset, split, sid, image_id, image_path, question, choices, answer,
                task="", source="", input_kind="wsi", assumed_mpp=None):
    choices = normalize_choices(choices)
    label = answer_label(answer, choices) if choices else None
    if choices and label is None:
        raise ValueError(f"Unmappable ground truth for {dataset}/{sid}: {answer!r}")
    if not str(question).strip():
        raise ValueError(f"Empty question for {dataset}/{sid}")
    return {"dataset": dataset, "split": split, "sample_id": f"{dataset}:{sid}",
            "image_id": str(image_id), "image_path": str(image_path), "question": str(question),
            "choices": choices, "ground_truth": str(answer), "ground_truth_label": label,
            "task": task, "source": source, "input_kind": input_kind,
            "assumed_mpp": assumed_mpp}


def load_bcnb(annotation, image_root, assumed_mpp=0.5, require_images=True):
    index = image_index(image_root) if require_images else {}
    samples = []
    with open(resolve(annotation), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_id = row["Slide"].strip()
            image = unique_image(index, image_id) if require_images else ""
            choices = {k: row[k] for k in "ABCD" if row.get(k, "").strip()}
            samples.append(make_sample("bcnb", "test", row["ID"], image_id, image,
                                       row["Question"], choices, row["Answer"], row["Task"],
                                       assumed_mpp=assumed_mpp))
    return samples


def load_pathmmu(annotation, image_root, require_images=True):
    data = read_json(resolve(annotation))
    samples, identities = [], set()
    for source in PATHMMU_SOURCES:
        if source not in data:
            raise ValueError(f"PathMMU is missing source {source}")
        for split in ("test", "test_tiny"):
            if split not in data[source]:
                raise ValueError(f"PathMMU is missing {source}/{split}")
            for row in data[source][split]:
                identity = digest([source, row["img"], row["question"], row["options"]])
                if identity in identities:
                    raise ValueError(f"Duplicate PathMMU test/test_tiny sample: {source}/{row['No']}")
                identities.add(identity)
                path = (resolve(image_root) / row["img"]).resolve()
                if not path.is_relative_to(resolve(image_root).resolve()):
                    raise ValueError("Image path escapes image_root")
                if require_images and not path.is_file():
                    raise FileNotFoundError(path)
                samples.append(make_sample("pathmmu", split, f"{source}:{split}:{row['No']}",
                                           row["img"], path, row["question"], row["options"],
                                           row["answer"], source=source, input_kind="roi"))
    return samples


def load_wsi_vqa(annotation, image_root, slide_map, require_images=True):
    data = read_json(resolve(annotation))
    mapping = read_json(resolve(slide_map)) if resolve(slide_map).exists() else {}
    samples = []
    for i, row in enumerate(data):
        case = str(row["Id"])
        mapped = mapping.get(case)
        if mapped is None:
            if require_images:
                raise ValueError(f"No explicit TCGA slide mapping for case {case}")
            image_id, path = case, ""
        else:
            image_id = Path(mapped).stem
            path = str((resolve(image_root) / mapped).resolve())
            if require_images and not Path(path).is_file():
                raise FileNotFoundError(path)
        samples.append(make_sample("wsi_vqa", "test", str(i), image_id, path,
                                   row["Question"], row.get("Choice"), row["Answer"]))
    return samples


def load_benchmark(config, require_images=True):
    name = config["name"]
    args = (config["annotations"], config["image_root"])
    if name == "bcnb":
        expected_hash = config.get("annotation_sha256")
        if expected_hash and sha256_file(resolve(args[0])) != expected_hash:
            raise ValueError("BCNB annotations differ from the pinned release")
        samples = load_bcnb(*args, assumed_mpp=config.get("assumed_mpp"), require_images=require_images)
    elif name == "pathmmu":
        samples = load_pathmmu(*args, require_images=require_images)
    elif name == "wsi_vqa":
        samples = load_wsi_vqa(*args, config["slide_map"], require_images=require_images)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    for key, actual in [("expected_questions", len(samples)),
                        ("expected_images", len({r['image_id'] for r in samples}))]:
        if key in config and actual != config[key]:
            raise ValueError(f"{name} {key}: expected {config[key]}, found {actual}; investigate release")
    return samples


def public_input(sample):
    return {k: v for k, v in sample.items() if k not in {"ground_truth", "ground_truth_label",
                                                       "answer", "explanation"}}


def summarize(samples):
    return {"questions": len(samples), "images": len({r["image_id"] for r in samples}),
            "tasks": dict(Counter(r["task"] for r in samples)),
            "sources": dict(Counter(r["source"] for r in samples))}
