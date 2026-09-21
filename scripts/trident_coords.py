"""Run only segmentation/coordinates under the isolated Trident Python environment."""
import argparse
import json
import os
from pathlib import Path


def main():
    os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "checkpoints/torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
    import torch
    import h5py
    from PIL import Image
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Trident requires an allocated GPU")
    Image.MAX_IMAGE_PIXELS = None
    torch.manual_seed(request["seed"])
    torch.cuda.manual_seed_all(request["seed"])
    config = request["preprocessing"]
    model = segmentation_model_factory(model_name=config["segmenter"],
                                       confidence_thresh=config["segmentation_threshold"])
    slide = load_wsi(request["image_path"], reader_type=request["reader_type"],
                     mpp=request["mpp"], lazy_init=False)
    # Pin the actual physical scale instead of Trident's rounded MPP bucket.
    slide.mag = 10.0 / request["mpp"]
    job = request["job_dir"]
    slide.segment_tissue(segmentation_model=model, target_mag=model.target_mag,
                         job_dir=job, device="cuda:0", holes_are_tissue=not config["remove_holes"],
                         num_workers=request["workers"])
    coords_path = slide.extract_tissue_coords(target_mag=slide.mag,
        patch_size=config["level0_patch_size"], save_coords=job, overlap=config["overlap"],
        min_tissue_proportion=config["min_tissue_proportion"])
    with h5py.File(coords_path) as h5:
        coords = h5["coords"][:].tolist()
    if not coords:
        raise RuntimeError("Trident returned no tissue regions; no silent whole-slide fallback")
    destination = Path(request["output"])
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({"coords": coords, "request": request}))
    temporary.replace(destination)


if __name__ == "__main__":
    main()
