"""Regions always refer to original pixels. Zoom never crops a downsampled cache."""
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    region_id: str
    image_path: str
    x: int
    y: int
    width: int
    height: int
    output_width: int
    output_height: int
    scale_kind: str
    scale: float
    mpp: object = None
    magnification_source: str = "unknown"
    parent_id: object = None

    def to_dict(self):
        return asdict(self)

    def metadata(self):
        name = "Magnification" if self.scale_kind == "physical" else "Relative crop scale"
        return (f"Region={self.region_id}; Original-pixel bounds=({self.x},{self.y},"
                f"{self.width},{self.height}); {name}={self.scale}x; "
                f"Scale source={self.magnification_source}")


class ImageSource:
    def __init__(self, path, assumed_mpp=None):
        from PIL import Image
        self.path = str(Path(path).resolve())
        self.slide, self.image = None, None
        self.mpp, self.mpp_source = None, "unknown"
        try:
            import openslide
            if openslide.OpenSlide.detect_format(self.path):
                self.slide = openslide.OpenSlide(self.path)
        except ImportError:
            if Path(path).suffix.lower() in {".svs", ".ndpi", ".mrxs"}:
                raise
        if self.slide is not None:
            self.width, self.height = self.slide.dimensions
            props = self.slide.properties
            raw_mpp = props.get("openslide.mpp-x") or props.get("aperio.MPP")
            if raw_mpp:
                self.mpp, self.mpp_source = float(raw_mpp), "slide_metadata"
            elif props.get("openslide.objective-power"):
                self.mpp = 10.0 / float(props["openslide.objective-power"])
                self.mpp_source = "objective_power_conversion"
        else:
            # Whole-slide JPEGs legitimately exceed Pillow's generic pixel limit.
            Image.MAX_IMAGE_PIXELS = None
            self.image = Image.open(self.path)
            self.width, self.height = self.image.size
        if self.mpp is None and assumed_mpp is not None:
            self.mpp, self.mpp_source = float(assumed_mpp), "assumed"
        if self.mpp is not None and (not math.isfinite(self.mpp) or self.mpp <= 0):
            raise ValueError(f"Invalid MPP for {path}: {self.mpp}")

    @property
    def native_magnification(self):
        return 10.0 / self.mpp if self.mpp is not None else None

    def read(self, region):
        from PIL import Image
        if self.slide is not None:
            # Read native pixels to retain genuine detail for every zoom scale.
            img = self.slide.read_region((region.x, region.y), 0,
                                         (region.width, region.height))
            background = Image.new("RGBA", img.size, "white")
            background.alpha_composite(img)
            img = background.convert("RGB")
        else:
            img = Image.new("RGB", (region.width, region.height), "white")
            x0, y0 = max(0, region.x), max(0, region.y)
            x1 = min(self.width, region.x + region.width)
            y1 = min(self.height, region.y + region.height)
            if x1 > x0 and y1 > y0:
                crop = self.image.crop((x0, y0, x1, y1)).convert("RGB")
                img.paste(crop, (x0 - region.x, y0 - region.y))
        if img.size != (region.output_width, region.output_height):
            img = img.resize((region.output_width, region.output_height), Image.Resampling.LANCZOS)
        return img

    def close(self):
        if self.slide is not None:
            self.slide.close()
        if self.image is not None:
            self.image.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def whole_roi(source):
    return Region("roi", source.path, 0, 0, source.width, source.height,
                  source.width, source.height, "relative", 1.0,
                  magnification_source="native_roi_relative_scale")


def wsi_region(source, x, y, size=4096, magnification=5):
    if source.native_magnification is None:
        raise ValueError(f"WSI needs MPP or a recorded assumption: {source.path}")
    output = max(1, round(size * magnification / source.native_magnification))
    return Region(f"{int(x)}_{int(y)}_m{magnification}", source.path,
                  int(x), int(y), size, size, output, output, "physical",
                  float(magnification), source.mpp, source.mpp_source)


def zoom_regions(parent, target):
    factor_float = float(target) / parent.scale
    factor = round(factor_float)
    if factor <= 1 or not math.isclose(factor, factor_float):
        raise ValueError(f"Zoom must be a greater integral scale: {parent.scale} -> {target}")
    if min(parent.width, parent.height) < factor:
        return []
    regions = []
    for row in range(factor):
        for col in range(factor):
            left, right = col * parent.width // factor, (col + 1) * parent.width // factor
            top, bottom = row * parent.height // factor, (row + 1) * parent.height // factor
            if parent.scale_kind == "relative":
                out_w, out_h = right - left, bottom - top
            else:
                out_w, out_h = parent.output_width, parent.output_height
            regions.append(Region(f"{parent.region_id}/z{target}/{row}_{col}", parent.image_path,
                                  parent.x + left, parent.y + top, right - left, bottom - top,
                                  out_w, out_h, parent.scale_kind, float(target), parent.mpp,
                                  parent.magnification_source, parent.region_id))
    return regions
