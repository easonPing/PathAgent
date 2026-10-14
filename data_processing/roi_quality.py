"""Conservative, reproducible rejection of definite blank/padded model inputs.

Sparse or uncertain tissue and partial-boundary crops remain eligible. The
checker intentionally does not infer tissue from low coverage alone.
"""
import numpy as np


QUALITY_VERSION = 'roi-validity-v2-1'
QUALITY_POLICY = {
    'colored_channel_difference_min': 15,
    'colored_max_channel_gt': 20,
    'colored_min_channel_lt': 245,
    'neutral_black_max': 12,
    'neutral_white_min': 243,
    'tile_size': 16,
    'flat_max_channel_std_lt': 3,
    'blank_fraction_min': 0.99,
}


def _flat_fraction(rgb, tile_size=16):
    """Pixel-weighted flatness, including partial edge tiles."""
    height, width = rgb.shape[:2]
    flat_pixels = 0
    rows, columns = height // tile_size, width // tile_size
    full_height, full_width = rows * tile_size, columns * tile_size
    if rows and columns:
        tiles = rgb[:full_height, :full_width].reshape(
            rows, tile_size, columns, tile_size, 3)
        flat_tiles = tiles.std(axis=(1, 3)).max(axis=2) < 3
        flat_pixels = int(flat_tiles.sum()) * tile_size * tile_size
    # Only the bottom/right edge needs individual partial-tile accounting.
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            if y < full_height and x < full_width:
                continue
            tile = rgb[y:y + tile_size, x:x + tile_size]
            if float(tile.std(axis=(0, 1)).max()) < 3:
                flat_pixels += tile.shape[0] * tile.shape[1]
    return flat_pixels / float(height * width)


def quality_for_region(source, region):
    """Check original bounds and the exact resized RGB supplied to Perceptor.

    Fully outside regions are rejected without calling source.read(). None
    marks image metrics that cannot be observed in that case.
    """
    area = int(region.width) * int(region.height)
    if area <= 0:
        raise ValueError('Region dimensions must be positive')
    x0, y0 = max(0, region.x), max(0, region.y)
    x1 = min(source.width, region.x + region.width)
    y1 = min(source.height, region.y + region.height)
    valid_area = max(0, x1 - x0) * max(0, y1 - y0)
    valid_fraction = valid_area / float(area)
    result = {
        'region_id': region.region_id,
        'quality_version': QUALITY_VERSION,
        'valid_pixel_fraction': valid_fraction,
        'padding_fraction': 1.0 - valid_fraction,
        'colored_pixel_fraction': None,
        'neutral_fraction': None,
        'flat_fraction': None,
        'hard_reject': valid_area == 0,
        'reason': 'fully_outside' if valid_area == 0 else 'kept',
    }
    if valid_area == 0:
        return result
    image = source.read(region)
    # ImageSource returns the actual resized model image; never inspect a
    # thumbnail or substitute a tissue-mask image here.
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('Model input must be a nonempty RGB image')
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    colored = ((maximum.astype(np.int16) - minimum.astype(np.int16) >= 15)
               & (maximum > 20) & (minimum < 245))
    neutral = (maximum <= 12) | (minimum >= 243)
    colored_fraction = float(colored.mean())
    neutral_fraction = float(neutral.mean())
    flat_fraction = _flat_fraction(rgb)
    reject = colored_fraction == 0 and (neutral_fraction >= 0.99 or flat_fraction >= 0.99)
    result.update({'colored_pixel_fraction': colored_fraction,
                   'neutral_fraction': neutral_fraction,
                   'flat_fraction': flat_fraction,
                   'hard_reject': reject,
                   'reason': 'definite_blank' if reject else 'kept'})
    return result
