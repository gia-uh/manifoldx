"""Segment the car from the studio background in a photo, as a boolean mask."""

import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def car_mask(path, width=1400, sat=38, dark=75):
    im = Image.open(path).convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)))
    hsv = np.asarray(im.convert("HSV")).astype(np.float32)
    # The backdrop is grey: the paint is the saturated part, glass and tyres the dark part.
    mask = (hsv[..., 1] > sat) | (hsv[..., 2] < dark)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=6))
    labels, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    return np.asarray(im), mask


if __name__ == "__main__":
    img, mask = car_mask(sys.argv[1])
    out = img.copy()
    out[~mask] = (out[~mask] * 0.25).astype(np.uint8)
    Image.fromarray(out).save(sys.argv[2])
    print(f"{mask.mean():.3f} of pixels are car")
