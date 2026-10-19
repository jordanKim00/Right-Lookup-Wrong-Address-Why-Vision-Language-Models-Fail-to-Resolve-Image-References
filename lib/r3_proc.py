"""R³-Ref PROC pool: procedurally generated training images (protocols/r3_ref_dataset_v1.md §4).

Every image is a deterministic function of its integer id: one large main shape (the label) with a colour and a fill
texture on a textured background, plus 0-2 small distractor shapes, random size / rotation / position, optional blur
and noise. No arrows, letters or digits (they would carry direction or digit references).
"""
from __future__ import annotations

import math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SHAPES = ["circle", "square", "triangle", "star", "hexagon", "pentagon", "diamond", "cross", "ring", "ellipse",
          "crescent", "heart"]
COLORS = {"red": (220, 40, 40), "orange": (245, 140, 30), "yellow": (240, 220, 40), "green": (50, 170, 70),
          "blue": (40, 90, 220), "purple": (130, 60, 190), "pink": (240, 120, 180), "brown": (130, 80, 40),
          "black": (25, 25, 25), "white": (245, 245, 245), "gray": (128, 128, 128), "cyan": (40, 200, 210)}
FILLS = ["solid", "striped", "dotted", "checkered", "gradient"]
BACKS = ["solid", "gradient", "noise", "stripes", "checker", "dots"]


def _poly(cx, cy, r, n, rot):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]


def _mask(shape, S, cx, cy, r, rot):
    m = Image.new("L", (S, S), 0); d = ImageDraw.Draw(m)
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif shape == "ellipse":
        d.ellipse([cx - r, cy - .6 * r, cx + r, cy + .6 * r], fill=255); m = m.rotate(math.degrees(rot), center=(cx, cy))
    elif shape == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255); d.ellipse([cx - .55 * r, cy - .55 * r, cx + .55 * r, cy + .55 * r], fill=0)
    elif shape == "square":                                                      # small tilt only (a 45-degree square reads as a diamond)
        d.polygon(_poly(cx, cy, r * 1.15, 4, math.pi / 4 + (rot % 1.0 - .5) * .5), fill=255)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + .65 * r, cy), (cx, cy + r), (cx - .65 * r, cy)], fill=255); m = m.rotate(math.degrees(rot) % 30, center=(cx, cy))
    elif shape == "triangle":
        d.polygon(_poly(cx, cy, r, 3, rot - math.pi / 2), fill=255)
    elif shape == "pentagon":
        d.polygon(_poly(cx, cy, r, 5, rot - math.pi / 2), fill=255)
    elif shape == "hexagon":
        d.polygon(_poly(cx, cy, r, 6, rot), fill=255)
    elif shape == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else .45 * r
            a = rot - math.pi / 2 + math.pi * i / 5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=255)
    elif shape == "cross":
        w = .32 * r
        d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=255); d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=255)
        m = m.rotate(math.degrees(rot) % 45, center=(cx, cy))
    elif shape == "crescent":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        o = .45 * r
        d.ellipse([cx - r + o, cy - r - .1 * r, cx + r + o, cy + r - .1 * r], fill=0); m = m.rotate(math.degrees(rot), center=(cx, cy))
    elif shape == "heart":
        d.ellipse([cx - r, cy - .75 * r, cx, cy + .2 * r], fill=255); d.ellipse([cx, cy - .75 * r, cx + r, cy + .2 * r], fill=255)
        d.polygon([(cx - .97 * r, cy - .1 * r), (cx + .97 * r, cy - .1 * r), (cx, cy + r)], fill=255)
    return m


def _texture(kind, S, col, rng):
    a = np.zeros((S, S, 3), np.float32) + np.array(col, np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    if kind == "striped":
        p = rng.randint(6, 14); ang = rng.random() * math.pi
        band = ((xx * math.cos(ang) + yy * math.sin(ang)) // p) % 2 == 0
        a[band] = a[band] * .45 + 255 * .55
    elif kind == "dotted":
        p = rng.randint(8, 16)
        dots = ((xx % p - p / 2) ** 2 + (yy % p - p / 2) ** 2) < (p / 4) ** 2
        a[dots] = a[dots] * .35
    elif kind == "checkered":
        p = rng.randint(8, 18)
        ch = ((xx // p + yy // p) % 2) == 0
        a[ch] = a[ch] * .55
    elif kind == "gradient":
        g = (xx / S)[..., None] if rng.random() < .5 else (yy / S)[..., None]
        a = a * (.45 + .55 * g)
    return a


def _background(kind, S, rng, avoid):
    names = [c for c in COLORS if c != avoid]
    c1 = np.array(COLORS[rng.choice(names)], np.float32) * .5 + 127
    c2 = np.array(COLORS[rng.choice(names)], np.float32) * .5 + 127
    yy, xx = np.mgrid[0:S, 0:S]
    if kind == "solid":
        a = np.zeros((S, S, 3), np.float32) + c1
    elif kind == "gradient":
        t = (xx / S)[..., None]; a = c1 * (1 - t) + c2 * t
    elif kind == "noise":
        nrs = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
        a = np.zeros((S, S, 3), np.float32) + c1 + nrs.normal(0, 25, (S, S, 3))
    elif kind == "stripes":
        p = rng.randint(10, 30); band = ((yy // p) % 2 == 0)[..., None]
        a = np.where(band, c1, c2).astype(np.float32)
    elif kind == "checker":
        p = rng.randint(14, 40); ch = (((xx // p) + (yy // p)) % 2 == 0)[..., None]
        a = np.where(ch, c1, c2).astype(np.float32)
    else:
        p = rng.randint(14, 28)
        dots = (((xx % p - p / 2) ** 2 + (yy % p - p / 2) ** 2) < (p / 5) ** 2)[..., None]
        a = np.where(dots, c2, c1).astype(np.float32)
    return a


def proc_label(pid: int):
    """(shape, colour) of proc_image(pid) without rendering (the first two draws of its generator)."""
    rng = random.Random(1_000_003 * pid + 17)
    shape = SHAPES[rng.randrange(len(SHAPES))]
    return shape, rng.choice(list(COLORS))


def proc_image(pid: int, S: int = 224):
    """Deterministic image for id pid -> (PIL image, shape word, colour word)."""
    rng = random.Random(1_000_003 * pid + 17)
    shape = SHAPES[rng.randrange(len(SHAPES))]
    color = rng.choice(list(COLORS))
    canvas = _background(rng.choice(BACKS), S, rng, color)
    for _ in range(rng.randint(0, 2)):                                           # small distractors (other shapes)
        ds = rng.choice([s_ for s_ in SHAPES if s_ != shape]); dc = rng.choice([c for c in COLORS if c != color])
        r = rng.uniform(.06, .11) * S
        cx = rng.choice([rng.uniform(.12, .25), rng.uniform(.75, .88)]) * S; cy = rng.uniform(.12, .88) * S
        mk = np.asarray(_mask(ds, S, cx, cy, r, rng.random() * math.pi), np.float32)[..., None] / 255
        canvas = canvas * (1 - mk) + np.array(COLORS[dc], np.float32) * mk
    r = rng.uniform(.26, .38) * S
    cx = rng.uniform(.42, .58) * S; cy = rng.uniform(.42, .58) * S
    mk = np.asarray(_mask(shape, S, cx, cy, r, rng.random() * 2 * math.pi), np.float32)[..., None] / 255
    canvas = canvas * (1 - mk) + _texture(rng.choice(FILLS), S, COLORS[color], rng) * mk
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    if rng.random() < .3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(.5, 1.5)))
    if rng.random() < .3:
        nrs = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
        img = Image.fromarray(np.clip(np.asarray(img, np.float32) + nrs.normal(0, 8, (S, S, 3)), 0, 255).astype(np.uint8))
    return img, shape, color


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/r3_proc_grid.png"
    tiles = [proc_image(i)[0].resize((112, 112)) for i in range(48)]
    grid = Image.new("RGB", (112 * 8, 112 * 6))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 8) * 112, (i // 8) * 112))
    grid.save(out)
    print(out, [proc_image(i)[1:] for i in range(8)])
