"""R³-Ref COCO pool: crops of large single objects from COCO 2017 train (protocols/r3_ref_dataset_v1.md §4).

Index: per image the largest non-crowd instance covering >= 15% of the image, single-word category names only;
key = coco:{image_id}:{ann_id}. Pixels are read from train2017.zip on demand and cropped with a 10% margin.
Usage: python r3_coco.py   (writes /vlm/assets/r3_images/coco2017/crops_index.json)
"""
from __future__ import annotations

import io, json, re, zipfile

ROOT = "/vlm/assets/r3_images/coco2017"
_Z = {}
_A = {}


def _zip():
    if "z" not in _Z:
        _Z["z"] = zipfile.ZipFile(f"{ROOT}/train2017.zip")
    return _Z["z"]


def build_index(min_frac=.15):
    with zipfile.ZipFile(f"{ROOT}/annotations_trainval2017.zip") as za:
        ann = json.load(za.open("annotations/instances_train2017.json"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {im["id"]: im for im in ann["images"]}
    best = {}
    for a in ann["annotations"]:
        if a.get("iscrowd"):
            continue
        im = imgs[a["image_id"]]
        frac = a["area"] / float(im["width"] * im["height"])
        if frac < min_frac or not re.match(r"^[a-z]+$", cats[a["category_id"]]):
            continue
        if a["image_id"] not in best or a["area"] > best[a["image_id"]]["area"]:
            best[a["image_id"]] = a
    index, boxes = [], {}
    for iid, a in sorted(best.items()):
        key = f"coco:{iid}:{a['id']}"
        index.append((key, cats[a["category_id"]]))
        boxes[key] = [imgs[iid]["file_name"]] + [float(x) for x in a["bbox"]]
    json.dump(index, open(f"{ROOT}/crops_index.json", "w"))
    json.dump(boxes, open(f"{ROOT}/crops_boxes.json", "w"))
    print(len(index), "crops,", len(set(c for _, c in index)), "categories")


def coco_crop(key, size=224):
    from PIL import Image
    if "boxes" not in _A:
        _A["boxes"] = json.load(open(f"{ROOT}/crops_boxes.json"))
    fn, x, y, w, h = _A["boxes"][key]
    im = Image.open(io.BytesIO(_zip().read(f"train2017/{fn}"))).convert("RGB")
    mx, my = .1 * w, .1 * h
    box = (max(0, x - mx), max(0, y - my), min(im.width, x + w + mx), min(im.height, y + h + my))
    im = im.crop(tuple(int(v) for v in box))
    s = size / max(im.size)
    return im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))))


if __name__ == "__main__":
    build_index()
