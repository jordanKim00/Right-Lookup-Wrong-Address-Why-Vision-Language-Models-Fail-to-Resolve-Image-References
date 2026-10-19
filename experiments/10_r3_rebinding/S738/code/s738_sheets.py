"""S738 visual check: contact sheets of each flagged calibration image next to its closest evaluation matches."""
import io, json, sys, zipfile
from collections import defaultdict
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
sys.path.insert(0, "/vlm/exp")
from s738_audit import R3I, _img_bytes
import glob, os

def cal_img(key):
    src, base, r = key.split(":"); r = int(r)
    path = f"{R3I}/{src}/data/{base}"
    if not os.path.exists(path):
        path = [h for h in sorted(glob.glob(f"{R3I}/{src}/**/{base}", recursive=True)) if "/.cache/" not in h][0]
    t = pq.read_table(path); first = t.slice(0, 1).to_pylist()[0]
    icol = next(c for c in t.column_names if _img_bytes(first[c]))
    return Image.open(io.BytesIO(_img_bytes(t.column(icol)[r].as_py())[0])).convert("RGB")

def eval_img(sid):
    if "!" in sid:
        z, n = sid.split("!", 1)
        with zipfile.ZipFile(z) as zz:
            return Image.open(io.BytesIO(zz.read(n))).convert("RGB")
    if "#" in sid:
        p, rest = sid.split("#", 1); rg, ri, col, j = rest.split(":")
        row = pq.ParquetFile(p).read_row_group(int(rg)).to_pylist()[int(ri)]
        return Image.open(io.BytesIO(_img_bytes(row[col])[int(j)])).convert("RGB")
    if sid.count(":") == 2 and sid.split(":")[1].endswith(".parquet"):
        return cal_img(sid)
    return Image.open(sid).convert("RGB")

d = json.load(open("/vlm/research/ces/outputs/r3_lure_latest/s738_audit.json"))
g = defaultdict(list)
for f in d["flagged"]:
    g[f["cal"]].append(f)
keys = sorted(g, key=lambda k: min(x["dhash"] * 10 + x["ahash16"] for x in g[k]))
T = 200; rows = []
for k in keys:
    best = sorted(g[k], key=lambda x: (x["dhash"], x["ahash16"]))
    seen, pick = set(), []
    for x in best:                                   # MMIU and MMIU(hf) hold the same files: keep one per basename
        b = x["eval"].split("/")[-1]
        if b in seen: continue
        seen.add(b); pick.append(x)
        if len(pick) == 4: break
    ims = [cal_img(k)] + [eval_img(x["eval"]) for x in pick]
    labs = [k.replace("validation-", "v").replace("test-00000-of-00001.parquet", "test")] + [f"{x['source'][:10]} d{x['dhash']} a{x['ahash16']}" for x in pick]
    rows.append((ims, labs, len(g[k])))
OUT = sys.argv[1]
per = 7
for si in range(0, len(rows), per):
    chunk = rows[si:si + per]
    sheet = Image.new("RGB", (5 * (T + 10), len(chunk) * (T + 28)), "white"); dr = ImageDraw.Draw(sheet)
    for ri, (ims, labs, n) in enumerate(chunk):
        for ci, (im, lab) in enumerate(zip(ims, labs)):
            im = im.copy(); im.thumbnail((T, T))
            x0, y0 = ci * (T + 10), ri * (T + 28)
            sheet.paste(im, (x0, y0 + 14)); dr.text((x0, y0), (lab if ci else f"{lab} n={n}")[:34], fill="black")
    sheet.save(f"{OUT}/sheet_{si // per:02d}.png")
print(len(rows), "rows")
