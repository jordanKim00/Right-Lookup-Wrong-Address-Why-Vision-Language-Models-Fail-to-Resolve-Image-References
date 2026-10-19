"""S738 (1): is the R³ calibration set unrelated to every evaluation set?  Perceptual-hash audit, no model call.
Calibration images = the natural photographs used by S724 / S729 (exp/s724_readcal.nat_keys(Random(72400), 12000): caltech256 and food101
rows, R³-Ref images excluded).  Procedural calibration images are rendered shapes and are reported but not matched.
Evaluation images = every image file, parquet image cell or zip member under the evaluation roots (MMIU all tasks incl. visual cloze,
MuirBench, BLINK / Q-Bench2 / Mantis / MMMU / MIBench / MIRB (assets/r3_bench), MathVista, imagenette, CIFAR-10 test batch of the synthetic
evaluation, R³-Ref images).
Hashes: dHash 64 bit (9x8 gray) for candidate search (five bands, so every pair within Hamming 4 shares a band) and a 16x16 mean hash for
confirmation.  Flat images (16x16 gray std < 3) are counted and left out of matching.  A pair is flagged when the dHash distance <= 4 and the
16x16 hash distance <= 24 of 256; flagged pairs are written with their sources for visual checking.
Usage (container): python /vlm/exp/s738_audit.py --out /vlm/research/ces/outputs/r3_lure_latest/s738_audit.json --workers 24
"""
from __future__ import annotations

import argparse, glob, io, json, os, pickle, random, re, sys, zipfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
R3I = "/vlm/assets/r3_images"


def hashes(img):
    g = img.convert("L")
    a9 = np.asarray(g.resize((9, 8), Image.BILINEAR), np.int16)
    d = (a9[:, 1:] > a9[:, :-1]).flatten()
    a16 = np.asarray(g.resize((16, 16), Image.BILINEAR), np.float32)
    m = (a16 > a16.mean()).flatten()
    return int(np.packbits(d).view(">u8")[0]), np.packbits(m).tobytes().hex(), float(a16.std())


def h_bytes(b):
    try:
        with Image.open(io.BytesIO(b)) as im:
            return hashes(im)
    except Exception:
        return None


def walk_files(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.join(d, f))
    return out


def job_file(path):
    """-> list of (source id, hashes)"""
    res = []
    low = path.lower()
    try:
        if low.endswith(IMG_EXT):
            with Image.open(path) as im:
                res.append((path, hashes(im)))
        elif low.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg).to_pylist()
                for ri, row in enumerate(t):
                    for col, val in row.items():
                        for j, b in enumerate(_img_bytes(val)):
                            h = h_bytes(b)
                            if h:
                                res.append((f"{path}#{rg}:{ri}:{col}:{j}", h))
        elif low.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():
                    if n.lower().endswith(IMG_EXT):
                        h = h_bytes(z.read(n))
                        if h:
                            res.append((f"{path}!{n}", h))
    except Exception as e:
        res.append((f"ERROR {path} {e!r}", None))
    return res


def _img_bytes(val):
    if isinstance(val, (bytes, bytearray)) and len(val) > 64:
        return [bytes(val)]
    if isinstance(val, dict) and isinstance(val.get("bytes"), (bytes, bytearray)):
        return [bytes(val["bytes"])]
    if isinstance(val, list):
        out = []
        for x in val:
            out += _img_bytes(x)
        return out
    return []


def r3_keys_job(arg):
    """arg = (src, base, [rows]) -> hashes of those rows of an r3_images parquet (first image-like column)."""
    src, base, rows = arg
    import pyarrow.parquet as pq
    path = f"{R3I}/{src}/data/{base}"
    if not os.path.exists(path):                                                 # some sources keep their parquet outside data/ (cifar100/cifar100/)
        hits = sorted(glob.glob(f"{R3I}/{src}/**/{base}", recursive=True))
        hits = [h for h in hits if "/.cache/" not in h]
        if not hits:
            return [(f"{src}:{base}:{r}", None) for r in sorted(set(rows))]
        path = hits[0]
    t = pq.read_table(path)
    cols = t.column_names
    want = set(rows); res = []
    # locate the image column on the first row
    first = t.slice(0, 1).to_pylist()[0]
    icol = next(c for c in cols if _img_bytes(first[c]))
    col = t.column(icol)
    for r in sorted(want):
        b = _img_bytes(col[r].as_py())
        h = h_bytes(b[0]) if b else None
        res.append((f"{src}:{base}:{r}", h))
    return res


def nat_keys(rng, n):
    used = set()
    for f in glob.glob("/vlm/assets/r3ref_v1/*.jsonl"):
        for l in open(f):
            used.update(json.loads(l).get("images", []))
    keys = []
    import pyarrow.parquet as pq
    for src, pat in (("caltech256", f"{R3I}/caltech256/data/*.parquet"), ("food101", f"{R3I}/food101/data/*.parquet")):
        for f in sorted(glob.glob(pat)):
            rows = pq.ParquetFile(f).metadata.num_rows; base = f.split("/")[-1]
            keys += [k for k in (f"{src}:{base}:{i}" for i in range(rows)) if k not in used]
    rng.shuffle(keys)
    return keys[:n]


def group_keys(keys):
    """parquet-row keys src:base.parquet:row only (procedural 'proc:pid' and 'coco:id:ann' keys are not r3_images rows)."""
    g = defaultdict(list)
    for k in keys:
        p = k.split(":")
        if len(p) == 3 and p[1].endswith(".parquet"):
            g[(p[0], p[1])].append(int(p[2]))
    return [(s, b, rs) for (s, b), rs in g.items()]


def main(a):
    pool = Pool(a.workers)
    # calibration
    cal_keys = nat_keys(random.Random(72400), 12000)
    if a.smoke:
        cal_keys = cal_keys[:300]
    cal = [x for part in pool.imap_unordered(r3_keys_job, group_keys(cal_keys)) for x in part]
    # evaluation
    roots = {"MMIU": ["/vlm/assets/mmiu"], "r3_bench(BLINK,Q-Bench2,Mantis,MMMU,MIBench,MIRB)": ["/vlm/assets/r3_bench"],
             "MuirBench": ["/vlm/assets/hf/hub/datasets--MUIRBENCH--MUIRBENCH"], "MathVista": ["/vlm/assets/hf/hub/datasets--AI4Math--MathVista"],
             "MMIU(hf)": ["/vlm/assets/hf/hub/datasets--FanqingM--MMIU-Benchmark"], "imagenette": ["/vlm/assets/imagenette2-160"]}
    ev = {}
    CD = os.path.join(os.path.dirname(a.out), "s738_audit_cache")
    os.makedirs(CD, exist_ok=True)
    def cached(name):
        f = os.path.join(CD, re.sub(r"[^A-Za-z0-9]+", "_", name) + ".json")
        return f, (json.load(open(f)) if os.path.exists(f) and not a.smoke else None)
    for name, rs in roots.items():
        cf, got = cached(name)
        if got is not None:
            ev[name] = [(sid, tuple(h) if h else None) for sid, h in got]; print(f"{name}: cached {len(ev[name])}", flush=True); continue
        files = [f for r in rs if os.path.exists(r) for f in walk_files(r) if f.lower().endswith(IMG_EXT + (".parquet", ".zip"))]
        uniq = {}
        for f in files:                                                          # HF snapshots are symlinks to extension-less blobs:
            uniq.setdefault(os.path.realpath(f), f)                               # dedupe by target, keep the path that carries the extension
        files = sorted(uniq.values())
        if a.smoke:
            files = files[:20]
        ev[name] = [x for part in pool.imap_unordered(job_file, files, chunksize=8) for x in part]
        print(f"{name}: {len(files)} files -> {len(ev[name])} images", flush=True)
        if not a.smoke:
            json.dump([(sid, list(h) if h else None) for sid, h in ev[name]], open(cf, "w"))
    # CIFAR-10 test batch (synthetic evaluation)
    d = pickle.load(open("/vlm/assets/cifar10/test_batch", "rb"), encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    ev["CIFAR-10 test"] = [(f"cifar10:test:{i}", hashes(Image.fromarray(x))) for i, x in enumerate(X)]
    # R³-Ref images
    rk = set()
    for f in glob.glob("/vlm/assets/r3ref_v1/*.jsonl"):
        for l in open(f):
            rk.update(json.loads(l).get("images", []))
    rk = sorted(rk)[:300] if a.smoke else sorted(rk)
    ev["R3-Ref"] = [x for part in pool.imap_unordered(r3_keys_job, group_keys(rk)) for x in part]
    other_keys = defaultdict(int)
    for k in rk:
        p = k.split(":")
        if not (len(p) == 3 and p[1].endswith(".parquet")):
            other_keys[p[0]] += 1
    pool.close()
    # matching
    def bands(h):
        return [(i, (h >> s) & ((1 << w) - 1)) for i, (s, w) in enumerate(((0, 13), (13, 13), (26, 13), (39, 13), (52, 12)))]
    idx = defaultdict(list); flat_ev = 0; n_ev = 0; errors = []
    for name, items in ev.items():
        for sid, h in items:
            if h is None:
                errors.append(sid); continue
            n_ev += 1
            if h[2] < 3:
                flat_ev += 1; continue
            for b in bands(h[0]):
                idx[b].append((name, sid, h))
    flagged = []; flat_cal = 0; n_cal = 0
    for sid, h in cal:
        if h is None:
            errors.append(sid); continue
        n_cal += 1
        if h[2] < 3:
            flat_cal += 1; continue
        seen = set()
        for b in bands(h[0]):
            for name, esid, eh in idx.get(b, []):
                if esid in seen:
                    continue
                seen.add(esid)
                dd = bin(h[0] ^ eh[0]).count("1")
                if dd <= 4:
                    da = bin(int(h[1], 16) ^ int(eh[1], 16)).count("1")
                    if da <= 24:
                        flagged.append(dict(cal=sid, eval=esid, source=name, dhash=dd, ahash16=da))
    out = dict(calibration=dict(natural=n_cal, flat=flat_cal, procedural="rendered shapes (not matched)"), r3ref_unmatched_keys=dict(other_keys),
               evaluation={k: len(v) for k, v in ev.items()}, evaluation_total=n_ev, evaluation_flat=flat_ev,
               flagged=flagged, n_flagged=len(flagged), decode_errors=errors[:50], n_errors=len(errors))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("flagged", "decode_errors")}, indent=1))
    print("flagged pairs:", len(flagged))
    for f in flagged[:20]:
        print(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args())
