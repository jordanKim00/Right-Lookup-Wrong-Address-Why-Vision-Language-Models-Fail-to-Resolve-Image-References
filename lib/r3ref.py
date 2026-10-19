"""R³-Ref v1 generator (protocols/r3_ref_dataset_v1.md): image pools, semantic programs, Tier-A realizations, splits.

A program fixes the scene (pool, N images with distinct labels), the references (k, frame) with their lexical family
and syntax template, the format, and the context nuisances. Each scene is emitted as a frame pair (START / END) that
differs only in the frame expression. Labels (the referent j) are used to render answers and to score; training
targets come from interventional discovery (r3_build.py).

Usage: python r3ref.py OUT_DIR   -> OUT_DIR/{split}.jsonl
"""
from __future__ import annotations

import glob, hashlib, io, json, os, pickle, random, re, sys
from collections import defaultdict

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
ROOT = "/vlm/assets/r3_images"

# ---------------- frame lexicon (§3.1) ----------------
FAM = {
    "explicit": {"END": [" from the end", " from the last", " counting from the end", " starting with the last image"],
                 "START": [" from the start", " from the beginning", " counting from the start", " starting with the first image"]},
    # (" from the final image" / " from the first image" dropped before data: "fourth from the final image" can be read as
    #  four steps away from it -- an ambiguous referent)
    "implicit": {"END": [" in reverse order", " in the reverse direction", " when the order is reversed"],
                 "START": [" in order", " in the given order", " in forward order"]},
    "spatial": {"END": [" from the right", " starting from the right"], "START": [" from the left", " starting from the left"]},
}
FRONT = {
    "explicit": {"END": ["Counting from the end", "Starting from the last image", "From the end"],
                 "START": ["Counting from the start", "Starting from the first image", "From the beginning"]},
    "implicit": {"END": ["In reverse order", "Going through the images in reverse"],
                 "START": ["In the given order", "Going through the images in order"]},
    "spatial": {"END": ["Counting from the right", "From the right"], "START": ["Counting from the left", "From the left"]},
}
RESERVED = ["rear", "backward", "backwards", "very end", "far end", "counting back"]          # S601 sealed lexemes
LAYOUT = "The images are arranged from left to right."
CTX = {
    "C1": lambda N, rng: f"Given {N} images.",
    "C2": lambda N, rng: rng.choice(["Consider the first {m} images.", "The first {m} images were taken on the same day.",
                                     "Look at the first {m} images first.", "The first {m} images are photos."]).format(m=rng.randint(2, max(2, min(4, N - 1)))),
    "C3": lambda N, rng: rng.choice(["The last image is brightly lit.", "The final image was added later.",
                                     "The first image is slightly blurry.", "The end of the sequence is not special."]),
    "C4": lambda N, rng: f"Image {rng.randint(1, N)} is a photograph.",
    "C5": lambda N, rng: "Read the question carefully.",
    "C6": lambda N, rng: rng.choice(["The images are shown in sequence.", LAYOUT]),
}
CTX_TEST = {
    "X1": lambda N, rng: f"There are {N - 1 if N > 3 else N + 1} images.",                    # contradicting count
    "X2": None,                                                                                 # negated frame (built in render)
}

# ---------------- image pools (§4) ----------------
SINGLE = re.compile(r"^[a-z]+$")


def _split_of(key, salt="r3"):
    h = int(hashlib.md5(f"{salt}:{key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("dev" if h < 90 else "test")


def load_pools():
    """Pool index without pixels: name -> {'role': 'train'|'test', 'items': {part: [(key, label)]}}."""
    import pyarrow.parquet as pq
    pools = {}
    from r3_proc import proc_label
    proc = {"train": [], "dev": [], "test": []}
    for part, base in (("train", 0), ("dev", 1_000_000), ("test", 2_000_000)):
        for i in range(base, base + 60_000):
            proc[part].append((f"proc:{i}", proc_label(i)[0]))                         # label from the generator's first draw
    pools["proc"] = dict(role="train", items=proc)
    # DomainNet train split (6 styles x 345 classes), single-word classes
    dn = {"train": [], "dev": [], "test": []}
    for f in sorted(glob.glob(f"{ROOT}/domainnet/data/train-*.parquet")):
        t = pq.read_table(f, columns=["label", "domain"])
        names = _classlabel_names(f, "label")
        for i, (lab, dom) in enumerate(zip(t.column("label").to_pylist(), t.column("domain").to_pylist())):
            nm = names[lab]
            if SINGLE.match(nm):
                key = f"domainnet:{os.path.basename(f)}:{i}"
                dn[_split_of(key)].append((key, nm))
    pools["domainnet"] = dict(role="train", items=dn)
    # CIFAR-100 train, single-word fine labels
    c1 = {"train": [], "dev": [], "test": []}
    f = sorted(glob.glob(f"{ROOT}/cifar100/cifar100/train-*.parquet"))[0]
    names = _classlabel_names(f, "fine_label")
    for i, lab in enumerate(pq.read_table(f, columns=["fine_label"]).column("fine_label").to_pylist()):
        nm = names[lab]
        if SINGLE.match(nm):
            key = f"cifar100:{os.path.basename(f)}:{i}"
            c1[_split_of(key)].append((key, nm))
    pools["cifar100"] = dict(role="train", items=c1)
    # COCO 2017 train object crops (if prepared by r3_coco.py)
    cj = f"{ROOT}/coco2017/crops_index.json"
    if os.path.exists(cj):
        co = {"train": [], "dev": [], "test": []}
        for key, nm in json.load(open(cj)):
            co[_split_of(key)].append((key, nm))
        pools["coco"] = dict(role="train", items=co)
    # test-only pools
    cal = []
    f = sorted(glob.glob(f"{ROOT}/caltech256/data/test-*.parquet"))[0]
    for i, nm in enumerate(pq.read_table(f, columns=["text"]).column("text").to_pylist()):
        nm = str(nm).strip().lower()
        if SINGLE.match(nm):
            cal.append((f"caltech256:{os.path.basename(f)}:{i}", nm))
    pools["caltech256"] = dict(role="test", items={"test": cal})
    fo = []
    for f in sorted(glob.glob(f"{ROOT}/food101/data/validation-*.parquet")):
        names = _classlabel_names(f, "label")
        for i, lab in enumerate(pq.read_table(f, columns=["label"]).column("label").to_pylist()):
            if SINGLE.match(names[lab]):
                fo.append((f"food101:{os.path.basename(f)}:{i}", names[lab]))
    pools["food101"] = dict(role="test", items={"test": fo})
    return pools


def _classlabel_names(f, col):
    import pyarrow.parquet as pq
    md = pq.ParquetFile(f).schema_arrow.metadata or {}
    feats = json.loads(md[b"huggingface"]).get("info", {}).get("features", {})
    return feats[col]["names"]


_PQ, _RG = {}, {}


def _pq_cell(path, col, i, keep=64):
    """Row i of one column, reading only its 100-row group (whole-column reads held ~13 GB of DomainNet per process)."""
    import bisect
    import pyarrow.parquet as pq
    if path not in _PQ:
        f = pq.ParquetFile(path)
        starts = [0]
        for g in range(f.num_row_groups):
            starts.append(starts[-1] + f.metadata.row_group(g).num_rows)
        _PQ[path] = (f, starts)
    f, starts = _PQ[path]
    g = bisect.bisect_right(starts, i) - 1
    if (path, g) not in _RG:
        if len(_RG) >= keep:
            _RG.pop(next(iter(_RG)))
        _RG[(path, g)] = f.read_row_group(g, columns=[col]).column(col)
    return _RG[(path, g)][i - starts[g]].as_py()


def load_image(key, size=224):
    """Pixels for a pool key (PIL RGB)."""
    from PIL import Image
    src = key.split(":")
    if src[0] == "proc":
        from r3_proc import proc_image
        return proc_image(int(src[1]))[0]
    if src[0] == "coco":
        from r3_coco import coco_crop
        return coco_crop(key)
    sub = {"domainnet": "domainnet/data", "cifar100": "cifar100/cifar100", "caltech256": "caltech256/data", "food101": "food101/data"}[src[0]]
    col = {"domainnet": "image", "cifar100": "img", "caltech256": "image", "food101": "image"}[src[0]]
    im = Image.open(io.BytesIO(_pq_cell(f"{ROOT}/{sub}/{src[1]}", col, int(src[2]))["bytes"])).convert("RGB")
    s = size / max(im.size)
    return im.resize((max(28, int(im.width * s)), max(28, int(im.height * s)))) if s != 1 else im


def label_of(key, lab):
    if key.startswith("proc:"):
        from r3_proc import proc_image
        return proc_image(int(key.split(":")[1]))[1]
    return lab


# ---------------- rendering ----------------
def _ref_np(k, frame, family, template, phr, compound=False):
    """The referring noun phrase for one source."""
    if compound:
        return {2: rnd_choice(["second-to-last image", "penultimate image"]), 3: "third-to-last image"}[k]
    if template == "T4":
        return f"image that is {ORD[k-1]}{phr}"
    return f"{ORD[k-1]} image{phr if template in ('T1', 'T5') else ''}"


_RNG = random.Random(0)


def rnd_choice(xs):
    return _RNG.choice(xs)


def render(prog, rng):
    """Tier-A text (without image tags) and answer spec for a program."""
    global _RNG
    _RNG = rng
    N, refs, fmt, tpl = prog["N"], prog["refs"], prog["format"], prog["template"]
    ctx = list(prog["context"])
    if prog["family"] == "spatial" and LAYOUT not in ctx:
        ctx = [LAYOUT] + ctx

    def phrase(r):
        if r["family"] == "bare":
            return ""
        return r["phrase"]

    if fmt in ("question", "statement"):
        r = refs[0]
        front = ""
        if tpl == "T2":
            front = r["front"] + ", "
        if tpl == "T3":
            start = "Starting with the last image" if r["frame"] == "END" else "Starting with the first image"
            body = f"{start}, find the {ORD[r['k']-1]} image. What is in it?"
        else:
            np_ = _ref_np(r["k"], r["frame"], r["family"], tpl, phrase(r), r.get("compound", False))
            if fmt == "question":
                body = (f"{front}what is in the {np_}?" if front else f"What is in the {np_}?")
            else:
                body = (f"{front}look at the {np_}. What is in it?" if front else f"Look at the {np_}. What is in it?")
            body = body[0].upper() + body[1:]
        if prog.get("negated"):
            body = "Do not count from the end; count from the start. " + body
        cands = list(prog["labels"]); rng.shuffle(cands)
        text = " ".join(ctx + [body]) + f" Answer with only one word from: {', '.join(cands)}."
        gold = prog["labels"][prog["targets"][0] - 1]
        return text, dict(type="word", gold=gold, cands=cands)
    # option lists: every option is a reference; the question names the gold option's referent label
    gi = prog["gold_option"]
    lines = []
    for i, r in enumerate(refs):
        np_ = _ref_np(r["k"], r["frame"], r["family"], "T1", phrase(r), r.get("compound", False))
        lines.append(f"{chr(65 + i)}: The {np_}")
    q = f"Which image shows a {prog['labels'][prog['targets'][gi] - 1]}?"
    text = " ".join(ctx + [q]) + "\nSelect from the following choices.\n" + "\n".join(lines) + \
        "\nAnswer with the option's letter from the given choices directly."
    return text, dict(type="letter", gold=chr(65 + gi), cands=[chr(65 + i) for i in range(len(refs))])


# ---------------- program sampler ----------------
_BYLAB = {}


def sample_scene(pools, pool, part, N, rng):
    """N distinct labels first, then one item per label (random order)."""
    key = (pool, part)
    if key not in _BYLAB:
        by = defaultdict(list)
        for k, l in pools[pool]["items"][part]:
            by[l].append(k)
        _BYLAB[key] = (sorted(by), by)
    labs_all, by = _BYLAB[key]
    if len(labs_all) < N:
        raise RuntimeError(f"{pool}/{part} has {len(labs_all)} labels < N={N}")
    labs = rng.sample(labs_all, N)
    return [rng.choice(by[l]) for l in labs], labs


def make_pair(pools, rng, split, pool, part, N, family, template, fmt, ctx_types, k_allowed=None, negated=False):
    """Two programs (START, END) on one scene; identical except the frame expression."""
    imgs, labs = sample_scene(pools, pool, part, N, rng)
    ctx = [CTX[c](N, rng) if c in CTX else CTX_TEST[c](N, rng) for c in ctx_types if c != "X2"]
    out = []
    if fmt == "options":
        m = rng.randint(2, min(5, N)); ks = rng.sample(range(1, N + 1), m)
        if k_allowed:
            ks = [k for k in ks if k in k_allowed] or [rng.choice(k_allowed)]
        mixed = rng.random() < .5
        gi = rng.randrange(len(ks))
        base_frames = [rng.choice(["START", "END"]) for _ in ks] if mixed else None
    else:
        ks = [rng.choice(k_allowed) if k_allowed else rng.randint(1, N)]
        gi = 0
    pair_id = f"{split}-{rng.getrandbits(48):012x}"
    for side in ("START", "END"):
        refs = []
        for i, k in enumerate(ks):
            fr = side if fmt != "options" or base_frames is None else (base_frames[i] if side == "START" else
                                                                        ("END" if base_frames[i] == "START" else "START"))
            fam = family
            if fam == "bare":
                fam_r = "bare" if fr == "START" else "explicit"
            else:
                fam_r = fam
            compound = template == "T5" and fr == "END" and k in (2, 3)
            phr = "" if fam_r == "bare" else rng.choice(FAM[fam_r][fr])
            front = "" if fam_r == "bare" else rng.choice(FRONT[fam_r][fr])
            refs.append(dict(k=k, frame=fr, family=fam_r, phrase=phr, front=front, compound=compound))
        targets = [r["k"] if r["frame"] == "START" else N + 1 - r["k"] for r in refs]
        prog = dict(pair=pair_id, pid=f"{pair_id}-{side}", split=split, pool=pool, N=N, images=imgs, labels=labs,
                    format=fmt, template=template, family=family, refs=refs, context=ctx, context_types=list(ctx_types),
                    targets=targets, gold_option=gi, negated=negated)
        text, ans = render(prog, rng)
        prog.update(text=text, answer=ans)
        out.append(prog)
    return out


def leak_ok(text):
    t = text.lower()
    return not any(w in t for w in RESERVED)


def build(out_dir, seed=60200):
    rng = random.Random(seed)
    pools = load_pools()
    train_pools = [p for p in ("proc", "domainnet", "coco", "cifar100") if p in pools]
    pool_w = {"proc": .4, "domainnet": .3, "coco": .2, "cifar100": .1}
    tw = [pool_w[p] for p in train_pools]
    fmts = ["question", "statement", "options"]

    def ctx_draw():
        return [] if rng.random() < .5 else [rng.choice(["C1", "C2", "C3", "C4", "C5", "C6"])]

    def draw(split, part, n_pairs, families, templates, Nr=(3, 8), k_rule=None, pools_override=None, ctx_override=None, negated=False,
             drop_end78=False):
        progs = []
        for _ in range(n_pairs):
            pool = rng.choices(train_pools, tw)[0] if pools_override is None else rng.choice(pools_override)
            N = rng.randint(*Nr)
            fam = rng.choice(families); tpl = rng.choice(templates); fmt = rng.choice(fmts)
            if tpl in ("T2", "T3", "T4") and fmt == "options":
                fmt = rng.choice(["question", "statement"])
            if tpl == "T5":
                fam = "explicit"
            ka = k_rule(N) if k_rule else None
            if tpl == "T5":
                ka = [k for k in (2, 3) if k <= N]
            ct = ctx_override() if ctx_override else ctx_draw()
            pr = make_pair(pools, rng, split, pool, part if pool not in ("caltech256", "food101") else "test", N, fam, tpl, fmt, ct,
                           k_allowed=ka, negated=negated)
            if split != "test_oodnoise" and not all(leak_ok(p["text"]) for p in pr):
                continue
            if negated:                                                                # "Do not count from the end; count from the start" + START body
                pr = [p for p in pr if p["refs"][0]["frame"] == "START"]
            if drop_end78:                                                             # OOD-COMPOSITION: END x {7, 8} only in its test split
                pr = [p for p in pr if not any(r["frame"] == "END" and r["k"] in (7, 8) for r in p["refs"])]
            progs += pr
        return progs

    TRAIN_FAM = ["explicit", "bare"]
    splits = {}
    splits["train"] = draw("train", "train", 2500, TRAIN_FAM, ["T1", "T2", "T4", "T5"], drop_end78=True)
    splits["dev"] = (draw("dev", "dev", 150, TRAIN_FAM, ["T1", "T2", "T4", "T5"], drop_end78=True) +
                     draw("dev", "dev", 100, ["spatial"], ["T1", "T2", "T4"], drop_end78=True))
    splits["test_iid"] = draw("test_iid", "test", 150, TRAIN_FAM, ["T1", "T2", "T4", "T5"], drop_end78=True)
    splits["test_oodlex"] = draw("test_oodlex", "test", 150, ["implicit"], ["T1", "T2", "T4"])
    splits["test_oodtemplate"] = draw("test_oodtemplate", "test", 150, ["explicit"], ["T3"])
    splits["test_oodcomp"] = draw("test_oodcomp", "test", 150, ["explicit"], ["T1", "T2", "T4"], Nr=(7, 8),
                                  k_rule=lambda N: [k for k in (7, 8) if k <= N])
    splits["test_oodlength"] = draw("test_oodlength", "test", 150, TRAIN_FAM, ["T1", "T2", "T4"], Nr=(9, 10))
    splits["test_oodnoise"] = (draw("test_oodnoise", "test", 75, TRAIN_FAM, ["T1"], ctx_override=lambda: ["X1"]) +
                               draw("test_oodnoise", "test", 75, ["explicit"], ["T1"], ctx_override=lambda: [], negated=True))
    splits["test_oodpool"] = draw("test_oodpool", "test", 150, TRAIN_FAM, ["T1", "T2", "T4"], pools_override=["caltech256", "food101"])
    os.makedirs(out_dir, exist_ok=True)
    for nm, ps in splits.items():
        with open(f"{out_dir}/{nm}.jsonl", "w") as fh:
            for p in ps:
                fh.write(json.dumps(p) + "\n")
        print(nm, len(ps), "programs", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, "/vlm/exp")
    build(sys.argv[1] if len(sys.argv) > 1 else "/vlm/assets/r3ref_v1")


def build_cal(out_dir, seed=60600, n_train=600, n_dev=100):
    """R³-Cal (protocol A2): QUESTION format, "the k-th image" vs "the k-th image from the end" frame pairs, N 3..6,
    PROC images only, no context. Targets come from the self-teacher (r3_build --teacher self)."""
    rng = random.Random(seed)
    pools = load_pools()
    os.makedirs(out_dir, exist_ok=True)
    for split, part, n in (("train", "train", n_train), ("dev", "dev", n_dev)):
        progs = []
        for _ in range(n):
            N = rng.randint(3, 6)
            imgs, labs = sample_scene(pools, "proc", part, N, rng)
            k = rng.randint(1, N)
            pair_id = f"cal-{split}-{rng.getrandbits(48):012x}"
            for side in ("START", "END"):
                refs = [dict(k=k, frame=side, family="bare" if side == "START" else "explicit",
                             phrase="" if side == "START" else " from the end", front="", compound=False)]
                prog = dict(pair=pair_id, pid=f"{pair_id}-{side}", split=split, pool="proc", N=N, images=imgs, labels=labs,
                            format="question", template="T1", family="cal", refs=refs, context=[], context_types=[],
                            targets=[k if side == "START" else N + 1 - k], gold_option=0, negated=False)
                text, ans = render(prog, rng)
                prog.update(text=text, answer=ans)
                progs.append(prog)
        with open(f"{out_dir}/{split}.jsonl", "w") as fh:
            for p in progs:
                fh.write(json.dumps(p) + "\n")
        print("cal", split, len(progs), "programs", flush=True)


ROLE_LOC = {"END": ["lower right corner", "right edge", "bottom part", "back part"],
            "START": ["upper left corner", "left edge", "top part", "front part"]}
ROLE_NP = {"END": ["latter", "later", "final"], "START": ["former", "earlier", "initial"]}


def build_cal_a3v2(out_dir, src_dir="/vlm/assets/r3cal_a3", seed=60950):
    """S606 V2: the A3.1 calibration programs plus one role-contrast program per reference -- the same scene and reference
    with a position word in the same clause that is NOT the counting origin ("with the {loc} slightly blurred" /
    "rather than in the {w} images"); the gate label stays the reference's own frame. No reverse target anywhere."""
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "dev"):
        progs = [json.loads(l) for l in open(f"{src_dir}/{split}.jsonl")]
        out = list(progs)
        for p in progs:
            kind = rng.choice(["loc", "np"]); side = rng.choice(["END", "START"])
            mod = (f" with the {rng.choice(ROLE_LOC[side])} slightly blurred" if kind == "loc"
                   else f" rather than in the {rng.choice(ROLE_NP[side])} images")
            q = dict(p)
            head, sep, tail = p["text"].partition("? Answer with only one word from:")
            assert sep, p["text"]
            q["text"] = f"{head}{mod}{sep}{tail}"
            q["pid"] = p["pid"] + "-role"; q["family"] = "cal_a3_role"; q["role_mod"] = dict(kind=kind, side=side, text=mod)
            assert leak_ok(q["text"])
            out.append(q)
        with open(f"{out_dir}/{split}.jsonl", "w") as fh:
            for p in out:
                fh.write(json.dumps(p) + "\n")
        print("cal_a3v2", split, len(out), "programs", flush=True)


def build_cal_a3(out_dir, seed=60700, n_train=600, n_dev=100):
    """R³-Cal-A3 (protocol A3): per scene three QUESTION programs -- "the k-th image" (START), "the k-th image from the
    start" (START), "the k-th image from the end" (END); N 3..6; PROC images. Smaller training sizes are prefixes."""
    rng = random.Random(seed)
    pools = load_pools()
    os.makedirs(out_dir, exist_ok=True)
    for split, part, n in (("train", "train", n_train), ("dev", "dev", n_dev)):
        progs = []
        for si in range(n):
            N = rng.randint(3, 6)
            imgs, labs = sample_scene(pools, "proc", part, N, rng)
            k = rng.randint(1, N)
            sid = f"a3-{split}-{si:04d}"
            for side, fam, phr in (("START", "bare", ""), ("START", "explicit", " from the start"), ("END", "explicit", " from the end")):
                refs = [dict(k=k, frame=side, family=fam, phrase=phr, front="", compound=False)]
                prog = dict(pair=sid, pid=f"{sid}-{fam}-{side}", split=split, scene=si, pool="proc", N=N, images=imgs, labels=labs,
                            format="question", template="T1", family="cal_a3", refs=refs, context=[], context_types=[],
                            targets=[k if side == "START" else N + 1 - k], gold_option=0, negated=False)
                text, ans = render(prog, rng)
                prog.update(text=text, answer=ans)
                progs.append(prog)
        with open(f"{out_dir}/{split}.jsonl", "w") as fh:
            for p in progs:
                fh.write(json.dumps(p) + "\n")
        print("cal_a3", split, len(progs), "programs", flush=True)
