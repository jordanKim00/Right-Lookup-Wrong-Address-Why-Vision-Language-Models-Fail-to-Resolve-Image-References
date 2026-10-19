"""S730 discovery diagnostic (post hoc, no verdict; 2026-10-11): why does the dense selection rule find no index-matching head in a model?
The S730 discovery scenes (seed 73001, 15 scenes, START t = 1..7, the S730 question form and menu, same pixels), all heads of every
full-attention layer, answer row and the line before it: mean START diagonal share per head, START accuracy, block token counts.
Usage: python s730d_diag.py MODEL TAG [--pixels P]"""
import argparse, json, random, sys
import numpy as np, torch
sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half
N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    ST = {}

    def hook(li):
        def h(mod, args, kwargs, out):
            if not ST.get("on"):
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos.float(), sin.float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rows = ST["rows"]
            qp = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
            gated = qp.shape[-1] == 2 * nh * hd
            q = qp.view(len(rows), nh, 2 * hd)[..., :hd] if gated else qp.view(len(rows), nh, hd)
            qn, kn = getattr(mod, "q_norm", None), getattr(mod, "k_norm", None)
            q = qn(q.to(qn.weight.dtype)).float() if qn is not None else q
            kp = torch.nn.functional.linear(hs, mod.k_proj.weight.float(), None if mod.k_proj.bias is None else mod.k_proj.bias.float())
            k = kp.view(S, -1, hd); k = kn(k.to(kn.weight.dtype)).float() if kn is not None else k
            M_ = sys.modules[mod.__class__.__module__]
            if cos.dim() == 4 and hasattr(M_, "apply_multimodal_rotary_pos_emb"):
                sec = (getattr(mod.config, "rope_parameters", None) or mod.config.rope_scaling)["mrope_section"]
                qe = M_.apply_multimodal_rotary_pos_emb(q.transpose(0, 1)[None], q.transpose(0, 1)[None], cos[:, :, rows], sin[:, :, rows], sec)[0][0].transpose(0, 1)
                ke = M_.apply_multimodal_rotary_pos_emb(k.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin, sec)[1][0].transpose(0, 1)
            else:
                cos, sin = cos[0], sin[0]; rd = cos.shape[-1]
                qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
                ke = torch.cat([k[..., :rd] * cos[:, None] + rotate_half(k[..., :rd]) * sin[:, None], k[..., rd:]], -1)
            heads = torch.arange(nh, device=hs.device); kh = heads // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe[:, heads], ke[:, kh]) * mod.scaling
            sc = torch.stack([torch.logsumexp(lg[:, :, b0:b1 + 1], -1) for b0, b1 in ST["blocks"]], -1)
            ST["cap"][li] = torch.softmax(sc, -1).cpu()                                        # [rows, heads, N]
            return out
        return h
    hooks = [v.layers[li].self_attn.register_forward_hook(hook(li), with_kwargs=True) for li in FA_L]

    def prompt(t, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[t-1]} image from the first? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)
    drng = random.Random(73001); diag = {}; acc = []; bt = []
    for _ in range(15):
        labs = drng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(drng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; drng.shuffle(menu)
        for t in range(1, N + 1):
            inp = v.encode(prompt(t, menu), pics); ids = inp["input_ids"][0].tolist()
            B = v.image_blocks(inp["input_ids"], len(pics)); bt.append([len(b) for b in B])
            ST.clear(); ST.update(on=True, rows=[len(ids) - 2, len(ids) - 1], blocks=[(int(b[0]), int(b[-1])) for b in B], cap={})
            with torch.no_grad():
                lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
            cap = ST["cap"]; ST.clear()
            lab = menu[int(torch.stack([lg[ids1(c)[0]] for c in menu]).argmax())]; acc.append(lab == labs[t - 1])
            for li, sh in cap.items():
                diag.setdefault(li, []).append(sh[:, :, t - 1])                               # [rows, heads]
    for h in hooks:
        h.remove()
    res = []
    for li, xs in diag.items():
        m = torch.stack(xs).mean(0)                                                             # [rows, heads]
        for r in range(m.shape[0]):
            for hh in range(m.shape[1]):
                res.append((float(m[r, hh]), int(li), int(hh), ["prev", "ans"][r]))
    res.sort(reverse=True)
    print(f"[{a.tag}] START accuracy {np.mean(acc):.3f} (n={len(acc)}) · block tokens {sorted(set(map(tuple, bt)))[:3]} · FA layers {len(FA_L)}")
    print(f"[{a.tag}] heads with START diagonal share >= .5: {sum(1 for x in res if x[0] >= .5)} · >= .3: {sum(1 for x in res if x[0] >= .3)} · uniform .143")
    print(f"[{a.tag}] top 15: " + " · ".join(f"L{l}h{hh}@{rw} {s:.2f}" for s, l, hh, rw in res[:15]))
    json.dump(dict(tag=a.tag, model=a.model, start_acc=float(np.mean(acc)), top=res[:200]), open(f"/vlm/results/r3/s730d_{a.tag}_diag.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s730d_{a.tag}_diag.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--pixels", type=int, default=224)
    main(ap.parse_args())
