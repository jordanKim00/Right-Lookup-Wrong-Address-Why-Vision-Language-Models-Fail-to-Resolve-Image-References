"""S747a: the behavioural phenomenon in VisualSwap's model set without R³ (pre-registration in knowledge/DECISIONS.md before any S747
data; analysis exp/s747_analyze.py).  The paper's evaluator (exp/s582_evalrepair.run) on the later MMIU split (500 items) and MMIU
visual cloze (200), with a stand-in for R³ that returns the unedited answer, so "F|base" / "R|base" are the start- and end-relative
answers of the unedited model and "|rep" repeats them.  Writes results/s582_evalrepair_{TAG}.json in the evaluator's format.
Usage: python s747_beh.py MODEL TAG [--device_map balanced]
"""
import argparse, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s582_evalrepair as E


class NoRep:
    def __init__(self, v):
        self.v = v

    @torch.no_grad()
    def both(self, text, pics, cand):
        inp = self.v.encode(text, pics)
        lg = self.v.model(**inp, use_cache=False, return_dict=True, **self.v.forward_extra).logits[0, -1].float()
        a = int(torch.stack([lg[c] for c in cand]).argmax())
        return a, a


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    E.run(v, NoRep(v), a.tag, ["mmiu_pos"], fresh=True)
    E.run(v, NoRep(v), a.tag + "_cz", ["cloze"])
    print(f"[{a.tag}] 끝 -> /vlm/results/s582_evalrepair_{a.tag}.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
