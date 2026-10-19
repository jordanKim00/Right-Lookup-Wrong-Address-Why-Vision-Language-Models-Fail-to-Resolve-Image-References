"""Inner part of exp/s747_legacy.py: the S747a behavioural evaluation (exp/s747_beh.py) through exp/vlm_adapters.Adapter."""
import sys, json, torch
sys.path.insert(0, "/vlm/exp")
import transformers
from vlm_adapters import Adapter
import s582_evalrepair as E
from s747_beh import NoRep

model, tag = sys.argv[1], sys.argv[2]
print(f"[{tag}] transformers {transformers.__version__}", flush=True)
v = Adapter(model)
if "--smoke" in sys.argv:
    E.run(v, NoRep(v), tag, ["cloze"], cap=2)
else:
    E.run(v, NoRep(v), tag, ["mmiu_pos"], fresh=True)
    E.run(v, NoRep(v), tag + "_cz", ["cloze"])
print(f"[{tag}] 끝 -> /vlm/results/s582_evalrepair_{tag}.json", flush=True)
