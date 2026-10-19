"""Resident 27B worker (10-16 20:2x; efficiency only, no scientific content).  Loads Qwen3.5-27B once (balanced over the visible cards) and runs
experiment scripts one after another without reloading: each job is a JSON file in /vlm/exp/worker_jobs/ ({"script": path, "argv": [...],
"log": path}); the script is executed with runpy as __main__, its 'Uni(...)' returns the resident instance (same model path required), every
forward hook / pre-hook a job left on the model is removed after it (hooks present right after loading are kept), and the job file is renamed .done or .failed.
The scripts are unchanged: same model object, same forward code, same batching, so their numbers are those of a fresh process (checked by the
first job's NOOP / exactness arms as usual).
Usage (container): python -u /vlm/exp/w27b_worker.py /models/Qwen3.5-27B"""
from __future__ import annotations

import contextlib, glob, json, os, runpy, sys, time, traceback

sys.path.insert(0, "/vlm/exp")
import torch
import uni

JOBS = "/vlm/exp/worker_jobs"


def hook_snapshot(model):
    return {id(m): (set(m._forward_hooks), set(m._forward_pre_hooks)) for m in model.modules()}


def clear_new_hooks(model, base):
    """Remove only the hooks a job added (those present right after loading stay, so the forward is that of a fresh process)."""
    n = 0
    for m in model.modules():
        f0, p0 = base.get(id(m), (set(), set()))
        for d, keep in ((m._forward_hooks, f0), (m._forward_pre_hooks, p0)):
            for key in [x for x in d if x not in keep]:
                d.pop(key, None); n += 1
                for extra in ("_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks_with_kwargs"):
                    if hasattr(m, extra):
                        getattr(m, extra).pop(key, None)
    return n


def main(path):
    os.makedirs(JOBS, exist_ok=True)
    t0 = time.time()
    REAL = uni.Uni
    pre = REAL(path, device="cuda:0", attn="sdpa", device_map="balanced")
    base = hook_snapshot(pre.model)
    print(f"[w27b] loaded {path} in {time.time() - t0:.0f} s (hooks at load {sum(len(a) + len(b) for a, b in base.values())})", flush=True)

    def factory(model, *a, **k):
        assert os.path.realpath(model.rstrip("/")) == os.path.realpath(path.rstrip("/")), f"worker holds {path}, job asked {model}"
        return pre
    while True:
        jobs = sorted(glob.glob(f"{JOBS}/*.json"))
        if not jobs:
            time.sleep(5); continue
        jf = jobs[0]; job = json.load(open(jf)); os.rename(jf, jf + ".running")
        log = job.get("log", jf + ".log"); t1 = time.time(); ok = True
        print(f"[w27b] start {os.path.basename(jf)} {job['script']} {' '.join(job['argv'])}", flush=True)
        # 10-16 22:3x: purge every project module but uni before a job - a resident process otherwise keeps the module versions of its first
        # import (S829's MASS / R_ALL arms ran the old c_qpath and fell through to the 'perp' edit).
        stale = [m for m, mod in list(sys.modules.items()) if m not in ("uni", "__main__", "__mp_main__") and getattr(mod, "__file__", None)
                 and str(mod.__file__).startswith("/vlm/exp/")]
        for m in stale:
            del sys.modules[m]
        print(f"[w27b] purged {len(stale)} project modules: {sorted(stale)}", flush=True)
        with open(log, "a") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            uni.Uni = factory
            old_argv = sys.argv
            try:
                for k_, v_ in job.get("env", {}).items():
                    os.environ[k_] = v_
                sys.argv = [job["script"]] + list(job["argv"])
                runpy.run_path(job["script"], run_name="__main__")
            except BaseException:
                ok = False; traceback.print_exc()
            finally:
                sys.argv = old_argv; uni.Uni = REAL
                left = clear_new_hooks(pre.model, base)
                torch.cuda.empty_cache()
                print(f"[w27b] job end ok={ok} hooks removed {left} {time.time() - t1:.0f} s", flush=True)
        os.rename(jf + ".running", jf + (".done" if ok else ".failed"))
        print(f"[w27b] {'done' if ok else 'FAILED'} {os.path.basename(jf)} in {time.time() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
