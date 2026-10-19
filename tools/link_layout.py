"""Recreate the original workspace layout as symbolic links, so the scripts run unchanged.

The experiments were run in a container with the workspace mounted at /vlm: scripts import from /vlm/exp, read and write
/vlm/results/r3/*.json, and the paper builders read /vlm/research/ces/outputs and /vlm/results/figures.  This script reads
manifests/files.json (original path -> released path) and creates, inside the repository,

    exp/<script>.py            -> lib/, experiments/*/S*/code/, analysis/posthoc/
    exp/figs/<builder>.py      -> paper_builders/
    results/..., research/...  -> experiments/*/S*/data/, paper_builders/inputs/

Mount the repository at /vlm (docker run -v $PWD:/vlm ...) and the original absolute paths resolve.  Model checkpoints are expected
under /models (Qwen3.5) and /dev/shm/vlm_models (Qwen3-VL, Qwen3.5-4B / 9B); see docs/REPRODUCE.md.
Usage: python tools/link_layout.py [--force]"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
force = "--force" in sys.argv
made = 0
for e in json.load(open(os.path.join(ROOT, "manifests", "files.json"))):
    src = e["src"]
    if src.startswith("archive/legacy_2026-09-08/"):
        src = src[len("archive/legacy_2026-09-08/"):]
    link = os.path.join(ROOT, src)
    target = os.path.join(ROOT, e["dst"])
    if not os.path.exists(target) or os.path.abspath(link) == os.path.abspath(target):
        continue
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.lexists(link):
        if not force:
            continue
        os.remove(link)
    os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
    made += 1
print(f"{made} links created under exp/, results/ and research/")
