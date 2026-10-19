"""S747a for remote-code models that need an older transformers (Kimi-VL: 4.48.2, ERNIE-4.5-VL: <= 4.57.6, their model cards): installs that
version inside the running container, then runs exp/s747_beh_adapter.py in a fresh interpreter.  Same evaluator, items and prompts.
Usage: python s747_legacy.py MODEL TAG [--smoke]"""
import subprocess, sys
model, tag = sys.argv[1], sys.argv[2]
ver = "4.48.2" if "kimi" in model.lower() else "4.57.6"
extra = ["tiktoken", "blobfile"] if "kimi" in model.lower() else ["decord", "moviepy", "einops"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"transformers=={ver}"] + extra, check=True)
r = subprocess.run([sys.executable, "/vlm/exp/s747_beh_adapter.py", model, tag] + sys.argv[3:])
sys.exit(r.returncode)
