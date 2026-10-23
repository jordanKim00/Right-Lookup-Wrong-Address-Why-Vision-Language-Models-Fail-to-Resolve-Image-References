"""Thread safety for Triton's autotuner (10-22, infrastructure): Triton 3.1's Autotuner keeps per-call state on the instance (self.nargs) while it
benchmarks a new key, so two threads hitting a new shape at once crash ("'NoneType' object is not a mapping", S897 run 415f7).  install()
wraps Autotuner.run in one process-wide lock (idempotent; the patch stays in the resident worker for later jobs).  Kernel launches through an
already-tuned key only hold the lock for the launch call."""
from __future__ import annotations

import threading


def install():
    import triton.runtime.autotuner as at
    if getattr(at.Autotuner, "_r3_locked", False):
        return False
    orig = at.Autotuner.run
    lock = threading.RLock()

    def run(self, *args, **kwargs):
        with lock:
            return orig(self, *args, **kwargs)
    at.Autotuner.run = run
    at.Autotuner._r3_locked = True
    return True
