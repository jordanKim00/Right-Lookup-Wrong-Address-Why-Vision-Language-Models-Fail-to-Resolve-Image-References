# S833

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S833 (pre-registered in knowledge/DECISIONS.md before any S833 data): does R3's frame rebind work by turning off END's late re-lookup? Phase-1 mechanism (§921-§929): END re-reads pictures late (L51-55) and moves the early-found position a fixed step toward its end-relative target, even at the target. Predictions: writing the frame END -> START (R3's Rebind of the frame) turns the late re-reading off (late picture mass START-like) so the early lookup's position stands; writing only the ordinal k -> j leaves the END frame on, the early lookup moves to j and the late step moves it toward the end-

Code: `code/s833_analyze.py`, `code/s833_r3mode.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s833_analysis.json`, `data/s833_s833_27b_r3mode.json`
