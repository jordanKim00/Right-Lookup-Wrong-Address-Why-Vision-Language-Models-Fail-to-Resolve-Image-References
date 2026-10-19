# S709

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S709-cal: model-internal R³ readers for Qwen3.5-27B under the existing R³ calibration protocol (no end-relative answers anywhere). (1) End reader (S609 protocol, exp/s609_reader.py): block index codes at the vision_end site, fitted on images-first procedural scenes (N 2-10, forward questions "What is in the k-th image?"), checked on held-out scenes in the four S608 layouts; candidate layers L35 / 39 / 43 / 47 / 51 (27B; relative .56-.81) -- the layer with the best mean held-out accuracy is kept. Saved as /vlm/results/s588c_reader_s709_{base}.pt (the format r3_build / r3_repair load). (2) R³ ca

Code: `code/s709_analyze.py`, `code/s709_cal.py`, `code/s709_eval.py`

Shared modules: `lib/r3_build.py`, `lib/r3_build_md.py`, `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/s608_layout.py`, `lib/uni.py`

Data: `data/s709_analysis.json`, `data/s709_e2e_27b_eval.json`, `data/s709_s709se_eval.json`
