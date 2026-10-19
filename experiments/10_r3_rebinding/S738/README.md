# S738

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S738 (1): is the R³ calibration set unrelated to every evaluation set? Perceptual-hash audit, no model call. Calibration images = the natural photographs used by S724 / S729 (exp/s724_readcal.nat_keys(Random(72400), 12000): caltech256 and food101 rows, R³-Ref images excluded). Procedural calibration images are rendered shapes and are reported but not matched. Evaluation images = every image file, parquet image cell or zip member under the evaluation roots (MMIU all tasks incl. visual cloze, MuirBench, BLINK / Q-Bench2 / Mantis / MMMU / MIBench / MIRB (assets/r3_bench), MathVista, imagenette, C

Code: `code/s738_audit.py`, `code/s738_sheets.py`

Shared modules: none

Data: `data/c8gen_27b_general.json`, `data/s738_audit.json`
