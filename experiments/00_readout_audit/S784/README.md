# S784

Category: `00_readout_audit`. Validity of the answer readout (S782) and the clean-readout re-measurements of every affected experiment (S783-S794, S789).

From the main script's docstring:

> S784 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 18:4x) before any S784 data. S784 re-runs S643 (all ops, seed 64300) and S660 (seed 66000) with R3_READ=bare: content answers are read from the bare tokens (primary); the old space-token readout (ans_space / *_space) and free generation's first word (ans_free / *_free) are recorded alongside. Per model and content op (Start, End, RelO+, RelO-, RelC+, RelC-, EndC, RelOx+, RelOx-): AGREE share of questions where the bare answer equals the free answer >= .95 (instrument check). SPACE_OK share where the space answe

Re-measures: S643, S660 (their files sit in those experiments' folders).

Code: `code/s784_analyze.py`

Shared modules: none

Data: `data/s784_analysis.json`
