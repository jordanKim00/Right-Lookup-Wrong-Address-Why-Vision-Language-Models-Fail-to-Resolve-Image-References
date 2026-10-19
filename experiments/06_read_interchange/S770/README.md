# S770

Category: `06_read_interchange`. Route-complete read interchanges: later tokens read one token as in a donor question, through full attention, the DeltaNet state and its convolution.

From the main script's docstring:

> S770: which part of the full-attention read at L35-59 carries the frame effect (S768's missing third) and the late query control of START answers (S767 late .85, image selection alone 0)? Output split o = G * (M * D + (1 - M) * T) (exp/c_dmtg.py): image mass M, image read D (which images), text read T (where in the text and what it carries), output gate G. Pre-registration in knowledge/DECISIONS.md (2026-10-13 23:4x) before any S770 data; analysis exp/s770_analyze.py. Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt, new scenes (seed 77000). Per scene END(5), START(5), START(3)

Re-measured with the clean readout in: S783b (files tagged with the re-run name are kept here).

Code: `code/s770_analyze.py`, `code/s770_dmtg.py`

Shared modules: `lib/c_dmtg.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s770_analysis.json`, `data/s770_analysis_bare.json`, `data/s770_s770_27b_dmtg.json`, `data/s770_s783d_27b_dmtg.json`, `data/s770_sm_s770_dmtg.json`
