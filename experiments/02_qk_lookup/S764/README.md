# S764

Category: `02_qk_lookup`. Query-key captures of the index-matching heads, the address lens and the nearest start-relative lookup.

From the main script's docstring:

> S764: same-scene START / END pairs at N = 7 and N = 8 - where every full-attention head from L27 looks, how open its output gate is, and what the model answers. Pre-registration in knowledge/DECISIONS.md (2026-10-13 18:3x) before any S764 data; analysis exp/s764_analyze.py. Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count. N = 7 (seed 76400): END(k) k = 2..6 and START(8 - k). N = 8 (seed 76401): END(k) k = 2..6 and START(9 - k). Each pair names the same image. At N = 8, END(4) and END(5) separate three readings (literal k; two before the last, N - 2 = 6; the cor

Code: `code/s764_analyze.py`, `code/s764_pairs.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s764_analysis.json`, `data/s764_s764_27b7_pairs.json`, `data/s764_s764_27b8_pairs.json`, `data/s764_sm_s764a_pairs.json`, `data/s764_sm_s764b_pairs.json`
