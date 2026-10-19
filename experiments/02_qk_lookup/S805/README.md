# S805

Category: `02_qk_lookup`. Query-key captures of the index-matching heads, the address lens and the nearest start-relative lookup.

From the main script's docstring:

> S805 (descriptive re-analysis of saved S705a QK captures, no model call; proposed by the user 2026-10-15, not pre-registered): is the blurred far end-relative lookup of Qwen3.5-27B (A) diffuse within each head, (B) sharp heads that disagree, or (C) variable across scenes? For each lookup layer L of the S786 index-matching heads (39, 43, 47, 51, 55; L35 has one head), each question (START t, END k, natural wording m0, N = 7, 30 scenes) and the answer row, p[s, h] = the head's attention over the 7 image blocks (softmax over the 462 image-token logits, summed per block). Exact decomposition of th

Code: `code/s805_entropy_decomp.py`, `code/s805j_interp.py`

Shared modules: none

Data: `data/s805_entropy_decomp.json`
