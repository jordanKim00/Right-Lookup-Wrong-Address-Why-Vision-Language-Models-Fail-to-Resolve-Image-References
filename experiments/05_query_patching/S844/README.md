# S844

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S844 (pre-registered in knowledge/DECISIONS.md before any S844 data; user 10-17): separate the predecessor bias of the late update from goal- conditioned movement. 8 pictures shown (the same 8 pictures for every condition; main-series recipe: 8 labels + the first unused word), ordinal k = 4, stated count N_s in {6, 7, 8, 9} (stated target t = N_s - 3: 3, 4, 5, 6; N_s = 8 is the matched count), early lookup pinned to p in {3, 4, 5, 6} (E_EARLY_p: the address part U of the question-mark, boundary and answer rows set to START(p)'s at the early look layers 35-47 only), so the stated target lies le

Code: `code/s844_analyze.py`, `code/s844_bias.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s844_analysis.json`, `data/s844_frozen_laws.json`, `data/s844_s844_27b_anchor.json`, `data/s844_within_p.json`
