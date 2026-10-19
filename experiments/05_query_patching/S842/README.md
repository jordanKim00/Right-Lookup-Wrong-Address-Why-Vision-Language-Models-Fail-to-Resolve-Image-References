# S842

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S842 (pre-registered in knowledge/DECISIONS.md before any S842 data): gap 1 continued - where in the non-address part of the queries is the END reading mode? S839: not the picture-vs-menu, picture-vs-all-text or norm directions (the repairing non-address edit works without the menu direction). Here the START-minus-END difference of the queries (look layers 35-55, question-mark + boundary + answer rows) is split into orthogonal subspaces per head (exp/c_qpath_m.py): W = within-picture token pattern (top 16 directions of image-token keys minus their block mean), T = text keys (top 16 directions 

Code: `code/s842_analyze.py`, `code/s842_mode3.py`

Shared modules: `lib/c_qpath_b.py`, `lib/c_qpath_m.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s842_analysis.json`, `data/s842_s842_27b_mode3.json`
