# S856b

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S856b (rerun of S856 after its reference check missed by one run; pre-registered in knowledge/DECISIONS.md before any S856b data). Change from S856: every stage-2 and stage-3 batch of a form and picture count holds exactly 10 runs (S856 captured donor queries in 8-run batches and injected them into 20-run batches; with 6 pictures the two batch shapes differed numerically), and both prompt forms carry all five stage-3 arms. Otherwise as S856: Does the visual-prefix-conditioned late query carry the actual-count difference of the lookup? The same scene with 6 pictures (X6 = pictures 1-6) or 8 (X8

Code: `code/s856b_analyze.py`, `code/s856b_qswap.py`

Shared modules: `lib/c_qpath_j.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s856b_analysis.json`
