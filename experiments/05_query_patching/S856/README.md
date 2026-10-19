# S856

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S856 (P1, user 10-18; pre-registered in knowledge/DECISIONS.md before any S856 data). Does the visual-prefix-conditioned late query carry the actual-count difference of the lookup? The same scene with 6 pictures (X6 = pictures 1-6) or 8 (X8 = 1-8), the same menu; k in {3, 5} (targets for 6 / 8 pictures: k 3 -> 4 / 6, k 5 -> 2 / 4, all among the shared pictures 1-6), early lookup pinned to p in {3, 4}. Two prompt forms: A states 'There are 8 images in a row.' (X6 is then a counterfactual probe), NC has no count sentence (no mismatch). stage 1 END(k), START(q) per form and picture count (pin don

Code: `code/s856_analyze.py`, `code/s856_qswap.py`

Shared modules: `lib/c_qpath_j.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s856_analysis.json`, `data/s856_s856_27b_qswap.json`, `data/s856_s856b_27b_qswap.json`, `data/why_decomp.json`, `data/why_profiles.json`, `data/why_shapes.json`, `data/why_writer.json`
