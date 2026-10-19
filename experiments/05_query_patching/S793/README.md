# S793

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S793 (pre-registered in knowledge/DECISIONS.md before any S793 data): does the natural end-relative error live in the lookup queries? Bidirectional query transplant between the natural END(k) run and the START(j) run of the same scene (same images and menu; the two prompts differ only in the ordinal and frame words, so every token position aligns), at the full-attention reader layers of two depth bands, for the rows from the question mark to the answer. Qwen3.5 only (exp/c_qpath.py recomputes the gated full attention of Qwen3.5). Scenes: procedural pictures (160 px), N = 7, seed 79300 (new), 4

Code: `code/s793_analyze.py`, `code/s793_transplant.py`

Shared modules: `lib/c_qpath.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s793_analysis.json`, `data/s793_s793_27b_transplant.json`, `data/s793_s793_35_4b_transplant.json`, `data/s793_s793_35_9b_transplant.json`
