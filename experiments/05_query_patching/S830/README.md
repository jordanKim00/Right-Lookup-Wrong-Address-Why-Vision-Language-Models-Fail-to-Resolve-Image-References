# S830

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S830 (pre-registered in knowledge/DECISIONS.md before any S830 data): does END's late computation answer 'one before the position the early lookup found'? (S829's valid arms: putting START(4)'s address on the early look layers only (35-47) moved N = 8 END(5) to picture 3 (.575); the 'one before the copy' errors recur across N, e.g. END(5) -> 4 at N = 7, 8, 9.) Cases: N = 7 END(4), N = 9 END(5) (identity). Settings exactly as the main series (same recipe, sentence, 160 px pictures, think prefix, two-form readout); seed 83000 (new), 30 scenes per case. Edit rows = the question mark + the five bo

Code: `code/s830_analyze.py`, `code/s830_anchor.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s827_pre_landings.json`, `data/s830_analysis.json`, `data/s830_analysis_s838id_35_4b.json`, `data/s830_analysis_s838id_35_9b.json`, `data/s830_s830_27b_anchor.json`, `data/s830_s838id_35_4b_anchor.json`, `data/s830_s838id_35_9b_anchor.json`
