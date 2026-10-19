# S720

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S720 head sets: index-matching heads of Qwen3.5-4B / 9B from the saved S705a arrays (START diagonal share >= .5 at the answer or tail1 row on the discovery scenes 0-14) -- the small-model analogue of S705a's MATCHER set without the END condition (small models do not aim END at the target, §740). Writes /vlm/results/r3/s720_{4b,9b}_heads.json.

Re-measured with the clean readout in: S789 (files tagged with the re-run name are kept here).

Code: `code/s720_analyze.py`, `code/s720_heads.py`, `code/s720_select.py`

Shared modules: `lib/c_pat.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s720_4b_heads.json`, `data/s720_9b_heads.json`, `data/s720_analysis.json`, `data/s720_s720s_4b_select.json`, `data/s720_s720s_9b_select.json`, `data/s720_s789sel_4b_select.json`, `data/s720_s789sel_9b_select.json`, `data/s720_sel_4b_select.json`, `data/s720_sel_9b_select.json`
