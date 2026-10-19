# S740

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S740: does R³'s repair on photographs travel through the full-attention queries? (pre-registration knowledge/DECISIONS.md 2026-10-11, before any S740 data; analysis exp/s740_analyze.py, frozen with it). The S722 design on natural items. Items: the MMIU development split (exp/s737_natqk.dev_items: first 50 items per position task, N <= 9), prompts as the evaluator builds them, end-relative options ("X: The m-th image from the end"). R³ = the model's final configuration (exp/r3_select.build, plan file in force). R³ writes only at option tokens after the images, so image keys and values are ident

Code: `code/s740_analyze.py`, `code/s740_natmed.py`

Shared modules: `lib/c_tools.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s740_analysis.json`, `data/s740_nm740_27b_natmed.json`, `data/s740_nm740_35_4b_natmed.json`, `data/s740_nm740_35_9b_natmed.json`
