# S705b

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S705b: is 27B's END failure a failure of image selection at L35-47, which part of the selection matters (which image vs how much image), and does the matching heads' failure travel through their query or their output gate? Pre-registration in knowledge/DECISIONS.md before data. Motivation: S704 -- imposing the START selection at L51 and later does not steer the answer (SEL_W .19); the decisive selection is earlier (S687a re-analysis: only correct END items transmit the target block at L35-47; S688 tables: matcher heads at L39 / L47). Qwen3.5-27B, N = 7, new scenes (seed 70510), targets t in {2

Code: `code/s705b_analyze.py`, `code/s705b_select.py`

Shared modules: `lib/c_pat.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s705b_analysis.json`, `data/s705b_b5_27b_select.json`
