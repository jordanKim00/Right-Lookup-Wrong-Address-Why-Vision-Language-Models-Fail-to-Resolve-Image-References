# S727

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S727: where do 27B's index-matching queries aim when the end-relative reference sits in an option line (the MMIU form)? §750 found that 81% of 27B's MMIU errors pick the lure (the option whose literal ordinal names the target's forward position); this run asks whether the matcher heads' queries at the option lines align with the literal block, at the circuit level. Scenes: procedural images, N = 7 (seed 72700), one label per image; the question "Which image shows a {label}?" with four options "X: The {k}-th image from the {end|start}", the correct option, its lure (k_lure = N + 1 - k_correct, 

Code: `code/s727_analyze.py`, `code/s727_optqk.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s727_oq2_35_9b_analysis.json`, `data/s727_oq2_35_9b_optqk.json`, `data/s727_oq_27b_analysis.json`, `data/s727_oq_27b_optqk.json`, `data/s727_oq_35_4b_analysis.json`, `data/s727_oq_35_4b_optqk.json`, `data/s727_oq_35_9b_analysis.json`, `data/s727_oq_35_9b_optqk.json`
