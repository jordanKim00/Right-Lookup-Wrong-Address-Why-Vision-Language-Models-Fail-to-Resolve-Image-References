# S837

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S837 (pre-registered in knowledge/DECISIONS.md before any S837 data): which components write the END-specific information that the late lookup's queries use (S836 located the band), and does the size of their contribution set the size of the step (gap 4)? Component path patches into the late reader queries only (exp/c_qpath_e.py 'comps'): for a band of layers given on the command line, the same-pass START(k) - END(k) difference of one component's output (token mixer or MLP of one layer) at the question mark, boundary and answer rows is added to END(k)'s residual at the input of the late reader

Code: `code/s837_analyze.py`, `code/s837_writer.py`

Shared modules: `lib/c_qpath_e.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s837_analysis.json`, `data/s837_s837_27b_writer.json`
