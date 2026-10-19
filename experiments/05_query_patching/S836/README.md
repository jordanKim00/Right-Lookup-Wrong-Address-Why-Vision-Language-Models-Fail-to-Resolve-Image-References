# S836

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S836 (pre-registered in knowledge/DECISIONS.md before any S836 data): where is the END-specific information that the late lookup's queries use - the step of S830-S832 and the reading mode of S829 - written? Path patch into the late reader queries only (exp/c_qpath_d.py): the same-scene START(k) - END(k) residual difference at the input of layer b is added to END(k)'s residual at the input of the late readers (layers 51 and 55) for the question mark, boundary and answer rows, and those readers' queries are recomputed from it (keys, values, output gates and every other layer keep END's own compu

Code: `code/s836_analyze.py`, `code/s836_latewrite.py`

Shared modules: `lib/c_qpath_d.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s836_analysis.json`, `data/s836_s836_27b_latewrite.json`
