# S847

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S847 (pre-registered in knowledge/DECISIONS.md before any S847 data; continues S845 after its PATH near miss): does the band's correction reach the answer through the late full-attention queries once the last two full-attention layers (59, 63; S839: START's protection) are included, and are those queries needed? Writers = token mixers + MLPs of layers 46-49; readers = the queries of the ten read rows (question mark, boundary B0-B4, answer-side four) at the late full-attention layers 51, 55, 59, 63. A (one pass per scene): START(k) (donor), END(k), A_BAND (END with the eight band outputs at the

Code: `code/s847_analyze.py`, `code/s847_close.py`

Shared modules: `lib/c_qpath_f.py`, `lib/c_qpath_m.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s847_analysis.json`, `data/s847_qk_map.json`, `data/s847_s847_27b_close.json`
