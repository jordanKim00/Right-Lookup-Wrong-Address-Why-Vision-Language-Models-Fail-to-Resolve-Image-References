# S790b

Category: `02_qk_lookup`. Query-key captures of the index-matching heads, the address lens and the nearest start-relative lookup.

From the main script's docstring:

> S790b (saved-capture re-analysis, no model call; control for S790): the S790 query cosine can favour the same-ordinal start-relative query for a lexical reason (END(k) and START(k) share the ordinal word). Selection space has no such confound: for each (held-out scene, matcher head), which start-relative question START(t), t = 1..7, has the block-share distribution nearest (Jensen-Shannon) to that of END(k)? Reported: share of pairs whose nearest t is k (copy), N+1-k (composition) or another t, per model and k. Same captures, heads and rows as S790. Usage: python s790b_selnn.py -> results/r3/s

Code: `code/s790b_selnn.py`, `code/s799_qaxis.py`

Shared modules: none

Data: `data/s790b_selnn.json`, `data/s799_qaxis.json`
