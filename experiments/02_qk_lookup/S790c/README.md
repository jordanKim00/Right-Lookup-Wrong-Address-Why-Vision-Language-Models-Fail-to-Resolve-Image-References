# S790c

Category: `02_qk_lookup`. Query-key captures of the index-matching heads, the address lens and the nearest start-relative lookup.

From the main script's docstring:

> S790c (saved-capture re-analysis; the full table behind S790b): for each END(k), the distribution over t of the start-relative question START(t) whose block-share distribution is nearest (Jensen-Shannon), per model -- a 7 x 7 matrix per model. Same captures, heads, rows and scenes as S790b. Usage: python s790c_selnn_full.py -> results/r3/s790c_selnn_full.json S790b (saved-capture re-analysis, no model call; control for S790): the S790 query cosine can favour the same-ordinal start-relative query for a lexical reason (END(k) and START(k) share the ordinal word). Selection space has no such conf

Code: `code/s790c_selnn_full.py`

Shared modules: none

Data: `data/s790c_selnn_full.json`
