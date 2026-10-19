# S790

Category: `02_qk_lookup`. Query-key captures of the index-matching heads, the address lens and the nearest start-relative lookup.

From the main script's docstring:

> S790 (saved-capture analysis, no model call): is the end-relative query the start-relative query of the same ordinal? For the index-matching heads of Qwen3.5-4B / 9B / 27B (S705a captures, held-out half of the scenes, answer row), compare each END(k) query with START(k) (same ordinal word, copy) and START(N+1-k) (same image, composition): mean cosine over heads and scenes, and the share of (scene, head) pairs whose END(k) query is closer to START(k) than to START(N+1-k). Also the lookup's own decoded address: the block-share-weighted mean block index E[b] of each query (no fitted map). N = 7, 

Code: `code/s790_qident.py`

Shared modules: none

Data: `data/s790_qident.json`
