# S741b

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S741b: does the lookup read the index code from the image keys? (follow-up to S741, reply to the mock review W2, 2026-10-11; pre-registration in knowledge/DECISIONS.md before any S741b data; analysis exp/s741b_analyze.py, frozen with it.) S741 showed that the S545 code shift is not carried by the seven-dimensional span of the block position means. Here the same donor (two new pictures in front, so blocks 1..N hold codes 3..N+2) is split by what attention reads from the image tokens instead: NONE no edit FULL every image-block token's residual state <- the donor's, at every layer (S545) KEY the

Re-measured with the clean readout in: S789 (files tagged with the re-run name are kept here).

Code: `code/s741b_analyze.py`, `code/s741b_kv.py`

Shared modules: `lib/d1_grammar.py`, `lib/uni.py`

Data: `data/s741b_analysis.json`, `data/s741b_s741b_4b_kv.json`, `data/s741b_s741b_9b_kv.json`, `data/s741b_s789kv_4b_kv.json`, `data/s741b_s789kv_9b_kv.json`
