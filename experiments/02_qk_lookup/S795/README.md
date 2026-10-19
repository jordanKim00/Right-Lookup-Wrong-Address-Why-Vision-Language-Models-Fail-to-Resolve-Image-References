# S795

Category: `02_qk_lookup`. Query-key captures of the index-matching heads, the address lens and the nearest start-relative lookup.

From the main script's docstring:

> S795 (pre-registered in knowledge/DECISIONS.md before any S795 data): an address lens -- how the end-relative query is formed over depth. No answer is read (QK only), so the menu-token readout plays no part. For each lookup layer L* in the index-matching band, the residual stream at the answer position after every earlier layer l is passed through L*'s own input norm, query projection, q_norm and rotary embedding, and matched against L*'s actual keys of the image tokens: the block shares of the index-matching heads at L* tell which image the address carried by the stream up to layer l would se

Code: `code/s795_analyze.py`, `code/s795_lens.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s795_analysis.json`, `data/s795_s795_27b_lens.json`, `data/s795_s795_35_4b_lens.json`, `data/s795_s795_35_9b_lens.json`
