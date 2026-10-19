# S568

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S568: does the retrieval query's attention follow the block that carries the computed target code? A QK readout under the code manipulations whose behavioural effects are known (27B, reverse questions without a count sentence). Arms: none; M2 (the last block claims N-1 -> the target code becomes j-1; §593 k = 2: 27/36 land on j-1); res_frozen (the target block j claims x, blocks after it frozen -> code j is missing; §591); zclaim (another block z, 1 <= z <= j-2, claims code j, blocks after z frozen -> two blocks carry code j). Readout: at every full-attention layer, the answer token's attentio

Code: `code/s568_qkfollow.py`

Shared modules: `lib/d1_grammar.py`, `lib/uni.py`

Data: `data/s568_qkfollow_568a27b.json`, `data/s568_qkfollow_smk568.json`
