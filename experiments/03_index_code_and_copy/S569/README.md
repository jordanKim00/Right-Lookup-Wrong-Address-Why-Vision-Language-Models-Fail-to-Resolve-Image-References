# S569

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S569: at which layers does the question read the END? The S567 full-attention route (keys + values of the last block lowered to index N-1) applied only in windows of full-attention layers (27B: full layers 3, 7, ..., 63). §596: the whole effect of lowering the last block (k = 2: j-1 18/24) comes through the full-attention keys/values. Windows: early L3-23, mid L27-39, L43, L47, L51, L55-63, and L43+47. Reverse questions without a count sentence, k = 2. S567 docstring follows. S567: through which route does the question read the END? Route-specific transplants into the LAST block (27B). §593: l

Code: `code/s569_endlayer.py`

Shared modules: `lib/d1_grammar.py`, `lib/uni.py`

Data: `data/s569_endlayer_569a27b.json`, `data/s569_endlayer_smk569.json`
