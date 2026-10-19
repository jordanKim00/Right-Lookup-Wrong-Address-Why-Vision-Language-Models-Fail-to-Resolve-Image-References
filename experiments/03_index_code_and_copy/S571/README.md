# S571

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S571: where is the end turned into the target code? Path decomposition between the L47 end reading and retrieval. §599: the whole effect of lowering the end (k = 2: j -> j-1) enters through L47's attention output at the suffix (P47 = 100%), and the L51 query carries about half of it. Between L47's attention and L51 lie L47's MLP and three linear-attention layers (L48-50, each Gated DeltaNet + MLP). On top of P47, components at the suffix positions are held at their clean-run values: P47_mlp MLP outputs of L47-L50 P47_lin linear-attention outputs of L48-L50 P47_both both (the end information th

Code: `code/s571_arith.py`

Shared modules: `lib/d1_grammar.py`, `lib/uni.py`

Data: `data/s571_arith_571a27b.json`, `data/s571_arith_smk571.json`
