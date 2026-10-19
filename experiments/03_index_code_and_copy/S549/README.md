# S549

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S549: is the reverse target computed as end + 1 - k with end = the largest block index? Move ONLY the last block's own index code (-1 or +1); every other block keeps its codes. §574: with no count sentence the end is the largest index code carried by the blocks. Rule under test: target index = end + 1 - k, then an absolute-index lookup (§571). Recipient pictures P1..PN. Donors for the last block (residuals at every layer, block with its vision markers): m1 [P1 .. P(N-2), PN] PN sits at index N-1 -> end N-1 -> target j-1 for every k (prediction: offset -1) p1 [P1 .. P(N-1), X, PN] PN sits at in

Re-measured with the clean readout in: S787 (files tagged with the re-run name are kept here).

Code: `code/s549_endshift.py`

Shared modules: `lib/d1_grammar.py`, `lib/uni.py`

Data: `data/s549_endshift_549a27b.json`, `data/s549_endshift_549smk.json`, `data/s549_endshift_bat_35_4b.json`, `data/s549_endshift_bat_35_9b.json`, `data/s549_endshift_bat_4b.json`, `data/s549_endshift_bat_8b.json`, `data/s549_endshift_endshift2_27b.json`, `data/s549_endshift_s787_27b.json`, `data/s549_endshift_s787_35_4b.json`, `data/s549_endshift_s787_35_9b.json`, `data/s549_endshift_s787_4b.json`, `data/s549_endshift_s787_8b.json`
