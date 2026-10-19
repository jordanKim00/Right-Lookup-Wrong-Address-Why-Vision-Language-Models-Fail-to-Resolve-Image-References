# S545

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S545: how does 27B know where the end is? Shift every image block's own index by +2 without changing content or order. An image block cannot know that it is the last one (causal mask), so the end must be inferred from the blocks by the text that follows. Candidates: (i) the largest index code among the blocks ("the last block says it is the N-th"), (ii) the number of blocks, (iii) no index at all (a relation). Donor scene: two new pictures followed by the recipient's N pictures, so donor blocks 3..N+2 hold the recipient's pictures 1..N but believe they are the 3rd..(N+2)-th. Every image-block 

Re-measured with the clean readout in: S787 (files tagged with the re-run name are kept here).

Code: `code/s545_indexshift.py`, `code/s545b_endcues.py`

Shared modules: `lib/d1_grammar.py`, `lib/uni.py`

Data: `data/s545_indexshift_545a27b.json`, `data/s545_indexshift_545ca4b.json`, `data/s545_indexshift_545ca9q.json`, `data/s545_indexshift_545smk.json`, `data/s545_indexshift_bat_27b.json`, `data/s545_indexshift_bat_35_4b.json`, `data/s545_indexshift_bat_35_9b.json`, `data/s545_indexshift_bat_4b.json`, `data/s545_indexshift_bat_8b.json`, `data/s545_indexshift_s787_27b.json`, `data/s545_indexshift_s787_35_4b.json`, `data/s545_indexshift_s787_35_9b.json`, `data/s545_indexshift_s787_4b.json`, `data/s545_indexshift_s787_8b.json`
