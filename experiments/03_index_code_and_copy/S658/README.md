# S658

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S658: does a frame placed before the ordinal rescue end-relative reference? (review point: the ordinal token of a postposed question "the third image from the end" cannot see "from the end" under the causal mask, so a literal early state is expected.) Same procedural scenes as S648 (same seed and draw order, so QR and OR reproduce S648 row by row). QF "What is in the {t}th image?" forward ceiling QFP "Counting from the start, what is in the {t}th image?" preposed-construction ceiling QR "What is in the {N+1-t}th image from the end?" postposed (S648 QR) QRP1 "Counting from the end, what is in t

Re-measured with the clean readout in: S787 (files tagged with the re-run name are kept here).

Code: `code/s658_preposed.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s658_pp_27b_preposed.json`, `data/s658_pp_35_4b_preposed.json`, `data/s658_pp_35_9b_preposed.json`, `data/s658_pp_4b_preposed.json`, `data/s658_pp_8b_preposed.json`, `data/s658_s787_27b_preposed.json`, `data/s658_s787_35_4b_preposed.json`, `data/s658_s787_35_9b_preposed.json`, `data/s658_s787_4b_preposed.json`, `data/s658_s787_8b_preposed.json`
