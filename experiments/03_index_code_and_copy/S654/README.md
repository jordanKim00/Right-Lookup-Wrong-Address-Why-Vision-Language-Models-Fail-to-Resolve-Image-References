# S654

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S654: content-first vs reference-first yes/no with END and START frames. Per scene (procedural, N 5-7, k 2-4, j = N+1-k != k fixed before outcomes): three contents of the SAME scene T = image j (END target), L = image k (start-relative lure), U = image u, u not in {j, k} (neutral). Twelve paired prompts = frame {END, START} x order {CF "Is the {X} in the {k-th} image[ from the end]?", RF "Does the {k-th} image[ from the end] show the {X}?"} x role {T, L, U}, each ending "Answer yes or no." The ordinal word is always k (END target j; STA

Code: `code/s654_yn.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s654_yn_27b_yn.json`, `data/s654_yn_35_4b_yn.json`, `data/s654_yn_35_9b_yn.json`, `data/s654_yn_4b_yn.json`, `data/s654_yn_8b_yn.json`
