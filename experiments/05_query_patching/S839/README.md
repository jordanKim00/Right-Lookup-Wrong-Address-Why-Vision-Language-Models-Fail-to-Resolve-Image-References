# S839

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S839 (pre-registered in knowledge/DECISIONS.md before any S839 data): gaps 1 and 2 of the user's list (10-17) - what is the END reading mode (S829: the non-address part of the queries at every look layer repairs much of the identity error, P_FULL +.43 / +.40, while the picture-vs-text direction built from the question text before '?' does not), and what protects START from END's query (S829: END's whole query at the look layers 35-55 lowers START only by .30 / .58; S820: the answer rows read the menu words with full attention at layers 55-63)? Cases: identity (N, k) = (7, 4), (9, 5); donor STA

Code: `code/s839_analyze.py`, `code/s839_mode2.py`

Shared modules: `lib/c_qpath_b.py`, `lib/c_qpath_m.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s839_analysis.json`, `data/s839_s839_27b_mode2.json`
