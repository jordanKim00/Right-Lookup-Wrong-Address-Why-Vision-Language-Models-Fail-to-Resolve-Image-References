# S737

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S737: the option-line query on natural images (pre-registration knowledge/DECISIONS.md 2026-10-10, before any S737 data). Items: the MMIU development split (exp/s582_evalrepair.run's selection, first 50 items per position task, N <= 9), prompts exactly as the evaluator builds them. Forms: START (the original options "X: The k-th image"), END (the evaluator's end-relative rewrite "X: The (N+1-k)-th image from the end"), END + the model's R³ (exp/r3_select.build, plan file in force). Rows: the ordinal token of every option line (the last n_options ordinal words of the prompt; the option block en

Code: `code/s737_analyze.py`, `code/s737_natqk.py`

Shared modules: `lib/c_tools.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s737_analysis.json`, `data/s737_nq737_27b_natqk.json`, `data/s737_nq737_35_4b_natqk.json`, `data/s737_nq737_35_9b_natqk.json`, `data/s737_nq737_4b_natqk.json`, `data/s737_nq737_8b_natqk.json`, `data/s737_nq737f_27b_natqk.json`
