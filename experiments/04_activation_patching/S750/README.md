# S750

Category: `04_activation_patching`. Residual-stream transplants and writes at the ordinal token and after it (ordinal state as a position, written addresses).

From the main script's docstring:

> S750 exploratory analysis (planned before any S750 data, DECISIONS 2026-10-12 22:5x; not a registered verdict). For a donor END(k') split by whether its natural answer was right or wrong, how often does each (layer, token group) residual transplant make the recipient answer the donor's ACTUAL choice (C), against the donor's target (F) and the recipient staying on its own target (R)? Baseline: the recipient's natural answer equals the donor's choice. Pairs: (2 <- 5) main, (5 <- 4) and (5 <- 2) for contrast. Usage: python s750_explore_donorchoice.py TAG

Code: `code/s750_analyze.py`, `code/s750_explore_donorchoice.py`, `code/s750_merge.py`, `code/s750_where.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s750_analysis.json`, `data/s750_s750_27b_p0_where.json`, `data/s750_s750_27b_p1_where.json`, `data/s750_s750_27b_p2_where.json`, `data/s750_s750_27b_p3_where.json`, `data/s750_s750_27b_p4_where.json`, `data/s750_s750_27b_where.json`, `data/s750_sm_s750_where.json`
