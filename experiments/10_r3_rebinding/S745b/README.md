# S745b

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S745b analysis (amended after the S745 gate, before any main-stage data; knowledge/DECISIONS.md 2026-10-11 20:29): a copy of exp/s745_analyze.py with the gate at .40 and the landing thresholds relative to the model's own start-relative accuracy. GATE start-relative accuracy >= .40 on the discovery scenes and >= 5 index-matching heads (else the lookup cannot be traced) LITERAL_QUERY END: the heads' share on the literal block > on the target, and END accuracy <= START accuracy - .20 ADDRESS_CONTROL pooled over the early layers 2, 4, 6: answers on the written position x >= .5 x START accuracy and

Code: `code/s745b_analyze.py`

Shared modules: none

Data: `data/s745b_analysis.json`
