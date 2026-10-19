# S708

Category: `04_activation_patching`. Residual-stream transplants and writes at the ordinal token and after it (ordinal state as a position, written addresses).

From the main script's docstring:

> S708: what the identified mechanism predicts for a repair of 27B. Pre-registration in knowledge/DECISIONS.md before any S708 data. Mechanism (FINDINGS §729-§741): one index-matching route (image keys carry a forward index code); START queries hit it exactly for every position; 27B's END query aims one step before the visible end and blurs beyond; 27B applies the END frame as an operator to whatever ordinal address arrives (S679). Predictions for an early rewrite at the ordinal interface (R3-style, oracle address): rewriting the ordinal to the forward target AND the frame to START sends every E

Re-measured with the clean readout in: S789 (files tagged with the re-run name are kept here).

Code: `code/s708_analyze.py`, `code/s708_rebind.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s708_analysis.json`, `data/s708_rb_27b_rebind.json`, `data/s708_s789rb_27b_rebind.json`
