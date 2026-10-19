# S783

Category: `00_readout_audit`. Validity of the answer readout (S782) and the clean-readout re-measurements of every affected experiment (S783-S794, S789).

From the main script's docstring:

> S783 descriptive comparison (not a registered verdict): the registered analyses of S780 / S769 / S772 / S778 / S770 / S758 / S781 under the space-token readout (original runs) and the bare-token readout (S783 re-runs, same scenes), side by side, plus the share of answer readouts whose vocabulary-wide top-1 token is a bare menu token (when it is, the bare readout equals greedy generation). Usage: python s783_compare.py

Code: `code/s783_compare.py`, `code/s783_run.py`

Shared modules: `lib/uni.py`

Data: none released here
