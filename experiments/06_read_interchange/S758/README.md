# S758

Category: `06_read_interchange`. Route-complete read interchanges: later tokens read one token as in a donor question, through full attention, the DeltaNet state and its convolution.

From the main script's docstring:

> S758 descriptive tables (exploratory, after the frozen analysis; not a registered verdict). Per arm: answer shares on 6 / 3 / 5 (literal of "fifth") / 2 (literal of "second"), the natural END(5) answer a5 (scenes where a5 is not 3 or 6), the matcher pointer distribution (S705a head-rows, L35-47, tail1 / answer) and the mean block share on each image. Usage: python s758_describe.py TAG

Re-measured with the clean readout in: S783r (files tagged with the re-run name are kept here).

Code: `code/s758_analyze.py`, `code/s758_describe.py`, `code/s758_paths.py`, `code/s775_analyze.py`, `code/s775_dnroute.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s758_analysis.json`, `data/s758_analysis_bare.json`, `data/s758_s758_27b_describe.json`, `data/s758_s758_27b_paths.json`, `data/s758_s783r_27b_paths.json`, `data/s758_sm_s758_paths.json`, `data/s775_s775_27b_dnroute.json`
