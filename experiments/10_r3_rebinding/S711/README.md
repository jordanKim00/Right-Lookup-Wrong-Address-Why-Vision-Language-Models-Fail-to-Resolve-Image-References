# S711

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S711: R³-v2 -- the one change the 27B mechanism asks for: an ordinal bound to an END frame always gets the frame write (END -> START), also when its resolved target equals the literal ordinal (t = k_hat, no ordinal write). Pre-registration in knowledge/DECISIONS.md before any S711 data. Why: S709 -- every R³ failure but two sat at N7 k4, where t = k_hat made R³-min write nothing and 27B's END policy left the answer at .40 although the address equals the literal ordinal (FINDINGS §733, §743); the oracle with the frame write alone gave 1.00. Readers and calibration exactly as S709 (exp/s709_cal.

Code: `code/s711_analyze.py`, `code/s711_eval.py`

Shared modules: `lib/r3_build.py`, `lib/r3_build_md.py`, `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/uni.py`

Data: `data/s711_analysis.json`, `data/s711_v2_27b_eval.json`
