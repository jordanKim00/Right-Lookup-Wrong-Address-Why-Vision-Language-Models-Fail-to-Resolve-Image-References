# S769

Category: `06_read_interchange`. Route-complete read interchanges: later tokens read one token as in a donor question, through full attention, the DeltaNet state and its convolution.

From the main script's docstring:

> S769: the END(4) paradox - at N = 7 the end-relative target equals the literal image 4, so no arithmetic is needed, yet END(4) fails. Does the frame word read at L27-34 alone cause the failure? Complete-route read interchange of the frame word (S758 / S759 instrument). Pre-registration in knowledge/DECISIONS.md (2026-10-13 23:1x) before any S769 data; analysis exp/s769_analyze.py. Qwen3.5-27B, procedural pictures (160 px), the S749 prompt. Per scene END(4) and START(4) (same pictures, menu). N = 7 (seed 76900): both target image 4. N = 8 (seed 76901): START(4) targets 4, END(4) targets 5 (lite

Re-measured with the clean readout in: S783a (files tagged with the re-run name are kept here).

Code: `code/s769_analyze.py`, `code/s769_end4.py`, `code/s769_run.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s769_analysis.json`, `data/s769_analysis_bare.json`, `data/s769_s769_27b7_end4.json`, `data/s769_s769_27b8_end4.json`, `data/s769_s783e_27b7_end4.json`, `data/s769_s783e_27b8_end4.json`, `data/s769_sm_s769a_end4.json`, `data/s769_sm_s769b_end4.json`
