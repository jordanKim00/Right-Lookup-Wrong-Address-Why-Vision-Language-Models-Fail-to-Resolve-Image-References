# S759

Category: `06_read_interchange`. Route-complete read interchanges: later tokens read one token as in a donor question, through full attention, the DeltaNet state and its convolution.

From the main script's docstring:

> S759: where after layer 26 is the frame word read, and through which route? Complete-route read interchange of the frame word. Pre-registration in knowledge/DECISIONS.md (2026-10-13 13:2x) before any S759 data; analysis exp/s759_analyze.py. Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt. Per scene END(2), END(5), START(2), START(5), same images and menu. Same read interchange as S758 (exp/s758_paths.py): at every layer of a set the sequence mixer receives the donor's normalized input at the source token p, so every later row reads the donor's p through every route, and the mi

Code: `code/s759_analyze.py`, `code/s759_frame.py`, `code/s766_analyze.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s759_analysis.json`, `data/s759_s759_27b_frame.json`, `data/s759_s766_27b_frame.json`, `data/s759_sm_s759_frame.json`
