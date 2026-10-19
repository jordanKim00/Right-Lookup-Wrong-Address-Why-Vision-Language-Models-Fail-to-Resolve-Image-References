# S778

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S778: are the turn-boundary rows reading sites? S777: in START the boundary row reads the target image at L35-47 and carries the answer's content, in END it does not. Row-specific query transplant (sufficiency) and attention knockout (necessity) at the full-attention layers L35-47. Pre-registration in knowledge/DECISIONS.md (2026-10-14 12:2x) before any S778 data; analysis exp/s778_analyze.py. Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count, wording "last"; cells N = 7 and N = 8, 40 new scenes each. Per scene E5 (target ta = N - 4), Sa = START(ta), E2 (target t

Re-measured with the clean readout in: S783b (files tagged with the re-run name are kept here).

Code: `code/s778_analyze.py`, `code/s778_readrows.py`, `code/s778_run.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s778_analysis.json`, `data/s778_analysis_bare.json`, `data/s778_s778_27b7_readrows.json`, `data/s778_s778_27b8_readrows.json`, `data/s778_s783b_27b7_readrows.json`, `data/s778_s783b_27b8_readrows.json`, `data/s778_sm_s778_readrows.json`
