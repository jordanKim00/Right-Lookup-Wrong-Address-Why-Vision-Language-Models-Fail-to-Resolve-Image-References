# S739

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S739: does the index code count image blocks or tokens? Pre-registration in knowledge/DECISIONS.md (2026-10-10 20:48) before any S739 data; analysis exp/s739_analyze.py (frozen with it). The S730 harness with one change in the scenes: the same scene and pictures are shown in two size regimes -- UNI (all seven images 224 px) and MIX (each image 224, 448 or 672 px, drawn independently; block token counts recorded). Scenes: procedural, N = 7, seed 73900, k in {2, ..., 6} (target j = N + 1 - k), the S730 question form and menu. Arms per regime: START (forward wording of the same target), END (end-

Re-measured with the clean readout in: S789 (files tagged with the re-run name are kept here).

Code: `code/s739_analyze.py`, `code/s739_tokens.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s739_analysis.json`, `data/s739_s789tk_27b_tokens.json`, `data/s739_s789tk_35_4b_tokens.json`, `data/s739_s789tk_35_9b_tokens.json`, `data/s739_s789tk_4b_tokens.json`, `data/s739_s789tk_8b_tokens.json`, `data/s739_tk739_27b_tokens.json`, `data/s739_tk739_35_4b_tokens.json`, `data/s739_tk739_35_9b_tokens.json`, `data/s739_tk739_4b_tokens.json`, `data/s739_tk739_8b_tokens.json`, `data/s739_tk739s_4b_tokens.json`
