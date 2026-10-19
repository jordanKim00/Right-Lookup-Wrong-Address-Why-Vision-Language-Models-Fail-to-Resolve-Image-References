# S787

Category: `00_readout_audit`. Validity of the answer readout (S782) and the clean-readout re-measurements of every affected experiment (S783-S794, S789).

From the main script's docstring:

> S787 27B final (registered analysis rules unchanged, exp/s787_analyze.py functions): S652 fit (I0-I4), S658 f rule, S736 checks (FRAME_NEEDED_NEAR, FRAME_SEMANTIC, FRAME_NEEDED_ALL for 27B) and the re-reflection = composition law on the 27B cells.

Re-measures: S545, S549, S474b, S658, S736 (their files sit in those experiments' folders).

Code: `code/s787_27b_final.py`, `code/s787_27b_fit_view.py`, `code/s787_27b_partial_view.py`, `code/s787_27b_s658_view.py`, `code/s787_analyze.py`, `code/s787_run.py`

Shared modules: `lib/uni.py`

Data: `data/s787_27b_final.json`, `data/s787_analysis.json`
