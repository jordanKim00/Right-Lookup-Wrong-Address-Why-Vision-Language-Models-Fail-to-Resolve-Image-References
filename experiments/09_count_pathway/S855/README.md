# S855

Category: `09_count_pathway`. Image count shown versus stated (factorial) and multimodal rotary geometry.

From the main script's docstring:

> S855 (Gap 2, second stage; pre-registered in knowledge/DECISIONS.md before any S855 data). Does the actual-count dependence of the late lookup come from positional geometry (where the question sits relative to the pictures under multimodal RoPE) or from the extra pictures themselves? Pictures 1-6 occupy the same positions and keys whether 6 or 8 pictures are shown; with 8 shown, every text token after the pictures sits D positions later (D = the position advance of two pictures). Arms (stated count 8 in every prompt, so the text is identical; early lookup pinned to p as in S854-1): E6 6 pictur

Code: `code/s855_analyze.py`, `code/s855_geom.py`

Shared modules: `lib/c_qpath_j.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s855_analysis.json`, `data/s855_rows.json`, `data/s855_s855_27b_geom.json`, `data/s855_xeval.json`
