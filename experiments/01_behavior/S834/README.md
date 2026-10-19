# S834

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S834 (descriptive, registered in knowledge/DECISIONS.md before any S834 data; no hypothesis): complete the natural END(k) answer grid of the v2 paper figure (fig_grid) at N = 7 and 8, whose k = 1, 2 (and k = 8 at N = 8) cells S824 did not run. Scenes are those of S824: the scene code below is S824's (exp/s824_count.py) line for line - seed 82400, 40 scenes, 8 pictures drawn per scene, the 7-picture condition shows the first seven, menu = the eight labels + one extra word - so every scene is checked against S824's saved labels and menu. Conditions (shown = stated count): N = 7, k in {1, 2, 3}; 

Code: `code/s834_analyze.py`, `code/s834_grid.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s834_analysis.json`, `data/s834_s834_27b_grid.json`
