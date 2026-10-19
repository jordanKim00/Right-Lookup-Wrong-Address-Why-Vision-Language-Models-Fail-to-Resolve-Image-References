# S745

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S745: the core causal tests in a famous VLM with a non-Qwen language model, without R³ (reply to the mock reviews; the user: "LLaVA 말고 선행 연구에서 자주 쓰는 유명한 모델로"). Pre-registration in knowledge/DECISIONS.md before any S745 data; analysis exp/s745_analyze.py, frozen with it. Procedural scenes of seven pictures (the S730 question form and menu, 160 px). 0 GATE discovery scenes (seed 74500, 15 scenes, START t = 1..7): start-relative accuracy; index-matching heads = mean START diagonal share >= .5 at the answer row (the S730 rule for dense models) 1 MEASURE test scenes (seed 74501, 30), k in {2, 3, 5,

Code: `code/s745_analyze.py`, `code/s745_core.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s745_s745_ivl25_8b_core.json`, `data/s745_s745b_ivl25_8b_core.json`, `data/s745_s747m_q3vl30a3_core.json`, `data/s745_s747m_q3vl32b_core.json`
