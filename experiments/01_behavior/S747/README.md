# S747

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S747a: the behavioural phenomenon in VisualSwap's model set without R³ (pre-registration in knowledge/DECISIONS.md before any S747 data; analysis exp/s747_analyze.py). The paper's evaluator (exp/s582_evalrepair.run) on the later MMIU split (500 items) and MMIU visual cloze (200), with a stand-in for R³ that returns the unedited answer, so "F|base" / "R|base" are the start- and end-relative answers of the unedited model and "|rep" repeats them. Writes results/s582_evalrepair_{TAG}.json in the evaluator's format. Usage: python s747_beh.py MODEL TAG [--device_map balanced]

Code: `code/s747_analyze.py`, `code/s747_beh.py`, `code/s747_beh_adapter.py`, `code/s747_legacy.py`

Shared modules: `lib/uni.py`, `lib/vlm_adapters.py`

Data: `data/s747_analysis.json`
