# S746

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S746: does reasoning resolve end-relative references, and does the answer follow the address the reasoning states? (VisualSwap's Instruct-vs-Thinking axis; pre-registration in knowledge/DECISIONS.md before any S746 data; analysis exp/s746_analyze.py.) Qwen3-VL Thinking checkpoints, the S730 question form and menu, seven procedural pictures (160 px), seed 74600. OFF reasoning skipped ("<think>\n\n</think>\n\n" prefix), restricted argmax over the menu (as every other S7xx run) ON reasoning on ("<think>\n" prefix, the model's own template), greedy generation up to --max_new tokens; the answer is 

Code: `code/s746_analyze.py`, `code/s746_think.py`, `code/s746b_analyze.py`, `code/s746b_mmiu.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s746_analysis.json`, `data/s746_s746_4bt_think.json`, `data/s746_s746_8bt_think.json`, `data/s746_s788_4bt_think.json`, `data/s746_s788_8bt_think.json`
