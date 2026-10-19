# S648

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S648: question-form vs option-form end-relative references on the same procedural scenes. 27B composes end-relative references in question form (S626/§576/§597) but fails MMIU/MUIR option lines (base .262/.329, §535 §540). QF "What is in the {t}th image?" content answer (scene shapes + 2 distractors) QR "What is in the {N+1-t}th image from the end?" content answer OF "Which image shows the {X}? A: The {p}th image B: ..." letter answer, 4 options, X = image t OR same, options "The {N+1-p}th image from the end" letter answer Options cover four distinct physical positions: the target t, the lure 

Code: `code/s648_format.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s648_fmt_27b_format.json`, `data/s648_fmt_35_4b_format.json`, `data/s648_fmt_35_9b_format.json`, `data/s648_fmt_4b_format.json`, `data/s648_fmt_8b_format.json`
