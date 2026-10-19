# S643

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S643: moved anchors in multi-image references -- pointer form (ordinal word vs cardinal offset) x anchor (start / end / an item). Procedural scenes as in exp/s626_patch.py (one-token shape words, distinct shapes per scene), restricted choice over the scene's shape words plus two distractors (logit argmax, as S626); N in 5..7 (only 9 one-token shape words). Ops (m in 1..3): Start "What is in the {ord} image?" t = k End "What is in the {ord} image from the end?" t = N+1-k RelO+/- "What is in the {ord(m)} image after/before the {X}?" t = idx(X) +/- m (ordinal pointer word, item anchor) RelC+/- "W

Re-measured with the clean readout in: S784 (files tagged with the re-run name are kept here).

Code: `code/s643_relimg.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s643_ri2_27b_relimg.json`, `data/s643_ri2_35_4b_relimg.json`, `data/s643_ri2_35_9b_relimg.json`, `data/s643_ri2_4b_relimg.json`, `data/s643_ri2_8b_relimg.json`, `data/s643_rx_35_4b_relimg.json`, `data/s643_rx_35_9b_relimg.json`, `data/s643_rx_4b_relimg.json`, `data/s643_rx_8b_relimg.json`, `data/s643_s784_27b_relimg.json`, `data/s643_s784_35_4b_relimg.json`, `data/s643_s784_35_9b_relimg.json`, `data/s643_s784_4b_relimg.json`, `data/s643_s784_8b_relimg.json`, `data/s643_s784x_35_4b_relimg.json`, `data/s643_s784x_35_9b_relimg.json`, `data/s643_s784x_4b_relimg.json`, `data/s643_s784x_8b_relimg.json`
