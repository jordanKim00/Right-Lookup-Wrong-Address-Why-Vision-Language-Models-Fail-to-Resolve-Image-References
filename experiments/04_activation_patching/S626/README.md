# S626

Category: `04_activation_patching`. Residual-stream transplants and writes at the ordinal token and after it (ordinal state as a position, written addresses).

From the main script's docstring:

> S626 (1)-(2): full token x layer maps of the reverse-ordinal computation (exp/QUEUE.md S626, amended design). Fixed example: 7 PROC images (160 px, distinct single-token shapes), "What is in the third image from the end? Answer with only one word from: <7 scene shapes + 2 new shapes>" (target block 5, surface block 3), 24 scenes. Counterfactual sources, token-aligned: A block 5 (target) replaced by a new shape metric logit(new) - logit(old block-5 shape) B block 3 (surface) replaced by a new shape metric logit(new) - logit(old block-3 shape) C "third" -> "second" (target 6, surface 2) metrics 

Code: `code/s626_atp.py`, `code/s626_dasmap.py`, `code/s626_flow.py`, `code/s626_pack.py`, `code/s626_patch.py`, `code/s626_timeline.py`

Shared modules: `lib/r3_proc.py`, `lib/r3_repair.py`, `lib/r3_select.py`, `lib/r3ref.py`, `lib/uni.py`

Data: `data/s626_atp2_35_4b_atp.json`, `data/s626_atp2_35_9b_atp.json`, `data/s626_atp_27b_atp.json`, `data/s626_atp_4b_atp.json`, `data/s626_atp_4b_r3_atp.json`, `data/s626_atp_8b_atp.json`, `data/s626_atpseg_35_4b_atp.json`, `data/s626_atpseg_35_4b_r3_atp.json`, `data/s626_atpseg_35_9b_r3_atp.json`, `data/s626_first25.json`, `data/s626_flow.json`, `data/s626_p726_35_4b_patch.json`, `data/s626_patch2_35_9b_patch.json`, `data/s626_patch_27b_patch.json`, `data/s626_patch_35_4b_patch.json`, `data/s626_patch_4b_patch.json`, `data/s626_patch_4b_r3_patch.json`, `data/s626_patch_8b_patch.json`, `data/s626_patchc_27b_patch.json`, `data/s626_timeline.json`
