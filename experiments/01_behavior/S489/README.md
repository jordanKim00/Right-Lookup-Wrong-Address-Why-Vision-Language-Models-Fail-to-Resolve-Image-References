# S489

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S489: the mechanism-derived fix on a natural benchmark (MMIU positional tasks, the S462 items). S487 (§499): computing the index in the question-first (QIQ) layout, writing it, and reading by the written index lifts synthetic reverse retrieval .26-.35 -> .96-.99, exactly as the multiplicative law predicts. Here the same fix is applied to the twelve MMIU tasks whose options name images by forward ordinals; the options are rewritten as reverse ordinals ("The m-th image from the end") as in S462. Arms on the same items: F original forward ordinals E explicit "Image k" (reference ceiling) R revers

Code: `code/s489_mmiu_fix.py`

Shared modules: `lib/uni.py`

Data: `data/s489_mmiu_fix_489a27b.json`, `data/s489_mmiu_fix_489a4b.json`, `data/s489_mmiu_fix_489a8b.json`, `data/s489_mmiu_fix_489a9b.json`, `data/s489_mmiu_fix_489aq4b.json`, `data/s489_mmiu_fix_489smk.json`
