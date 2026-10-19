# S827

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S827 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S827 data). Cases N7 (k = 4) and N9 (k = 5), identity references: j = k. For an arm: dP(x) = P(arm -> x) - P(recipient natural -> x); neighbour mass nb = P(j - 1) + P(j + 1); logit margin lm = logit(j) - max(logit(j - 1), logit(j + 1)) (two-form menu logits); internal margin im(row, layer) = block score(j) - max(block score(j - 1), block score(j + 1)) (index-matching heads). Scene bootstrap, 4,000 draws, seed 0. I0 t1 >= .95; NOOP_E lands as END and NOOP_S as START in >= .95 of scenes per case, 95th percentile o

Code: `code/s827_analyze.py`, `code/s827_analyze_draft1.py`, `code/s827_heads_desc.py`, `code/s827_identity.py`, `code/s827_identity_draft1.py`, `code/s827_post_tworoute.py`, `code/s827_pre_landings.py`, `code/s827_predict.py`

Shared modules: `lib/c_qpath_b.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s827_analysis.json`, `data/s827_heads_desc_s827_27b.json`, `data/s827_predictions_s827_27b.json`, `data/s827_s827_27b_identity.json`, `data/s827_s827_27b_natural.json`
