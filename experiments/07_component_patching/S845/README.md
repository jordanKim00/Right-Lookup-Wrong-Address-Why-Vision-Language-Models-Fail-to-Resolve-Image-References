# S845

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S845 (pre-registered in knowledge/DECISIONS.md before any S845 data; user 10-17): close the writer -> reader path. Writers = the token mixers and MLPs of layers 46-49 (S840: their activation patch removes the late step, .99; largest single parts full-attention 47 and DeltaNet 49). Readers = the late queries (layers 51, 55) of the question-mark, boundary and answer rows. Two stages per scene group: A (one batch): START(k) (donor), END(k), A_BAND (END with the eight band outputs at the read rows replaced by START's, everything downstream sees it) and A_4749 (the same for full-attention 47 and De

Code: `code/s845_analyze.py`, `code/s845_wr.py`

Shared modules: `lib/c_qpath_f.py`, `lib/c_qpath_m.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s845_analysis.json`, `data/s845_s845_27b_wr.json`, `data/s845_scores.json`
