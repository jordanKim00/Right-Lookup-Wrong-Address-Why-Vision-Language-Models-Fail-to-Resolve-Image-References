# S829

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S829 (pre-registered in knowledge/DECISIONS.md before any S829 data): is the END error a conjunction of a wrong address and an END reading mode (S828 §922: restoring either the two-route address or the non-address query part repairs the identity failure; START resists END's address)? Hypothesis (S828 gated outputs): START reads the copy early (L35-47) and reads pictures little late; END suppresses the early copy and re-reads pictures heavily late (L51-55) with an imprecise end-relative address. Tests: END's whole query breaks START while its non-address part alone does not; the picture-vs-text

Code: `code/s829_analyze.py`, `code/s829_mode.py`, `code/s829_predict.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s829_analysis.json`, `data/s829_analysis_STALE_QPATH.json`, `data/s829_s829_27b_mode.json`, `data/s829_s829_27b_mode_STALE_QPATH.json`, `data/s829_s829_27b_natural.json`
