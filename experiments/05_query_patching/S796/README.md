# S796

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S796 (pre-registered in knowledge/DECISIONS.md before any S796 data): does the frame read reach the answer through the lookup queries? Frame -> query -> answer mediation on the END(4) / END(5) instrument of S769, with content swaps for the retrieval step. Qwen3.5 only (gated full attention recomputed by exp/c_qpath.py; complete-route read interchange of the frame word as exp/s769_end4.py, without its convolution patch). Scenes: procedural pictures (160 px), N = 7, seed 79600 (new), 40 scenes; q in {4, 5}. END(q) = "the q-th image from the last" (target N + 1 - q: END(4) targets the literal ima

Code: `code/s796_analyze.py`, `code/s796_mediate.py`

Shared modules: `lib/c_qpath.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s796_analysis.json`, `data/s796_s796_27b_mediate.json`
