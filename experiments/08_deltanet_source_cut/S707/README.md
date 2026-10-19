# S707

Category: `08_deltanet_source_cut`. Removing image sources from the text: full-attention key masks and the exact segmented Gated DeltaNet cut with the convolution splice.

From the main script's docstring:

> S707: which heads read the end at L43 / L47 -- the matcher heads themselves, or the other heads? Pre-registration in knowledge/DECISIONS.md before any S707 data. Qwen3.5-27B, N = 7, new scenes (seed 70700). S706: hiding the last images from the 11 last-block-reading heads does not move the end (.14), hiding them from every head at L43 + L47 does (S703 .60 / .55). MATCHER heads (73) fixed on S705a's discovery scenes. END_k, k in {2, 3}; the last m in {1, 2} images are made invisible to the question's text rows (keys masked) for: ALL_4347 every head at L43 and L47 (reference) MAT_4347 the MATCHE

Code: `code/s707_analyze.py`, `code/s707_endreaders.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s707_analysis.json`, `data/s707_er_27b_endreaders.json`
