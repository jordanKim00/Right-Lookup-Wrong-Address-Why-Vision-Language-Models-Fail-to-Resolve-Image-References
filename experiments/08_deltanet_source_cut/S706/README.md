# S706

Category: `08_deltanet_source_cut`. Removing image sources from the text: full-attention key masks and the exact segmented Gated DeltaNet cut with the convolution splice.

From the main script's docstring:

> S706: do the anchor heads hand the end of the image sequence to the matcher heads' queries? (sender -> receiver, Cheng et al. style) Pre-registration in knowledge/DECISIONS.md before any S706 data. Qwen3.5-27B, N = 7, new scenes (seed 70600). Head sets fixed on S705a's discovery scenes (results/r3/s705a_qa_27b_matchers.json): ANCHOR (11; 6 of them are not matchers -- PURE) and MATCHER (73). END_k, k in {2, 3}; the last m in {1, 2} images are made invisible to the question's text rows (keys masked) for: FA_ALL every head of every full-attention layer (reference; S703) ANC the ANCHOR heads only 

Code: `code/s706_analyze.py`, `code/s706_anchorpath.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s706_analysis.json`, `data/s706_ap_27b_anchorpath.json`
