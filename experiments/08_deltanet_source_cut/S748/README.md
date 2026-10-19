# S748

Category: `08_deltanet_source_cut`. Removing image sources from the text: full-attention key masks and the exact segmented Gated DeltaNet cut with the convolution splice.

From the main script's docstring:

> S748: is the end that Qwen3.5-27B counts back from the last block's code or the number of visible images? Pre-registration in knowledge/DECISIONS.md (2026-10-12 19:05, amended 19:4x before any data); analysis exp/s748_analyze.py. Seven procedural pictures (160 px), the S703 prompt ("What is in the {ordinal} image from the last?"), menu = 7 labels + 2. Image blocks are made invisible to the question's text rows (every token after the last image) through the full-attention keys at every full-attention layer (exp/c_tools.FullAttn, edit "ko"); image rows and the DeltaNet route are untouched. NONE 

Code: `code/s748_analyze.py`, `code/s748_whatend.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s748_analysis.json`, `data/s748_s748_27b_whatend.json`, `data/s748_sm_s748_whatend.json`
