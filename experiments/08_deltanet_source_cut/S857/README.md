# S857

Category: `08_deltanet_source_cut`. Removing image sources from the text: full-attention key masks and the exact segmented Gated DeltaNet cut with the convolution splice.

From the main script's docstring:

> S857 (P2, user 10-18 plan; pre-registered in knowledge/DECISIONS.md before any S857 data). How do pictures 7-8 reach the late query? Same scenes and prompt as S856 form A ('There are 8 images in a row.'), 8 pictures (X8) or the first 6 (X6), k {3, 5}, early lookup pinned to p {3, 4} (address part U over the shared pictures 1-6 <- START(p) of the same configuration, layers 35-47). The source = pictures 7-8 (picture 7's vision-start .. picture 8's vision-end); cuts remove it from every text row after it (exp/c_cut.py: full-attention mask, DeltaNet state segmenting, convolution splice), per layer

Code: `code/s857_analyze.py`, `code/s857_cut.py`

Shared modules: `lib/c_cut.py`, `lib/c_qpath_j.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s857_analysis.json`, `data/s857_s857_27b_cut.json`
