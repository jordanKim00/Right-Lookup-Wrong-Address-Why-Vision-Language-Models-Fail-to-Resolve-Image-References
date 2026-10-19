# S703

Category: `08_deltanet_source_cut`. Removing image sources from the text: full-attention key masks and the exact segmented Gated DeltaNet cut with the convolution splice.

From the main script's docstring:

> S703: can the end of the image sequence be moved by hiding the last images from one module only? (the end-moving table) Pre-registration in knowledge/DECISIONS.md before any S703 data. Qwen3.5-27B, seven pictures, END questions. Prior (FINDINGS §596-§599, k = 2): the question reads the end through full-attention keys / values of the last block (L47 71%, L43 + L47 93%; the route into the recurrent state 0), and L51 then selects the block one before. S688: 27B is accurate for k = 2 and k = N, and blurs for k in between. The last m in {1, 2} images are made invisible to the question's text rows (

Code: `code/s703_analyze.py`, `code/s703_moveend.py`

Shared modules: `lib/c_hide.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s703_analysis.json`, `data/s703_me_27b_moveend.json`
