# S741

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S741: is the index code a low-dimensional component of the image-block states, separable from content? (reply to the mock review W2, 2026-10-11; pre-registration in knowledge/DECISIONS.md before any S741 data; analysis exp/s741_analyze.py, frozen with it.) The S545 code shift is a full-state exchange: every image-block token (with its vision markers) of the recipient takes, at every layer, the residual state of the same picture in a donor sequence with two new pictures in front, so blocks 1..N carry codes 3..N+2 together with whatever else the donor context changed. Here the same exchange is s

Code: `code/s741_analyze.py`, `code/s741_codesub.py`

Shared modules: `lib/d1_grammar.py`, `lib/uni.py`

Data: `data/s741_analysis.json`, `data/s741_s741_4b_codesub.json`, `data/s741_s741_9b_codesub.json`
