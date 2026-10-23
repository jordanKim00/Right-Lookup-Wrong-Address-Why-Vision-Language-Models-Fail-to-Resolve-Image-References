# S931

Category: `15_text_only`. The same instruments on text-only word lists: same-scene composition, the END-circuit interventions and the parameter-free prediction.

From the main script's docstring:

> S931 text-only collector - S926's collector (all arms and sub-passes unchanged) on WORD LISTS instead of pictures (user 10-23: can the text-only failure be explained the same way?). Prompt: "There are N words in a row: w1, ..., wN. What is the {k} word from the {last}? Answer with only one word from: {menu}." (START: "... from the first?"); the words are the image experiments' single-token class names, the menu the scene's N words + distractors as before, no image. Blocks (owner) = the list positions of the N words (one token each); read rows, pins (U part of the read-row queries <- START(q)'s

Code: `code/s931_analyze.py`, `code/s931_textcollect.py`, `code/s931_textquad.py`, `code/s931_textss.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s931_analysis_s931n8_27b.json`, `data/s931_s931n8_27b_collect.json`, `data/s931q_s931n8_27b_quad.json`, `data/s931ss_s931ss_27b_samescene.json`
