# S826

Category: `06_read_interchange`. Route-complete read interchanges: later tokens read one token as in a donor question, through full attention, the DeltaNet state and its convolution.

From the main script's docstring:

> S826 (pre-registered in knowledge/DECISIONS.md before any S826 data): where is the stated count read, and which rows hold it? FINDINGS §916 (S824): the end-relative answer follows the stated count ('There are N images in a row.') more than the pictures shown - with 8 pictures, END(7) answers the first picture when told '7' (.85) and not when told '8' (0). Minimal pair (the S815 design with the count token in place of the frame word): recipient END(k) told 8, donor END(k) told 7, the same 8 pictures; the two prompts differ only at the count token, so a state swap at it is exact from that layer 

Code: `code/s826_analyze.py`, `code/s826_count_window.py`

Shared modules: `lib/c_qpath_b.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s826_analysis.json`, `data/s826_s826_27b_count.json`
