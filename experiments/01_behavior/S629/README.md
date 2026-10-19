# S629

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S629: the Position Curse protocol (Zhang et al. 2026, arXiv 2605.07127, Sec. 3) on our models, text only, item retrieval. A list of L distinct items (uppercase letters, or one-token animal names) and one query, with the four operations of the paper: End+ "What item is at the {k}th position from the beginning?" target = item k End- "What item is at the {k}th position from the end?" target = item L+1-k Rel+ "What item is {m} positions after {X}?" target = item idx(X)+m Rel- "What item is {m} positions before {X}?" target = item idx(X)-m Greedy free generation (thinking off), first item-like toke

Code: `code/s629_poscurse.py`

Shared modules: `lib/uni.py`

Data: `data/s629_pc_27b_poscurse.json`, `data/s629_pc_35_4b_poscurse.json`, `data/s629_pc_35_9b_poscurse.json`, `data/s629_pc_4b_poscurse.json`, `data/s629_pc_8b_poscurse.json`, `data/s629_pc_q25_7b_poscurse.json`, `data/s629_pcb_27b_poscurse.json`, `data/s629_pcb_35_4b_poscurse.json`, `data/s629_pcb_35_9b_poscurse.json`, `data/s629_pcb_4b_poscurse.json`, `data/s629_pcb_8b_poscurse.json`, `data/s629_pcb_q25_7b_poscurse.json`
