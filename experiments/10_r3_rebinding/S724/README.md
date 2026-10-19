# S724

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S724 plan-level evaluation (amendment before any S724 data; user direction 2026-10-10: "개별 부품 선택 규칙은 유지하되, 완성된 쓰기 계획의 정확도를 별도의 보정용 검증 자료에서 확인"). No answer is read: every gold plan follows from the item's construction. Gold plan of an input: for every END-bound reference r (ordinal token p_r, ordinal k_r, frame word token f_r, N images), the frame write at f_r and, when j_r = N + 1 - k_r differs from k_r, the ordinal write (p_r -> j_r); nothing for START / unframed references or other tokens. A plan is exact when its ordinal writes (position -> target) and its frame-write positions equal the go

Code: `code/s724_plan.py`, `code/s724_readcal.py`, `code/s724_select.py`, `code/s724_widegrid.py`

Shared modules: `lib/r3_min_md.py`, `lib/r3_repair.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/uni.py`

Data: `data/s724_plan.json`, `data/s724_rc_27b_readcal.json`, `data/s724_rc_35_4b_readcal.json`, `data/s724_rc_35_9b_readcal.json`, `data/s724_rc_4b_readcal.json`, `data/s724_rc_8b_readcal.json`, `data/s724_rc_ivl3_8b_readcal.json`, `data/s724_rc_llov_7b_readcal.json`, `data/s724_rc_q25_7b_readcal.json`, `data/s724_rcs2_4b_readcal.json`, `data/s724_rcs_4b_readcal.json`, `data/s724_rvl8_27b_readcal.json`, `data/s724_rvl8_35_4b_readcal.json`, `data/s724_rvl8_35_9b_readcal.json`, `data/s724_rvl8_4b_readcal.json`, `data/s724_rvl8_8b_readcal.json`, `data/s724_rvl_27b_readcal.json`, `data/s724_rvl_35_4b_readcal.json`, `data/s724_rvl_35_9b_readcal.json`, `data/s724_rvl_4b_readcal.json`, `data/s724_rvl_8b_readcal.json`, `data/s724_rvl_ivl3_8b_readcal.json`, `data/s724_rvl_llov_7b_readcal.json`, `data/s724_rvl_q25_7b_readcal.json`, `data/s724_widegrid.json`
