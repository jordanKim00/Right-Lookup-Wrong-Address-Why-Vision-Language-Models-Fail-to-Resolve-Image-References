# S731

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S731: write vectors at the model's rule write layer L_w (S623(b) forward-only consistency), read layer unchanged. Ordinal prototypes and frame anchors at L_w from the same canonical calibration sentences as the read layer (r3_build.calibrate, seed 60400; forward questions and frame-marked texts only). Writes copies the assembled module can load: /vlm/results/r3/s709_{base}w{L}_cal.pt (R3MinMD: the base calibration + Lw + cal_w) /vlm/results/r3/r3s731w{L}_{short}_r3.pt (R3V1C: the earlier module's checkpoint + Lw + cal_w; smaller models only) Usage: python s731_wcal.py MODEL TAG --base B --Lw L

Code: `code/s731_wcal.py`

Shared modules: `lib/r3_build.py`, `lib/r3_build_md.py`, `lib/uni.py`

Data: none of its own (see below)

Its scripts and records are released with the experiments it was run with (the same notebook section):

- `experiments/10_r3_rebinding/S730/code/s730_analyze.py`
- `experiments/10_r3_rebinding/S730/code/s730_stagewise.py`
- `experiments/10_r3_rebinding/S730/data/s730_stw_27b_stagewise.json`
- `experiments/10_r3_rebinding/S730/data/s730_stw_35_4b_stagewise.json`

Notebook sections: §758, §767 (the notebook itself is not released).
