# S717

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S717-cal: R³ frame anchors for other end-relative wordings (frame identification only -- no end-relative answers anywhere). R³-v3 binds an ordinal to an END frame only through the "end" / "last" anchors of exp/r3_build.calibrate ("... from the end / last"); on the S628 wordings "the k-th-to-last image", "the k-th image in reverse order" and "the k-th image counting backwards" it finds no anchor and writes nothing (S716 dev runs). Same procedure as calibrate's anchors: procedural scenes (N 3-10, the calibrate image source), the three calibrate formats (question / look-at / option list), the L_b

Code: `code/s717_analyze.py`, `code/s717_framecal.py`

Shared modules: `lib/r3_build_md.py`, `lib/r3ref.py`, `lib/uni.py`

Data: none of its own (see below)

Its scripts and records are released with the experiments it was run with (the same notebook section):

- `experiments/10_r3_rebinding/S713/code/s713_mmiu.py`
- `experiments/10_r3_rebinding/S717/code/s717_analyze.py`
- `experiments/10_r3_rebinding/S717/code/s717_framecal.py`
- `experiments/10_r3_rebinding/S734/data/s734_35_9b_analysis.json`
- `experiments/10_r3_rebinding/S734/data/s734_8b_analysis.json`
- `experiments/11_benchmarks/S582/data/s582_evalrepair_a7bw_35_9b.json`
- `experiments/11_benchmarks/S582/data/s582_evalrepair_a7dv_35_9b.json`
- `experiments/11_benchmarks/S582/data/s582_evalrepair_a7rv_35_9b.json`
- `experiments/11_benchmarks/S582/data/s582_evalrepair_a7tl_35_9b.json`

Notebook sections: §761 (the notebook itself is not released).
