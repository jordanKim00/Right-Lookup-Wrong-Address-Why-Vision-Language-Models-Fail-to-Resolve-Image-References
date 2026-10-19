# S609c

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S609 (3): the end reader at the site where the model carries the image index independently of the prompt layout (S608). Fitted on images-first scenes only (procedural training-pool images, N 2-10); checked on held-out scenes in all four layouts. Saved in the format the repair modules load: /vlm/results/s588c_reader_s609_{base}.pt Usage: python s609_reader.py MODEL TAG --base B --layer L --site vend [--ntr 500] [--nte 60]

Code: `code/s609_reader.py`, `code/s609_table.py`

Shared modules: `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/s608_layout.py`, `lib/uni.py`

Data: none released here
