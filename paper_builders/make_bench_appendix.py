"""Appendix benchmark tables from /vlm/results/r3/s718_analysis.json (latest R³ vs the earlier module, same items).
Writes /vlm/results/figures/bench/{app_main.tex, app_general.tex, app_ref.tex}.  Usage: python figs/make_bench_appendix.py"""
import json, os

R = "/vlm/results"
OUT = f"{R}/figures/bench"
NAMES = {"4b": "Qwen3-VL-4B", "8b": "Qwen3-VL-8B", "35_4b": "Qwen3.5-4B", "35_9b": "Qwen3.5-9B"}
P = lambda x: "--" if x is None else f"{100 * x:.1f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    A = json.load(open(f"{R}/r3/s718_analysis.json"))["res"]
    rows = [("sealed", "mmiu_pos|R", "MMIU first-use"), ("sealed", "muir|R", "MUIR first-use"), ("later", "mmiu_pos|R", "MMIU later"),
            ("cloze", "cloze|R", "MMIU cloze"), ("dev", "mmiu_pos|R", "MMIU development"), ("dev", "muir|R", "MUIR development"),
            ("tolast", "mmiu_pos|R", "MMIU, -to-last"), ("reverse", "mmiu_pos|R", "MMIU, reverse order")]
    L = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Set & Rev. & v1 & Latest & Latest $-$ v1 \\\\", "\\midrule"]
    for b, name in NAMES.items():
        r = A.get(b, {})
        for part, key, lab in rows:
            x = r.get(part, {}).get(key)
            if not x:
                continue
            L.append(f"{name} & {lab} & {P(x['base'])} & {P(x['v1'])} & {P(x['latest'])} & {100 * x['d']:+.1f} [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}] \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"; L.append("\\end{tabular}")
    open(f"{OUT}/app_main.tex", "w").write("\n".join(L) + "\n")
    G = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Subset & $n_R$ & Rev. & v1 & Latest \\\\", "\\midrule"]
    for b, name in NAMES.items():
        g = A.get(b, {}).get("general", {})
        for sub in ("blink", "qbench2", "mantis", "mibench", "mirb"):
            x = g.get(f"{sub}|R")
            if x:
                G.append(f"{name} & {sub} & {x['n']} & {P(x['base'])} & {P(x['v1'])} & {P(x['latest'])} \\\\")
        ch = [v for k, v in list(g.items()) + list(A.get(b, {}).get("reason", {}).items()) if k.endswith("|O")]
        if ch:
            G.append(f"{name} & original forms changed & {sum(v['n'] for v in ch)} & \\multicolumn{{3}}{{r}}{{{sum(round(v['changed'] * v['n']) for v in ch)}}} \\\\")
        G.append("\\midrule")
    G[-1] = "\\bottomrule"; G.append("\\end{tabular}")
    open(f"{OUT}/app_general.tex", "w").write("\n".join(G) + "\n")
    F = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Split & $n$ & Rev. & v1 & Latest \\\\", "\\midrule"]
    for b, name in NAMES.items():
        f = A.get(b, {}).get("ref", {})
        for sp in ("test_iid", "test_oodcomp", "test_oodlength", "test_oodlex", "test_oodnoise", "test_oodpool", "test_oodtemplate"):
            x = f.get(sp)
            if x:
                F.append(f"{name} & {sp.replace('test_', '').replace('ood', 'OOD ')} & {x['n']} & {P(x['base'])} & {P(x['v1'])} & {P(x['latest'])} \\\\")
        F.append("\\midrule")
    F[-1] = "\\bottomrule"; F.append("\\end{tabular}")
    open(f"{OUT}/app_ref.tex", "w").write("\n".join(F) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
