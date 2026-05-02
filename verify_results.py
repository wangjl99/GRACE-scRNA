"""
docs/verify_results.py
=======================
Verify that computed results match the paper's reported numbers.
Run after the full pipeline to confirm reproducibility.

Usage:
    python docs/verify_results.py
"""

import json
import pandas as pd
from pathlib import Path

TOLERANCE = 0.005  # Allow ±0.5% for floating point

EXPECTED = {
    "luad_grace_weighted":    100.0,
    "luad_grace_macro":       100.0,
    "luad_gpt_weighted":       85.7,
    "luad_gpt_macro":          80.0,
    "luad_singleR_weighted":   91.1,
    "luad_singleR_macro":      90.0,
    "hcc_grace_weighted":      93.3,
    "hcc_grace_macro":         92.0,
    "hcc_gpt_weighted":        43.9,
    "hcc_singleR_weighted":    80.4,
    "go_f1_grace":              0.689,
    "go_prec_grace":            0.601,
    "go_f1_gpt":                0.572,
    "calibration_gap_grace":   0.132,
    "mean_conf_luad":          0.55,
    "mean_conf_hcc":           0.42,
}

def check(name, actual, expected, tol=TOLERANCE):
    ok = abs(actual - expected) <= tol
    status = "✓" if ok else "✗"
    print(f"  {status}  {name}: {actual:.3f}  (expected {expected:.3f})")
    return ok

passed = 0
failed = 0

print("=" * 60)
print("GRACE — Results verification")
print("=" * 60)

# SingleR results
sr_path = Path("results/singleR_summary.json")
if sr_path.exists():
    sr = json.load(open(sr_path))
    print("\nSingleR accuracy:")
    for k, exp_key in [("luad_singleR_weighted", ("luad","weighted")),
                       ("luad_singleR_macro",    ("luad","macro")),
                       ("hcc_singleR_weighted",  ("hcc","weighted"))]:
        v = sr
        for part in exp_key: v = v[part]
        r = check(k, v, EXPECTED[k])
        passed += r; failed += (1-r)
else:
    print("\n  [SKIP] results/singleR_summary.json not found")
    print("         Run: python evaluation/run_singleR_python.py")

# Table 1 (LUAD metrics)
t1_path = Path("results/table1_definitive_final.csv")
if t1_path.exists():
    t1 = pd.read_csv(t1_path)
    print("\nGO-term metrics:")
    grace_row = t1[t1["Method"].str.contains("GRACE v2", na=False)]
    gpt_row   = t1[t1["Method"].str.contains("GPT", na=False)]
    if not grace_row.empty:
        go_f1  = float(grace_row["GO-F1"].iloc[0])
        go_prec= float(grace_row["GO-Prec"].iloc[0])
        r1 = check("go_f1_grace",  go_f1,  EXPECTED["go_f1_grace"])
        r2 = check("go_prec_grace",go_prec,EXPECTED["go_prec_grace"])
        passed += r1+r2; failed += (1-r1)+(1-r2)
    if not gpt_row.empty:
        go_f1_gpt = float(gpt_row["GO-F1"].iloc[0])
        r = check("go_f1_gpt", go_f1_gpt, EXPECTED["go_f1_gpt"])
        passed += r; failed += (1-r)
else:
    print("\n  [SKIP] results/table1_definitive_final.csv not found")

# Table 2 (cross-cancer accuracy)
t2_path = Path("results/table2_definitive_final.csv")
if t2_path.exists():
    t2 = pd.read_csv(t2_path)
    print("\nCross-cancer accuracy:")
    for ds, method, key in [
        ("LUAD","GRACE","luad_grace_weighted"),
        ("LUAD","GPT",  "luad_gpt_weighted"),
        ("HCC", "GRACE","hcc_grace_weighted"),
        ("HCC", "GPT",  "hcc_gpt_weighted"),
    ]:
        row = t2[(t2["Dataset"].str.contains(ds)) &
                 (t2["Method"].str.contains(method))]
        if not row.empty:
            v = float(row["Weighted"].iloc[0].replace("%",""))
            r = check(key, v, EXPECTED[key])
            passed += r; failed += (1-r)
else:
    print("\n  [SKIP] results/table2_definitive_final.csv not found")

# Calibration
cal_path = Path("results/calibration_results.json")
if cal_path.exists():
    cal = json.load(open(cal_path))
    print("\nCalibration:")
    gap = abs(float(cal.get("calibration_gap", 0)))
    r = check("calibration_gap_grace", gap, EXPECTED["calibration_gap_grace"])
    passed += r; failed += (1-r)
else:
    print("\n  [SKIP] results/calibration_results.json not found")

# Mean confidence
vb_path = Path("results/versionB_results.json")
if vb_path.exists():
    import json as _json
    vb = _json.load(open(vb_path))
    confs = [r.get("orchestration",{}).get("overall_confidence",0) for r in vb]
    if confs:
        mean_c = sum(confs)/len(confs)
        r = check("mean_conf_luad", mean_c, EXPECTED["mean_conf_luad"])
        passed += r; failed += (1-r)

hcc_vb_path = Path("results/hcc/hcc_versionB_results.json")
if hcc_vb_path.exists():
    vb_h = _json.load(open(hcc_vb_path))
    confs_h = [r.get("orchestration",{}).get("overall_confidence",0) for r in vb_h]
    if confs_h:
        mean_h = sum(confs_h)/len(confs_h)
        r = check("mean_conf_hcc", mean_h, EXPECTED["mean_conf_hcc"])
        passed += r; failed += (1-r)

print()
print("=" * 60)
total = passed + failed
if failed == 0:
    print(f"  All {total} checks passed ✓")
else:
    print(f"  {passed}/{total} checks passed  ({failed} failed ✗)")
    print("  See docs/REPRODUCIBILITY.md for troubleshooting.")
print("=" * 60)
