#!/usr/bin/env python3
"""
Compare @tangent.to/sem against semopy (which itself reproduces lavaan)
on the two canonical SEM benchmarks:

  1. Holzinger-Swineford 1939 three-factor CFA
  2. Bollen's political democracy full SEM (latent regressions and
     correlated indicator residuals)

Estimates and standard errors are matched parameter-by-parameter.
Run from the package root:

    uv run --with semopy python3 tests_compare-to-semopy/compare_with_semopy.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import semopy
from semopy.examples import holzinger39, political_democracy

ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = ROOT / "tests_compare-to-semopy" / "compare_sem.mjs"

FAILURES = []


def check(label, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    if not ok:
        FAILURES.append(label)


def run_node(syntax, df):
    spec = {"syntax": syntax, "data": {c: df[c].tolist() for c in df.columns}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    r = subprocess.run(["node", str(NODE_SCRIPT), path], check=True,
                       capture_output=True, text=True, cwd=ROOT)
    return json.loads(r.stdout)


def semopy_rows(model):
    """Normalize semopy's inspect() to (lhs, op, rhs) -> (est, se) keyed like ours."""
    rows = {}
    insp = model.inspect()
    for _, r in insp.iterrows():
        lval, op, rval = r["lval"], r["op"], r["rval"]
        est, se = float(r["Estimate"]), r["Std. Err"]
        se = float(se) if se == se and se != "-" else None
        if op == "~":
            # semopy writes loadings as indicator ~ factor; regressions as y ~ x
            key = ("=~", rval, lval)  # try measurement orientation first
            rows[key] = (est, se)
            rows[("~", lval, rval)] = (est, se)
        elif op == "~~":
            key = ("~~",) + tuple(sorted([lval, rval]))
            rows[key] = (est, se)
    return rows


def compare(label, syntax, df, est_tol, se_tol):
    js = run_node(syntax, df)
    m = semopy.Model(syntax)
    m.fit(df)
    ref = semopy_rows(m)

    worst_est = 0.0
    worst_se = 0.0
    matched = 0
    for e in js["estimates"]:
        if not e["free"]:
            continue
        if e["op"] == "~~":
            key = ("~~",) + tuple(sorted([e["lhs"], e["rhs"]]))
        elif e["op"] == "=~":
            key = ("=~", e["lhs"], e["rhs"])
        else:
            key = ("~", e["lhs"], e["rhs"])
        if key not in ref:
            continue
        est_ref, se_ref = ref[key]
        matched += 1
        # relative differences: variances can be large-scale, and semopy's
        # convergence is slightly looser than ours (we sit marginally below
        # semopy's objective on political democracy, matching lavaan's
        # published table more closely)
        worst_est = max(worst_est, abs(e["est"] - est_ref) / max(1, abs(est_ref)))
        if e["se"] is not None and se_ref is not None:
            worst_se = max(worst_se, abs(e["se"] - se_ref) / max(0.05, abs(se_ref)))

    check(f"{label}: estimates ({matched} matched)", worst_est < est_tol,
          f"max rel est diff = {worst_est:.2e}")
    check(f"{label}: standard errors", worst_se < se_tol,
          f"max rel se diff = {worst_se:.2e}")
    check(f"{label}: converged", js["converged"], f"chisq={js['chisq']:.3f} df={js['df']}")
    return js


def main():
    print("semopy comparison for @tangent.to/sem")

    # --- Holzinger-Swineford CFA ---
    hs = holzinger39.get_data()[[f"x{i}" for i in range(1, 10)]]
    syntax_hs = """
      visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9
    """
    js = compare("HS CFA", syntax_hs, hs, est_tol=2e-3, se_tol=5e-3)
    # lavaan's published chi-square for this model
    check("HS CFA: chisq vs lavaan (85.306)", abs(js["chisq"] - 85.306) < 0.05,
          f"chisq = {js['chisq']:.3f}")
    check("HS CFA: fit indices vs lavaan", abs(js["cfi"] - 0.931) < 0.005 and
          abs(js["rmsea"] - 0.092) < 0.003 and abs(js["srmr"] - 0.065) < 0.003,
          f"cfi={js['cfi']:.3f} rmsea={js['rmsea']:.3f} srmr={js['srmr']:.3f}")

    # --- Bollen political democracy: full SEM ---
    pd_data = political_democracy.get_data()
    syntax_pd = """
      ind60 =~ x1 + x2 + x3
      dem60 =~ y1 + y2 + y3 + y4
      dem65 =~ y5 + y6 + y7 + y8
      dem60 ~ ind60
      dem65 ~ ind60 + dem60
      y1 ~~ y5
      y2 ~~ y4
      y2 ~~ y6
      y3 ~~ y7
      y4 ~~ y8
      y6 ~~ y8
    """
    compare("Political democracy SEM", syntax_pd, pd_data, est_tol=5e-3, se_tol=1e-2)

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nAll comparisons passed.")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
