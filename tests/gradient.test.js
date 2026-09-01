/**
 * The exact gradient of the ML discrepancy, and the estimates it produces.
 *
 * `discrepancyAndGrad` replaces @tangent.to/opt's finite-difference fallback
 * inside `estimate`. The bar is that it changes nothing observable — the same
 * estimates, to the same digits, as the lavaan/semopy-validated references in
 * sem.test.js — while costing a fraction of the evaluations.
 */

import { describe, expect, it } from 'vitest';
import { cholesky } from '@tangent.to/lina';
import { lbfgs } from '@tangent.to/opt';
import { buildModel, setSampleStarts } from '../src/model.js';
import { parseModel } from '../src/parse.js';
import { discrepancy, discrepancyAndGrad, sampleCov } from '../src/fit.js';
import { sem } from '../src/index.js';

/** Deterministic two-factor CFA data. */
function fixture(n = 400) {
  let s = 7;
  const u = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const randn = () => Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u());
  const names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'];
  const data = [];
  for (let i = 0; i < n; i++) {
    const f1 = randn();
    const f2 = 0.5 * f1 + Math.sqrt(0.75) * randn();
    data.push({
      x1: 1.0 * f1 + 0.6 * randn(), x2: 0.9 * f1 + 0.6 * randn(), x3: 0.8 * f1 + 0.6 * randn(),
      x4: 1.0 * f2 + 0.6 * randn(), x5: 0.9 * f2 + 0.6 * randn(), x6: 0.8 * f2 + 0.6 * randn(),
    });
  }
  const spec = 'f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf1 ~~ f2';
  const model = buildModel(parseModel(spec), names);
  const { S } = sampleCov(data, names);
  const order = model.observed.map((v) => names.indexOf(v));
  const Sobs = order.map((i) => order.map((j) => S[i][j]));
  setSampleStarts(model, Sobs);
  const L = cholesky(Sobs);
  const logDetS = 2 * L.reduce((a, r, i) => a + Math.log(r[i]), 0);
  return { spec, data, model, Sobs, logDetS };
}

describe('discrepancyAndGrad', () => {
  const { spec, data, model, Sobs, logDetS } = fixture();
  const start = model.params.filter((p) => p.free).map((p) => p.start);

  it('returns the same value as the scalar discrepancy', () => {
    expect(discrepancyAndGrad(model, start, Sobs, logDetS).loss)
      .toBeCloseTo(discrepancy(model, start, Sobs, logDetS), 12);
  });

  it('matches central differences of the scalar discrepancy', () => {
    const { gradient } = discrepancyAndGrad(model, start, Sobs, logDetS);
    const fd = start.map((_, k) => {
      const h = 1e-6 * Math.max(1, Math.abs(start[k]));
      const a = start.slice(); a[k] += h;
      const b = start.slice(); b[k] -= h;
      return (discrepancy(model, a, Sobs, logDetS) - discrepancy(model, b, Sobs, logDetS)) / (2 * h);
    });
    gradient.forEach((g, i) => expect(g).toBeCloseTo(fd[i], 6));
  });

  it('is near zero at the optimum, to a tolerance finite differences cannot reach', () => {
    const theta = sem(spec, { data }).theta;
    const { gradient } = discrepancyAndGrad(model, theta, Sobs, logDetS);
    expect(Math.max(...gradient.map(Math.abs))).toBeLessThan(1e-5);
  });

  it('reports the penalty with a zero gradient when Sigma leaves the PD cone', () => {
    // The line search must back off, not follow a meaningless direction.
    const bad = start.map(() => -50);
    const out = discrepancyAndGrad(model, bad, Sobs, logDetS);
    expect(out.loss).toBe(1e10);
    expect(out.gradient.every((g) => g === 0)).toBe(true);
  });

  it('lands on the same optimum as the finite-difference objective', () => {
    // The guarantee that matters: swapping in an exact gradient must not move
    // the answer. Both objectives are handed to the same optimizer from the
    // same start, and compared parameter by parameter.
    const opts = { maxIter: 5000, tol: 1e-9 };
    const fd = lbfgs((t) => discrepancy(model, t, Sobs, logDetS), start, opts);
    const ad = lbfgs((t) => discrepancyAndGrad(model, t, Sobs, logDetS), start, opts);

    expect(ad.iterations).toBe(fd.iterations);
    expect(ad.fx).toBeCloseTo(fd.fx, 10);
    ad.x.forEach((v, i) => expect(v).toBeCloseTo(fd.x[i], 6));
  });

  it('costs far fewer objective evaluations than the fallback', () => {
    const opts = { maxIter: 5000, tol: 1e-9 };
    let fdCalls = 0;
    lbfgs((t) => { fdCalls++; return discrepancy(model, t, Sobs, logDetS); }, start, opts);
    let adCalls = 0;
    lbfgs((t) => { adCalls++; return discrepancyAndGrad(model, t, Sobs, logDetS); }, start, opts);
    // The fallback spends 2*q extra evaluations per gradient; q is 13 here.
    expect(adCalls * 10).toBeLessThan(fdCalls);
  });

  it('still produces usable estimates and standard errors', () => {
    const fit = sem(spec, { data });
    expect(fit.converged).toBe(true);
    const l2 = fit.estimates.find((e) => e.op === '=~' && e.rhs === 'x2');
    // Sampling error at n=400 is around 0.04, so this is a sanity band, not a
    // recovery claim: the lavaan-validated references live in sem.test.js.
    expect(l2.est).toBeGreaterThan(0.6);
    expect(l2.est).toBeLessThan(1.2);
    expect(l2.se).toBeGreaterThan(0);
    expect(Number.isFinite(l2.z)).toBe(true);
  });
});
