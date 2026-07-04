import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { lstsq } from '@tangent.to/lina';
import { createRng, normal } from '@tangent.to/proba';
import { sem } from '../src/index.js';

function loadHS() {
  const lines = readFileSync(new URL('../data/holzinger39.csv', import.meta.url), 'utf8')
    .trim().split('\n');
  const header = lines[0].split(',');
  return lines.slice(1).map((l) => {
    const c = l.split(',');
    const o = {};
    header.forEach((h, i) => o[h] = +c[i]);
    return o;
  });
}

const HS_MODEL = `
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
`;

describe('Holzinger-Swineford benchmark (lavaan published values)', () => {
  const fit = sem(HS_MODEL, { data: loadHS() });

  it('converges and reproduces the chi-square', () => {
    expect(fit.converged).toBe(true);
    expect(fit.fit.chisq).toBeCloseTo(85.306, 2);
    expect(fit.fit.df).toBe(24);
  });

  it('reproduces the published loadings', () => {
    const est = (lhs, rhs) =>
      fit.estimates.find((e) => e.lhs === lhs && e.op === '=~' && e.rhs === rhs).est;
    expect(est('visual', 'x2')).toBeCloseTo(0.554, 3);
    expect(est('visual', 'x3')).toBeCloseTo(0.729, 3);
    expect(est('textual', 'x5')).toBeCloseTo(1.113, 3);
    expect(est('textual', 'x6')).toBeCloseTo(0.926, 3);
    expect(est('speed', 'x8')).toBeCloseTo(1.180, 3);
    expect(est('speed', 'x9')).toBeCloseTo(1.082, 3);
  });

  it('reproduces the published standard errors (expected information)', () => {
    const se = (lhs, rhs) =>
      fit.estimates.find((e) => e.lhs === lhs && e.op === '=~' && e.rhs === rhs).se;
    expect(se('visual', 'x2')).toBeCloseTo(0.100, 3);
    expect(se('textual', 'x5')).toBeCloseTo(0.065, 3);
  });

  it('reproduces the published fit indices', () => {
    expect(fit.fit.cfi).toBeCloseTo(0.931, 3);
    expect(fit.fit.tli).toBeCloseTo(0.896, 3);
    expect(fit.fit.rmsea).toBeCloseTo(0.092, 3);
    expect(fit.fit.srmr).toBeCloseTo(0.065, 3);
    expect(fit.fit.aic).toBeCloseTo(7517.49, 1);
    expect(fit.fit.bic).toBeCloseTo(7595.34, 1);
  });

  it('marker loadings are fixed to 1 and have no se', () => {
    const marker = fit.estimates.find((e) => e.lhs === 'visual' && e.rhs === 'x1');
    expect(marker.est).toBe(1);
    expect(marker.free).toBe(false);
    expect(marker.se).toBeNull();
  });

  it('summary() renders a readable table', () => {
    const s = fit.summary();
    expect(s).toContain('chisq = 85.30');
    expect(s).toContain('visual');
    expect(s).toContain('=~');
  });
});

describe('parameter recovery on simulated data', () => {
  it('recovers a 2-factor structure', () => {
    const rng = createRng(2026);
    const n = 2000;
    const data = [];
    for (let i = 0; i < n; i++) {
      const f1 = rng.normal();
      const f2 = 0.5 * f1 + Math.sqrt(1 - 0.25) * rng.normal(); // corr = 0.5
      data.push({
        a1: f1 + 0.5 * rng.normal(),
        a2: 0.8 * f1 + 0.5 * rng.normal(),
        a3: 0.6 * f1 + 0.5 * rng.normal(),
        b1: f2 + 0.5 * rng.normal(),
        b2: 0.7 * f2 + 0.5 * rng.normal(),
        b3: 0.9 * f2 + 0.5 * rng.normal(),
      });
    }
    const fit = sem(`
      fa =~ a1 + a2 + a3
      fb =~ b1 + b2 + b3
    `, { data });
    expect(fit.converged).toBe(true);
    const est = (l, r) => fit.estimates.find((e) => e.lhs === l && e.rhs === r).est;
    expect(est('fa', 'a2')).toBeCloseTo(0.8, 1);
    expect(est('fa', 'a3')).toBeCloseTo(0.6, 1);
    expect(est('fb', 'b2')).toBeCloseTo(0.7, 1);
    expect(est('fb', 'b3')).toBeCloseTo(0.9, 1);
    // factor covariance ~ 0.5 (both factor variances ~1)
    expect(est('fa', 'fb')).toBeCloseTo(0.5, 1);
    // good fit: the model is true
    expect(fit.fit.rmsea).toBeLessThan(0.03);
  });
});

describe('path analysis', () => {
  it('matches OLS for a single-equation model', () => {
    const rng = createRng(7);
    const n = 500;
    const data = [];
    for (let i = 0; i < n; i++) {
      const x1 = rng.normal();
      const x2 = 0.3 * x1 + rng.normal();
      data.push({ x1, x2, y: 2 * x1 - 1.5 * x2 + normal.sample({ mu: 0, sigma: 0.8 }, rng) });
    }
    const fit = sem('y ~ x1 + x2', { data });

    // Centered OLS via lina (sem is covariance-based, so compare centered fits)
    const mx1 = data.reduce((s, d) => s + d.x1, 0) / n;
    const mx2 = data.reduce((s, d) => s + d.x2, 0) / n;
    const my = data.reduce((s, d) => s + d.y, 0) / n;
    const X = data.map((d) => [d.x1 - mx1, d.x2 - mx2]);
    const yv = data.map((d) => d.y - my);
    const { x: ols } = lstsq(X, yv);

    const b = (rhs) => fit.estimates.find((e) => e.op === '~' && e.rhs === rhs).est;
    expect(b('x1')).toBeCloseTo(ols[0], 6);
    expect(b('x2')).toBeCloseTo(ols[1], 6);
    expect(fit.fit.df).toBe(0); // saturated
  });
});

describe('covariance-matrix input and errors', () => {
  it('accepts cov + n + names', () => {
    const data = loadHS();
    const fitData = sem(HS_MODEL, { data });
    const fitCov = sem(HS_MODEL, {
      cov: fitData.S,
      n: 301,
      names: fitData.observed,
    });
    expect(fitCov.fit.chisq).toBeCloseTo(fitData.fit.chisq, 6);
  });

  it('throws on unknown variables, bad specs and unidentified models', () => {
    const data = loadHS();
    expect(() => sem('f =~ x1 + nope', { data })).toThrow(/not in the data/);
    expect(() => sem('', { data })).toThrow(/non-empty/);
    expect(() => sem('f =~ x1 + x2', {})).toThrow(/spec\.data or spec\.cov/);
    expect(() => sem('f =~ x1 + x2', { cov: [[1, 0.5], [0.5, 1]] })).toThrow(/names and spec\.n/);
    // 2 indicators, 1 factor: 3 moments, 4 free params (loading freed) -> unidentified
    expect(() => sem('f =~ NA*x1 + x2', {
      cov: [[1, 0.5], [0.5, 1]], n: 100, names: ['x1', 'x2'],
    })).toThrow(/not identified/);
  });
});
