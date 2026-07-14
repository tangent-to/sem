/**
 * @tangent.to/sem - Structural equation modeling for JavaScript (ESM)
 *
 * CFA, path analysis and full SEM with lavaan-style model syntax,
 * maximum-likelihood estimation on the tangent suite's numeric leaves
 * (lina for the linear algebra, opt for L-BFGS, proba for the test
 * distributions). Validated against semopy/lavaan.
 */

import { parseModel } from './parse.js';
import { buildModel, setSampleStarts } from './model.js';
import { estimate, sampleCov } from './fit.js';

/**
 * Fit a structural equation model.
 *
 * @param {string} syntax - lavaan-style model syntax:
 *   `factor =~ ind1 + ind2` (measurement), `y ~ x1 + x2` (regression),
 *   `a ~~ b` ((co)variance), `1*x` (fix), `NA*x` (free a default-fixed
 *   parameter)
 * @param {Object} spec
 * @param {Array<Object>} [spec.data] - Rows as objects (column per variable)
 * @param {Array<Array<number>>} [spec.cov] - Sample covariance (instead of
 *   data), maximum-likelihood (divisor-N) scaling, matching what `sampleCov`
 *   produces and the returned `S`. If you hold an unbiased (divisor-(N-1))
 *   covariance — R's `cov()`, most stats packages — rescale it by (n-1)/n
 *   first so the chi-square, log-likelihood, AIC and BIC match lavaan.
 * @param {number} [spec.n] - Sample size (required with cov)
 * @param {Array<string>} [spec.names] - Variable names (required with cov)
 * @returns {{estimates: Array<{lhs: string, op: string, rhs: string, est: number,
 *   se: number|null, z: number|null, pvalue: number|null, free: boolean}>,
 *   fit: {chisq: number, df: number, pvalue: number, baselineChisq: number,
 *   baselineDf: number, cfi: number, tli: number, rmsea: number, srmr: number,
 *   logLik: number, aic: number, bic: number, npar: number, n: number, fmin: number},
 *   Sigma: Array<Array<number>>, theta: Array<number>, converged: boolean,
 *   iterations: number, S: Array<Array<number>>, variables: Array<string>,
 *   latents: Array<string>, observed: Array<string>, summary: () => string}}
 *   Fitted model: parameter estimates, fit measures, model-implied `Sigma`,
 *   reordered sample covariance `S`, and a `summary()` text formatter
 */
export function sem(syntax, spec = {}) {
  if (typeof syntax !== 'string' || !syntax.trim()) {
    throw new Error('sem: model syntax must be a non-empty string');
  }

  let S;
  let n;
  let names;
  if (spec.data) {
    if (!Array.isArray(spec.data) || spec.data.length < 3) {
      throw new Error('sem: spec.data must be an array of at least 3 row objects');
    }
    names = Object.keys(spec.data[0]).filter((k) => typeof spec.data[0][k] === 'number');
    const rows = parseModel(syntax);
    const latents = new Set(rows.filter((r) => r.op === '=~').map((r) => r.lhs));
    const used = [...new Set(rows.flatMap((r) => [r.lhs, r.rhs]))].filter((v) => !latents.has(v));
    names = names.filter((name) => used.includes(name));
    ({ S, n } = sampleCov(spec.data, names));
  } else if (spec.cov) {
    if (!Array.isArray(spec.names) || typeof spec.n !== 'number') {
      throw new Error('sem: with spec.cov, spec.names and spec.n are required');
    }
    S = spec.cov;
    n = spec.n;
    names = spec.names;
  } else {
    throw new Error('sem: provide spec.data or spec.cov (+ names, n)');
  }

  const rows = parseModel(syntax);
  const model = buildModel(rows, names);

  // Reorder S to the model's observed-variable order
  const order = model.observed.map((v) => names.indexOf(v));
  const Sobs = order.map((i) => order.map((j) => S[i][j]));
  setSampleStarts(model, Sobs);

  const result = estimate(model, Sobs, n);

  return {
    ...result,
    S: Sobs,
    variables: model.variables,
    latents: model.latents,
    observed: model.observed,
    summary() {
      const f = result.fit;
      const lines = [];
      lines.push(`tangent/sem — ML estimation (${f.npar} free parameters, N = ${f.n})`);
      lines.push(`  chisq = ${f.chisq.toFixed(3)}  df = ${f.df}  p = ${f.pvalue.toFixed(4)}`);
      lines.push(`  CFI = ${f.cfi.toFixed(3)}  TLI = ${f.tli.toFixed(3)}  ` +
        `RMSEA = ${f.rmsea.toFixed(3)}  SRMR = ${f.srmr.toFixed(3)}`);
      lines.push(`  logLik = ${f.logLik.toFixed(2)}  AIC = ${f.aic.toFixed(2)}  BIC = ${f.bic.toFixed(2)}`);
      lines.push('');
      lines.push('  lhs        op  rhs          est      se       z    p');
      for (const e of result.estimates) {
        const se = e.se === null ? '     —' : e.se.toFixed(3).padStart(7);
        const z = e.z === null ? '     —' : e.z.toFixed(2).padStart(7);
        const pv = e.pvalue === null ? '    —' : e.pvalue < 0.001 ? '<.001' : e.pvalue.toFixed(3);
        lines.push(`  ${e.lhs.padEnd(10)} ${e.op.padEnd(3)} ${e.rhs.padEnd(10)} ` +
          `${e.est.toFixed(3).padStart(7)} ${se} ${z}  ${pv}`);
      }
      return lines.join('\n');
    },
  };
}

/**
 * Alias: confirmatory factor analysis (same engine, reads better in code).
 * @type {typeof sem}
 */
export const cfa = sem;

export { parseModel } from './parse.js';
export { buildModel } from './model.js';
export { sampleCov } from './fit.js';

/** Default export bundling the primary entry points ({@link sem}, {@link cfa}, {@link parseModel}). */
export default { sem, cfa, parseModel };
