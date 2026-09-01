/**
 * Maximum-likelihood estimation of RAM models.
 *
 * F_ML(theta) = log|Sigma| + tr(S Sigma^{-1}) - log|S| - p
 * minimized with @tangent.to/opt's L-BFGS (finite-difference gradients).
 * Standard errors come from the expected (Fisher) information matrix at the
 * optimum, acov(theta) = I^{-1} / N; the observed numerical Hessian,
 * acov(theta) = (2 / N) H^{-1}, is the fallback when I is singular. The test
 * statistic is T = N * F_ML with S the maximum-likelihood (divisor-N) sample
 * covariance — lavaan's default normal-theory statistic. This reproduces
 * lavaan's published chi-square exactly (e.g. 85.306, df 24, on the classic
 * Holzinger-Swineford three-factor model at N = 301). Because F_ML is
 * invariant to a global rescaling of S, using N (rather than N - 1) as the
 * divisor and the multiplier are what pin T to lavaan's value.
 */

import { cholesky, identity, inv, matmul, solve, transpose } from '@tangent.to/lina';
import {
  add as gAdd,
  concat as gConcat,
  inv as gInv,
  logdetPSD as gLogdet,
  matmul as gMatmul,
  reshape as gReshape,
  slice as gSlice,
  solvePSD as gSolve,
  sub as gSub,
  trace as gTrace,
  transpose as gTranspose,
  valueAndGrad as gValueAndGrad,
} from '@tangent.to/grad';
import { lbfgs, numericalHessian } from '@tangent.to/opt';
import { chi2 as chi2Dist, normal } from '@tangent.to/proba';

/**
 * Sample covariance matrix and means from raw data.
 *
 * Uses the maximum-likelihood divisor N (lavaan's default rescaling), not the
 * unbiased N - 1. Every value must be finite; missing or non-numeric data
 * raises an Error, so impute or drop it first.
 *
 * @param {Array<Object>} data - Rows as objects, one field per variable
 * @param {Array<string>} names - Variable names to include, in output order
 * @returns {{S: Array<Array<number>>, means: Array<number>, n: number}} `S` is
 *   the `names.length` square covariance matrix, `means` the per-variable
 *   means, `n` the number of rows
 */
export function sampleCov(data, names) {
  const n = data.length;
  const p = names.length;
  const means = new Array(p).fill(0);
  const X = data.map((row) => names.map((name) => {
    const v = row[name];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new Error(`Non-numeric value for '${name}'; remove or impute missing data first`);
    }
    return v;
  }));
  for (const row of X) {
    for (let j = 0; j < p; j++) means[j] += row[j];
  }
  for (let j = 0; j < p; j++) means[j] /= n;
  const S = Array.from({ length: p }, () => new Array(p).fill(0));
  for (const row of X) {
    for (let i = 0; i < p; i++) {
      const di = row[i] - means[i];
      for (let j = i; j < p; j++) {
        S[i][j] += di * (row[j] - means[j]);
      }
    }
  }
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      S[i][j] /= n;
      S[j][i] = S[i][j];
    }
  }
  return { S, means, n };
}

/** log|M| for a symmetric positive-definite matrix via Cholesky; null if not PD. */
function logDetSpd(M) {
  try {
    const L = cholesky(M);
    let ld = 0;
    for (let i = 0; i < L.length; i++) ld += Math.log(L[i][i]);
    return 2 * ld;
  } catch {
    return null;
  }
}

/** Build Sigma(theta) from the RAM matrices. */
export function buildSigma(model, theta) {
  const { t, p, params } = model;
  const A = Array.from({ length: t }, () => new Array(t).fill(0));
  const S = Array.from({ length: t }, () => new Array(t).fill(0));
  let k = 0;
  for (const par of params) {
    const value = par.free ? theta[k++] : par.value;
    if (par.matrix === 'A') {
      A[par.row][par.col] = value;
    } else {
      S[par.row][par.col] = value;
      S[par.col][par.row] = value;
    }
  }
  // B = I - A; M = B^{-1}; Sigma_full = M S M^T; Sigma = observed block
  const B = identity(t);
  for (let i = 0; i < t; i++) {
    for (let j = 0; j < t; j++) B[i][j] -= A[i][j];
  }
  const M = inv(B);
  const full = matmul(matmul(M, S), transpose(M));
  const Sigma = Array.from({ length: p }, (_, i) => full[i].slice(0, p));
  return Sigma;
}

/**
 * Sigma(theta) as a differentiable expression, mirroring {@link buildSigma}.
 *
 * Same RAM algebra — B = I - A, Sigma = (B^-1 S B^-T) restricted to the
 * observed block — written in @tangent.to/grad ops so the discrepancy carries
 * its own exact gradient. `theta` is a grad Var holding the free parameters in
 * `model.params` order.
 *
 * The matrices are assembled by concatenating their cells and reshaping, which
 * keeps every free parameter on the tape: a fixed cell contributes a constant,
 * a free cell a slice of `theta`. B is a matrix of directed paths and is not
 * symmetric, hence `inv` rather than the Cholesky-based solve.
 *
 * @private
 */
function buildSigmaAD(model, theta) {
  const { t, p, params } = model;
  const aCells = new Array(t * t).fill(0);
  const sCells = new Array(t * t).fill(0);
  let k = 0;
  for (const par of params) {
    const cell = par.free ? gSlice(theta, [k++], [1]) : par.value;
    if (par.matrix === 'A') {
      aCells[par.row * t + par.col] = cell;
    } else {
      sCells[par.row * t + par.col] = cell;
      sCells[par.col * t + par.row] = cell;
    }
  }
  const A = gReshape(gConcat(aCells), [t, t]);
  const S = gReshape(gConcat(sCells), [t, t]);
  const M = gInv(gSub(identity(t), A));
  const full = gMatmul(gMatmul(M, S), gTranspose(M));
  return gSlice(full, [0, 0], [p, p]);
}

/**
 * F_ML and its exact gradient with respect to the free parameters.
 *
 * The objective L-BFGS minimizes. Returning the combined `{loss, gradient}`
 * form lets @tangent.to/opt skip its finite-difference fallback, which costs
 * 2·q extra evaluations of the whole RAM algebra per gradient.
 *
 * The gain is speed, not convergence. Measured on confirmatory factor models,
 * both paths take the SAME number of iterations and reach the same optimum to
 * every printed digit — the finite-difference gradient is accurate enough to
 * steer L-BFGS perfectly well here, since F_ML is a smooth function of theta:
 *
 *     observed  q    objective evaluations      time
 *        6      13      783  ->    29       21 ms -> 8 ms
 *       12      27     2365  ->    43      126 ms -> 23 ms
 *       20      46     3348  ->    36      392 ms -> 45 ms
 *
 * The ratio grows with the parameter count, because the fallback's cost does
 * and this does not.
 *
 * @param {Object} model - From buildModel
 * @param {Array<number>} thetaArr - Free parameter values
 * @param {Array<Array<number>>} S - Sample covariance
 * @param {number} logDetS - log|S|, precomputed
 * @returns {{loss: number, gradient: Array<number>}}
 */
export function discrepancyAndGrad(model, thetaArr, S, logDetS) {
  const p = model.p;
  const build = (th) => {
    const Sigma = buildSigmaAD(model, th);
    // log|Sigma| + tr(S Sigma^-1); the two constants come off outside the tape.
    return gAdd(gLogdet(Sigma), gTrace(gSolve(Sigma, S)));
  };
  let out;
  try {
    out = gValueAndGrad(build)(thetaArr);
  } catch {
    // Sigma left the positive-definite cone: same large penalty the
    // finite-difference path reports, with a zero gradient so the line search
    // backs off rather than following a meaningless direction.
    return { loss: 1e10, gradient: new Array(thetaArr.length).fill(0) };
  }
  const f = out.value - logDetS - p;
  if (!Number.isFinite(f)) {
    return { loss: 1e10, gradient: new Array(thetaArr.length).fill(0) };
  }
  return { loss: f, gradient: out.gradient };
}

/** F_ML discrepancy; large penalty when Sigma is not positive definite. */
export function discrepancy(model, theta, S, logDetS) {
  const p = model.p;
  const Sigma = buildSigma(model, theta);
  const ld = logDetSpd(Sigma);
  if (ld === null) return 1e10;
  // tr(S Sigma^{-1}) = trace of Sigma^{-1} S = sum of diag of solve(Sigma, S)
  let tr = 0;
  const X = solve(Sigma, S);
  for (let i = 0; i < p; i++) tr += X[i][i];
  const f = ld + tr - logDetS - p;
  return Number.isFinite(f) ? f : 1e10;
}


/**
 * Fit the model by maximum likelihood.
 *
 * @param {Object} model - From buildModel (starts already set)
 * @param {Array<Array<number>>} S - Sample covariance, maximum-likelihood
 *   (divisor-N) scaling, as produced by {@link sampleCov}
 * @param {number} n - Sample size
 * @returns {{estimates: Array<{lhs: string, op: string, rhs: string, est: number,
 *   se: number|null, z: number|null, pvalue: number|null, free: boolean}>,
 *   fit: {chisq: number, df: number, pvalue: number, baselineChisq: number,
 *   baselineDf: number, cfi: number, tli: number, rmsea: number, srmr: number,
 *   logLik: number, aic: number, bic: number, npar: number, n: number, fmin: number},
 *   Sigma: Array<Array<number>>, theta: Array<number>, converged: boolean,
 *   iterations: number}} Estimates table, fit measures, model-implied `Sigma`,
 *   optimum `theta`, and optimiser status
 */
export function estimate(model, S, n) {
  const p = model.p;
  const logDetS = logDetSpd(S);
  if (logDetS === null) {
    throw new Error('Sample covariance matrix is not positive definite');
  }

  const freeParams = model.params.filter((par) => par.free);
  const q = freeParams.length;
  const dfModel = (p * (p + 1)) / 2 - q;
  if (dfModel < 0) {
    throw new Error(`Model is not identified: ${q} free parameters exceed ${(p * (p + 1)) / 2} covariance moments`);
  }

  const x0 = freeParams.map((par) => par.start);
  // Exact gradient, in the combined {loss, gradient} form @tangent.to/opt
  // accepts, in place of its 2·q-evaluations-per-gradient fallback. Same
  // iterations and same optimum either way; see discrepancyAndGrad for the
  // measured cost.
  const objective = (theta) => discrepancyAndGrad(model, theta, S, logDetS);

  const result = lbfgs(objective, x0, { maxIter: 5000, tol: 1e-9 });
  const theta = result.x;
  const fmin = result.fx;

  // Standard errors from the EXPECTED information matrix (lavaan default):
  // I_ij = 0.5 * tr(Sigma^{-1} dSigma/di Sigma^{-1} dSigma/dj), acov = I^{-1}/N
  let se = new Array(q).fill(NaN);
  try {
    const SigmaHat = buildSigma(model, theta);
    const SigmaInv = inv(SigmaHat);
    // Central-difference Jacobian of Sigma w.r.t. each free parameter.
    // Deliberately NOT autodiffed. Sigma is a smooth rational function of
    // theta, and halving the step from 1e-5 to 1e-7 moves the resulting
    // standard errors by 1e-9 relative — identical to every digit reported.
    // An exact Jacobian would cost p² reverse sweeps against q forward
    // evaluations here, buying nothing.
    const D = new Array(q);
    for (let k = 0; k < q; k++) {
      const h = 1e-5 * Math.max(1, Math.abs(theta[k]));
      const tp = theta.slice();
      tp[k] += h;
      const Sp = buildSigma(model, tp);
      tp[k] = theta[k] - h;
      const Sm = buildSigma(model, tp);
      D[k] = Sp.map((row, i) => row.map((v, j) => (v - Sm[i][j]) / (2 * h)));
    }
    // Precompute Sigma^{-1} D_k, then I_ij = 0.5 * tr(M_i M_j)
    const M = D.map((Dk) => matmul(SigmaInv, Dk));
    const I = Array.from({ length: q }, () => new Array(q).fill(0));
    for (let i = 0; i < q; i++) {
      for (let j = i; j < q; j++) {
        let tr = 0;
        const Mi = M[i];
        const Mj = M[j];
        for (let a = 0; a < p; a++) {
          for (let b = 0; b < p; b++) tr += Mi[a][b] * Mj[b][a];
        }
        I[i][j] = 0.5 * tr;
        I[j][i] = I[i][j];
      }
    }
    const Iinv = inv(I);
    se = theta.map((_, i) => {
      const v = Iinv[i][i] / n;
      return v > 0 ? Math.sqrt(v) : NaN;
    });
  } catch {
    // Singular information matrix: fall back to the observed Hessian
    try {
      const H = numericalHessian(objective, theta);
      const Hinv = inv(H);
      se = theta.map((_, i) => {
        const v = (2 / n) * Hinv[i][i];
        return v > 0 ? Math.sqrt(v) : NaN;
      });
    } catch {
      // leave NaN standard errors
    }
  }

  // Test statistic, lavaan's default (normal-theory) convention: T = N * F_ML,
  // with F evaluated on the ML (divisor-N) sample covariance S. Empirically
  // reproduces lavaan's published chi-square exactly (85.306 on
  // Holzinger-Swineford). F_ML is scale-invariant, so it is the multiplier N
  // (not N - 1) that pins T to lavaan; using N - 1 would deflate it to ~85.02.
  const T = n * fmin;
  const pValue = dfModel > 0 ? 1 - chi2Dist.cdf(T, { k: dfModel }) : 1;

  // Baseline (independence) model: Sigma = diag(S)
  // F_b = sum(log s_ii) - log|S|; df_b = p(p-1)/2
  let logDiag = 0;
  for (let i = 0; i < p; i++) logDiag += Math.log(S[i][i]);
  const fBaseline = logDiag - logDetS;
  const Tb = n * fBaseline;
  const dfB = (p * (p - 1)) / 2;

  const num = Math.max(T - dfModel, 0);
  const den = Math.max(Tb - dfB, T - dfModel, 0);
  const cfi = den > 0 ? 1 - num / den : 1;
  const tli = dfB > 0 && dfModel > 0
    ? (Tb / dfB - T / dfModel) / (Tb / dfB - 1)
    : 1;
  const rmsea = dfModel > 0 ? Math.sqrt(Math.max(T - dfModel, 0) / (dfModel * n)) : 0;

  // SRMR: standardized residuals of Sigma vs S
  const Sigma = buildSigma(model, theta);
  let srmrSum = 0;
  let srmrCount = 0;
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      const resid = (S[i][j] - Sigma[i][j]) / Math.sqrt(S[i][i] * S[j][j]);
      srmrSum += resid * resid;
      srmrCount++;
    }
  }
  const srmr = Math.sqrt(srmrSum / srmrCount);

  // Log-likelihood, AIC, BIC (normal likelihood at the MLE)
  let trSSigmaInv = 0;
  const X = solve(Sigma, S);
  for (let i = 0; i < p; i++) trSSigmaInv += X[i][i];
  const logDetSigma = logDetSpd(Sigma);
  // S is the ML (divisor N) covariance, so tr(S Sigma^{-1}) is the plug-in term
  const logLik = -0.5 * n * (p * Math.log(2 * Math.PI) + logDetSigma + trSSigmaInv);
  const aic = -2 * logLik + 2 * q;
  const bic = -2 * logLik + q * Math.log(n);

  // Assemble the estimates table
  let k = 0;
  const estimates = model.params.map((par) => {
    const est = par.free ? theta[k] : par.value;
    const stdErr = par.free ? se[k] : null;
    if (par.free) k++;
    const z = par.free && Number.isFinite(stdErr) ? est / stdErr : null;
    return {
      lhs: par.lhs,
      op: par.op,
      rhs: par.rhs,
      est,
      se: stdErr,
      z,
      pvalue: z !== null ? 2 * (1 - normal.cdf(Math.abs(z), { mu: 0, sigma: 1 })) : null,
      free: par.free,
    };
  });

  return {
    estimates,
    fit: {
      chisq: T,
      df: dfModel,
      pvalue: pValue,
      baselineChisq: Tb,
      baselineDf: dfB,
      cfi,
      tli,
      rmsea,
      srmr,
      logLik,
      aic,
      bic,
      npar: q,
      n,
      fmin,
    },
    Sigma,
    theta,
    converged: result.converged,
    iterations: result.iterations,
  };
}
