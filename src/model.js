/**
 * Model builder: turn parsed parameter rows into a RAM specification.
 *
 * RAM (Reticular Action Model): with B = I - A,
 *   Sigma(theta) = F B^{-1} S B^{-T} F^T
 * where A holds directed paths (loadings, regressions), S symmetric
 * (co)variances, and F selects the observed variables.
 *
 * Identification defaults follow lavaan's CFA conventions:
 * - each factor's first indicator loading is fixed to 1 (marker method)
 *   unless the user fixed another value or wrote NA* to free it
 * - residual variances for all observed endogenous variables are added, free
 * - variances for all latent variables are added, free
 * - covariances between exogenous latent variables are added, free
 * - exogenous observed variables get free variances and pairwise
 *   covariances (the fixed.x=FALSE convention; all variables are modeled)
 */

/**
 * Build the model specification from parsed rows.
 *
 * @param {Array<Object>} rows - Output of parseModel
 * @param {Array<string>} observedNames - Observed variable names present in the data
 * @returns {Object} {variables, latents, observed, params, aIndex, sIndex, t, p}
 */
export function buildModel(rows, observedNames) {
  const latents = [];
  for (const r of rows) {
    if (r.op === '=~' && !latents.includes(r.lhs)) latents.push(r.lhs);
  }

  // Every non-latent name mentioned must exist in the data
  const mentioned = new Set();
  for (const r of rows) {
    mentioned.add(r.lhs);
    mentioned.add(r.rhs);
  }
  const observed = [];
  for (const name of mentioned) {
    if (!latents.includes(name)) {
      if (!observedNames.includes(name)) {
        throw new Error(`Variable '${name}' is not in the data (columns: ${observedNames.join(', ')})`);
      }
      if (!observed.includes(name)) observed.push(name);
    }
  }
  // Keep data order for observed variables (stable Sigma layout)
  observed.sort((a, b) => observedNames.indexOf(a) - observedNames.indexOf(b));

  const variables = [...observed, ...latents];
  const idx = new Map(variables.map((v, i) => [v, i]));
  const p = observed.length;
  const t = variables.length;

  // Endogenous: indicators of factors, and lhs of regressions
  const endogenous = new Set();
  for (const r of rows) {
    if (r.op === '=~') endogenous.add(r.rhs);
    if (r.op === '~') endogenous.add(r.lhs);
  }

  // Assemble the parameter table. Each entry:
  // {lhs, op, rhs, matrix: 'A'|'S', row, col, free, value, start}
  const params = [];
  const seen = new Set();
  const key = (op, a, b) => op === '~~' ? `~~:${[a, b].sort().join(':')}` : `${op}:${a}:${b}`;

  const firstIndicator = new Map(); // factor -> first '=~' row
  for (const r of rows) {
    if (r.op === '=~' && !firstIndicator.has(r.lhs)) firstIndicator.set(r.lhs, r);
  }

  for (const r of rows) {
    seen.add(key(r.op, r.lhs, r.rhs));
    if (r.op === '=~') {
      // Loading: path factor -> indicator, A[indicator][factor]
      const isMarker = firstIndicator.get(r.lhs) === r;
      let free = true;
      let value = 1;
      if (r.fixed !== null) {
        free = false;
        value = r.fixed;
      } else if (isMarker && !r.freed) {
        free = false;
        value = 1;
      }
      params.push({
        lhs: r.lhs, op: '=~', rhs: r.rhs, matrix: 'A',
        row: idx.get(r.rhs), col: idx.get(r.lhs),
        free, value, start: free ? 1 : value,
      });
    } else if (r.op === '~') {
      params.push({
        lhs: r.lhs, op: '~', rhs: r.rhs, matrix: 'A',
        row: idx.get(r.lhs), col: idx.get(r.rhs),
        free: r.fixed === null, value: r.fixed ?? 0, start: 0,
      });
    } else {
      params.push({
        lhs: r.lhs, op: '~~', rhs: r.rhs, matrix: 'S',
        row: idx.get(r.lhs), col: idx.get(r.rhs),
        free: r.fixed === null, value: r.fixed ?? 0,
        start: r.lhs === r.rhs ? 0.05 : 0,
      });
    }
  }

  // Defaults: variances for every variable unless specified
  for (const v of variables) {
    if (!seen.has(key('~~', v, v))) {
      const isLatent = latents.includes(v);
      params.push({
        lhs: v, op: '~~', rhs: v, matrix: 'S',
        row: idx.get(v), col: idx.get(v),
        free: true, value: 0,
        start: isLatent ? 0.05 : null, // null start -> filled from sample variance
      });
      seen.add(key('~~', v, v));
    }
  }

  // Covariances among exogenous latents (CFA default)
  const exoLatents = latents.filter((f) => !endogenous.has(f));
  for (let i = 0; i < exoLatents.length; i++) {
    for (let j = i + 1; j < exoLatents.length; j++) {
      if (!seen.has(key('~~', exoLatents[i], exoLatents[j]))) {
        params.push({
          lhs: exoLatents[i], op: '~~', rhs: exoLatents[j], matrix: 'S',
          row: idx.get(exoLatents[i]), col: idx.get(exoLatents[j]),
          free: true, value: 0, start: 0,
        });
        seen.add(key('~~', exoLatents[i], exoLatents[j]));
      }
    }
  }

  // Covariances among exogenous observed variables used as predictors
  const exoObserved = observed.filter((v) =>
    !endogenous.has(v) && rows.some((r) => r.op === '~' && r.rhs === v));
  for (let i = 0; i < exoObserved.length; i++) {
    for (let j = i + 1; j < exoObserved.length; j++) {
      if (!seen.has(key('~~', exoObserved[i], exoObserved[j]))) {
        params.push({
          lhs: exoObserved[i], op: '~~', rhs: exoObserved[j], matrix: 'S',
          row: idx.get(exoObserved[i]), col: idx.get(exoObserved[j]),
          free: true, value: 0, start: null,
        });
        seen.add(key('~~', exoObserved[i], exoObserved[j]));
      }
    }
  }

  return { variables, latents, observed, params, t, p };
}

/**
 * Fill start values that depend on the sample covariance (observed
 * variances start at half the sample variance, lavaan-style; observed
 * covariance parameters start at the sample covariance).
 *
 * @param {Object} model - From buildModel
 * @param {Array<Array<number>>} S - Sample covariance (observed order)
 */
export function setSampleStarts(model, S) {
  for (const par of model.params) {
    if (par.start === null) {
      if (par.row === par.col) {
        par.start = 0.5 * S[par.row][par.row];
      } else {
        par.start = S[par.row][par.col];
      }
    }
  }
}
