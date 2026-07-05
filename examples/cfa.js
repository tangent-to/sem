// ---
// title: Confirmatory factor analysis with lavaan-style syntax
// id: sem-cfa
// ---

// %% [markdown]
/*
# Confirmatory factor analysis

Structural equation modeling (SEM) fits models in which unobserved *latent*
variables are measured by observed *indicators*. Confirmatory factor analysis
(CFA) is the measurement half of SEM: you name the latent factors, state which
indicators load on each, and estimate the loadings, residual variances, and
factor covariances by maximum likelihood.

`@tangent.to/sem` is the first maintained SEM implementation in JavaScript. It
takes lavaan-style model syntax, estimates by ML, and reports the same fit
indices and parameter estimates you would get from R's lavaan, matching its
published values.

This notebook imports the local build. Once the package is published you would
import it from a CDN instead:

    import { sem } from 'https://esm.sh/@tangent.to/sem';
*/

// %% [javascript]

import { sem } from '../dist/index.js';
import { readFileSync } from 'node:fs';

// The Holzinger-Swineford 1939 data: 301 seventh- and eighth-grade students,
// nine cognitive test scores x1..x9. We parse the CSV into row objects, which
// is the shape sem() expects for spec.data. In a browser notebook you would
// fetch the CSV instead; here we read it from disk.
const lines = readFileSync(new URL('../data/holzinger39.csv', import.meta.url), 'utf8')
  .trim()
  .split('\n');
const columns = lines[0].split(',');
const rows = lines.slice(1).map((line) => {
  const values = line.split(',').map(Number);
  return Object.fromEntries(columns.map((name, i) => [name, values[i]]));
});

({
  n_rows: rows.length,
  variables: columns,
  first_row: rows[0],
});

// %% [markdown]
/*
## The three-factor measurement model

The classic model for these data posits three correlated latent abilities. In
lavaan syntax `=~` reads "is measured by": the factor on the left is an
unobserved cause of the indicators on the right.

    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9

Each factor is measured by three tests. To fix the scale of a latent variable,
sem (like lavaan) fixes the first loading of each factor to 1; that indicator
is the *marker*. We pass the syntax and the data to `sem()`, which computes the
sample covariance, builds the model, and estimates it.
*/

// %% [javascript]

const model = `
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
`;

const result = sem(model, { data: rows });

({
  converged: result.converged,
  latents: result.latents,
  free_parameters: result.fit.npar,
});

// %% [markdown]
/*
## Global fit

`result.fit` reports how well the model-implied covariance matrix reproduces
the sample covariance. The chi-square tests exact fit (it is significant here,
as it almost always is at N = 301), so we read the approximate-fit indices: CFI
above ~0.90, RMSEA around 0.09, SRMR around 0.065. These are the values lavaan
reports for this model to the same precision.
*/

// %% [javascript]

({
  chisq: result.fit.chisq, // ~85.31
  df: result.fit.df, // 24
  pvalue: result.fit.pvalue,
  cfi: result.fit.cfi, // ~0.931
  tli: result.fit.tli,
  rmsea: result.fit.rmsea, // ~0.092
  srmr: result.fit.srmr, // ~0.065
});

// %% [markdown]
/*
## Factor loadings

The loadings (`op === '=~'`) are the regression weights of each indicator on
its factor: how strongly the latent ability drives the observed score. `est`
is the estimate, `se` its standard error, and `z = est / se` the Wald
statistic. The marker indicators (x1, x4, x7) are fixed to 1, so they carry no
standard error; every other loading is freely estimated and, here, strongly
significant.
*/

// %% [javascript]

const loadings = result.estimates
  .filter((e) => e.op === '=~')
  .map((e) => ({
    factor: e.lhs,
    indicator: e.rhs,
    est: e.est,
    se: e.se,
    z: e.z,
  }));

loadings;

// %% [markdown]
/*
## Factor variances and covariances

Entries with `op === '~~'` are (co)variances. The diagonal terms where lhs
equals rhs are variances (of the factors and of the indicator residuals); the
off-diagonal factor terms are the covariances among visual, textual, and speed.
All three factors are positively related. Dividing each covariance by the
square root of the two factor variances turns it into a correlation: visual and
textual correlate about 0.46, visual and speed about 0.47, textual and speed
about 0.28. The three abilities are distinct but share a common core, which is
exactly what a correlated-factors model is meant to express.
*/

// %% [javascript]

const factors = new Set(result.latents);
const factorCov = result.estimates
  .filter((e) => e.op === '~~' && factors.has(e.lhs) && factors.has(e.rhs) && e.lhs !== e.rhs)
  .map((e) => {
    const varLhs = result.estimates.find((v) => v.op === '~~' && v.lhs === e.lhs && v.rhs === e.lhs).est;
    const varRhs = result.estimates.find((v) => v.op === '~~' && v.lhs === e.rhs && v.rhs === e.rhs).est;
    return {
      between: `${e.lhs} ~ ${e.rhs}`,
      covariance: e.est,
      correlation: e.est / Math.sqrt(varLhs * varRhs),
      z: e.z,
    };
  });

factorCov;

// %% [markdown]
/*
## The full summary table

`result.summary()` renders the whole solution as formatted text: the fit line,
the information criteria, and every parameter with its estimate, standard
error, z value, and p value. This is the one-call view you would print to check
a fitted model at a glance.
*/

// %% [javascript]

console.log(result.summary());
