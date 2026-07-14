// ---
// title: Confirmatory factor analysis with lavaan-style syntax
// id: sem-cfa
// ---

// %% [markdown]
/*
Structural equation modeling (SEM) fits models in which unobserved *latent*
variables are measured by observed *indicators*. Confirmatory factor analysis
(CFA) is the measurement half of SEM: you name the latent factors, state which
indicators load on each, and estimate the loadings, residual variances, and
factor covariances by maximum likelihood.

`@tangent.to/sem` is the first maintained SEM implementation in JavaScript. It
takes lavaan-style model syntax, estimates by ML, and reports the same fit
indices and parameter estimates you would get from R's lavaan, matching its
published values.
*/

// %% [javascript]

import * as __lib from 'https://esm.sh/@tangent.to/sem';
const sem = __lib.sem;

// The Holzinger-Swineford 1939 data: 301 seventh- and eighth-grade students,
// nine cognitive test scores x1..x9. Rather than pasting hundreds of numbers
// into the notebook, we fetch the CSV over the network and parse it with d3
// (preloaded in tangent notebooks) into row objects, the shape sem() expects
// for spec.data. Loading real data from a URL is the normal, reproducible way
// to start an analysis, and keeps the notebook readable.
// Pinned to the v0.1.1 tag (not the mutable main branch) so the URL keeps
// resolving to this exact CSV. d3 is preloaded in tangent notebooks.
const dataUrl =
  'https://raw.githubusercontent.com/tangent-to/sem/v0.1.1/data/holzinger39.csv';
const res = await fetch(dataUrl);
if (!res.ok) {
  throw new Error(`Failed to fetch ${dataUrl}: ${res.status} ${res.statusText}`);
}
const rows = d3.csvParse(await res.text(), d3.autoType);
const columns = rows.columns;

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
as it almost always is at N = 301), so we read the approximate-fit indices:
CFI ~0.931, TLI ~0.896, RMSEA ~0.092, SRMR ~0.065. These are the values lavaan
reports for this model to the same precision. Judged against the strict
Hu-Bentler cutoffs (CFI/TLI >= 0.95, RMSEA <= 0.06), the classic three-factor
model has only mediocre fit — a well-known feature of these data, not a bug.
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
Each approximate-fit index has a conventional rule-of-thumb cutoff. The bars are
the fitted values; the dashed line is the threshold; green means the value meets
its cutoff, red means it misses. Incremental indices (CFI, TLI) should clear
their line, residual indices (RMSEA, SRMR) should stay under it.
*/

// %% [javascript]

const fitIndices = [
  { index: 'CFI', value: result.fit.cfi, threshold: 0.95, higherIsBetter: true },
  { index: 'TLI', value: result.fit.tli, threshold: 0.95, higherIsBetter: true },
  { index: 'RMSEA', value: result.fit.rmsea, threshold: 0.06, higherIsBetter: false },
  { index: 'SRMR', value: result.fit.srmr, threshold: 0.08, higherIsBetter: false },
].map((d) => ({
  ...d,
  pass: d.higherIsBetter ? d.value >= d.threshold : d.value <= d.threshold,
}));

Plot.plot({
  marginLeft: 70,
  height: 240,
  x: { domain: [0, 1], label: 'value (bar) vs rule-of-thumb cutoff (dashed)' },
  fy: { label: null },
  y: { axis: null },
  color: {
    domain: [true, false],
    range: ['#2e7d32', '#c62828'],
    legend: true,
    label: 'meets cutoff',
    tickFormat: (d) => (d ? 'yes' : 'no'),
  },
  marks: [
    Plot.barX(fitIndices, { x: 'value', fy: 'index', fill: 'pass' }),
    Plot.text(fitIndices, {
      x: 'value',
      fy: 'index',
      text: (d) => d.value.toFixed(3),
      textAnchor: 'start',
      dx: 4,
    }),
    Plot.ruleX(fitIndices, { x: 'threshold', fy: 'index', strokeWidth: 2, strokeDasharray: '4 3' }),
    Plot.ruleX([0]),
  ],
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
Standardizing each loading (multiplying by the factor SD and dividing by the
model-implied indicator SD) puts every item on a 0-1 scale, so the marker
indicators become comparable too. Faceting by factor shows the three
measurement blocks: every indicator loads strongly on its own ability.
*/

// %% [javascript]

const stdLoadings = result.estimates
  .filter((e) => e.op === '=~')
  .map((e) => {
    const factorVar = result.estimates.find((v) => v.op === '~~' && v.lhs === e.lhs && v.rhs === e.lhs).est;
    const residVar = result.estimates.find((v) => v.op === '~~' && v.lhs === e.rhs && v.rhs === e.rhs).est;
    const impliedSd = Math.sqrt(e.est * e.est * factorVar + residVar);
    return {
      factor: e.lhs,
      indicator: e.rhs,
      std: (e.est * Math.sqrt(factorVar)) / impliedSd,
    };
  });

Plot.plot({
  marginLeft: 60,
  x: { domain: [0, 1], label: 'standardized loading' },
  y: { label: null },
  fy: { label: 'factor' },
  color: { legend: true, label: 'factor' },
  marks: [
    Plot.barX(stdLoadings, { x: 'std', y: 'indicator', fy: 'factor', fill: 'factor' }),
    Plot.text(stdLoadings, {
      x: 'std',
      y: 'indicator',
      fy: 'factor',
      text: (d) => d.std.toFixed(2),
      textAnchor: 'start',
      dx: 4,
    }),
    Plot.ruleX([0]),
  ],
});

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
The same correlations read most easily as a heatmap of the full 3x3 inter-factor
matrix (diagonal fixed to 1). Darker cells are stronger relationships: visual
pairs moderately with textual and speed, while textual and speed are the most
weakly related pair.
*/

// %% [javascript]

const factorList = result.latents;
const factorCorr = (a, b) => {
  if (a === b) return 1;
  const cov = result.estimates.find(
    (e) => e.op === '~~' && e.lhs !== e.rhs &&
      ((e.lhs === a && e.rhs === b) || (e.lhs === b && e.rhs === a)),
  ).est;
  const varA = result.estimates.find((v) => v.op === '~~' && v.lhs === a && v.rhs === a).est;
  const varB = result.estimates.find((v) => v.op === '~~' && v.lhs === b && v.rhs === b).est;
  return cov / Math.sqrt(varA * varB);
};
const corrMatrix = factorList.flatMap((a) => factorList.map((b) => ({ a, b, r: factorCorr(a, b) })));

Plot.plot({
  marginLeft: 70,
  width: 380,
  height: 320,
  x: { label: null },
  y: { label: null },
  color: { scheme: 'blues', domain: [0, 1], legend: true, label: 'correlation' },
  marks: [
    Plot.cell(corrMatrix, { x: 'b', y: 'a', fill: 'r', inset: 0.5 }),
    Plot.text(corrMatrix, {
      x: 'b',
      y: 'a',
      text: (d) => d.r.toFixed(2),
      fill: (d) => (d.r > 0.6 ? 'white' : 'black'),
    }),
  ],
});

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
