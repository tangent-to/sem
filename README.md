# tangent/sem

Structural equation modeling for JavaScript (ESM). Browser-first, runs in
Node.js and Deno. The first maintained SEM implementation in the JavaScript
ecosystem — part of the [tangent suite](https://github.com/tangent-to),
built entirely on its validated numeric leaves:
[lina](https://github.com/tangent-to/lina) (linear algebra),
[opt](https://github.com/tangent-to/opt) (L-BFGS),
[proba](https://github.com/tangent-to/proba) (test distributions).

- **Confirmatory factor analysis**, **path analysis**, and **full SEM**
  (latent regressions, correlated residuals)
- **lavaan-style model syntax**: `=~`, `~`, `~~`, `1*x` to fix, `NA*x` to free
- **Maximum-likelihood estimation** (RAM parameterization), expected-
  information standard errors, χ²/CFI/TLI/RMSEA/SRMR/AIC/BIC
- lavaan identification defaults: first-indicator markers, automatic
  residual variances and exogenous-factor covariances

## Install

```bash
npm install @tangent.to/sem     # npm
deno add jsr:@tangent/sem       # Deno / JSR
```

## Usage

```javascript
import { sem } from '@tangent.to/sem';

const fit = sem(`
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
`, { data });          // rows as objects; or { cov, n, names }

fit.fit.chisq;         // 85.306 on Holzinger-Swineford, matching lavaan
fit.fit.cfi;           // 0.931
fit.estimates;         // [{lhs, op, rhs, est, se, z, pvalue}, ...]
console.log(fit.summary());
```

Full SEM with latent regressions:

```javascript
const pd = sem(`
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  dem65 =~ y5 + y6 + y7 + y8
  dem60 ~ ind60
  dem65 ~ ind60 + dem60
  y1 ~~ y5
`, { data });
```

## Validation

`tests_compare-to-semopy/` fits the two canonical benchmarks with both
tangent/sem and [semopy](https://semopy.com/) (which reproduces R's
lavaan) on identical data:

- **Holzinger-Swineford CFA**: χ² = 85.306 (lavaan's published value,
  exactly), estimates and expected-information standard errors matching
  semopy to ~1e-3 relative; CFI/TLI/RMSEA/SRMR/AIC/BIC match lavaan's
  published output.
- **Bollen political democracy full SEM**: χ² = 38.125 (lavaan's
  published value); our optimum is marginally *tighter* than semopy's and
  matches lavaan's published parameter table more closely.

```bash
npm run test:semopy    # requires uv and Node
```

## Scope

Covariance-structure ML for complete numeric data: CFA, path analysis,
full SEM. Not yet: mean structures/intercepts, multiple groups, missing
data (FIML), ordinal indicators (WLSMV), robust corrections. Those are
roadmap items, prioritized by demand.

## License

GPL-3.0 (application layer of the tangent suite; the numeric leaves it
builds on are MIT).
