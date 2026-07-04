#!/usr/bin/env node
/**
 * Helper: fit a model with @tangent.to/sem on data supplied by the
 * Python driver. Reads JSON {syntax, data: {col: [...]}} and prints
 * JSON {estimates, chisq, df, cfi, rmsea, srmr, converged}.
 */

import { readFileSync } from 'node:fs';
import { sem } from '../src/index.js';

const spec = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const cols = Object.keys(spec.data);
const nRows = spec.data[cols[0]].length;
const data = Array.from({ length: nRows }, (_, i) => {
  const row = {};
  for (const c of cols) row[c] = spec.data[c][i];
  return row;
});

const fit = sem(spec.syntax, { data });

process.stdout.write(JSON.stringify({
  estimates: fit.estimates.map((e) => ({
    lhs: e.lhs, op: e.op, rhs: e.rhs, est: e.est, se: e.se, free: e.free,
  })),
  chisq: fit.fit.chisq,
  df: fit.fit.df,
  cfi: fit.fit.cfi,
  rmsea: fit.fit.rmsea,
  srmr: fit.fit.srmr,
  converged: fit.converged,
}));
