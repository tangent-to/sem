import { describe, it, expect } from 'vitest';
import { parseModel } from '../src/parse.js';

const row = (lhs, op, rhs, fixed = null, freed = false) => ({ lhs, op, rhs, fixed, freed });

describe('parseModel: measurement statements', () => {
  it('parses the Holzinger-Swineford 3-factor model into 9 rows, order preserved', () => {
    const rows = parseModel(`
      visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9
    `);
    expect(rows).toEqual([
      row('visual', '=~', 'x1'),
      row('visual', '=~', 'x2'),
      row('visual', '=~', 'x3'),
      row('textual', '=~', 'x4'),
      row('textual', '=~', 'x5'),
      row('textual', '=~', 'x6'),
      row('speed', '=~', 'x7'),
      row('speed', '=~', 'x8'),
      row('speed', '=~', 'x9'),
    ]);
  });

  it('appends rows when the same lhs appears on multiple lines', () => {
    const rows = parseModel('visual =~ x1 + x2\nvisual =~ x3');
    expect(rows).toEqual([
      row('visual', '=~', 'x1'),
      row('visual', '=~', 'x2'),
      row('visual', '=~', 'x3'),
    ]);
  });
});

describe('parseModel: regressions', () => {
  it('parses a regression with multiple predictors', () => {
    const rows = parseModel('y ~ x1 + x2 + x3');
    expect(rows).toEqual([
      row('y', '~', 'x1'),
      row('y', '~', 'x2'),
      row('y', '~', 'x3'),
    ]);
  });

  it('parses several regression lines in source order', () => {
    const rows = parseModel('dem60 ~ ind60\ndem65 ~ ind60 + dem60');
    expect(rows).toEqual([
      row('dem60', '~', 'ind60'),
      row('dem65', '~', 'ind60'),
      row('dem65', '~', 'dem60'),
    ]);
  });
});

describe('parseModel: variances and covariances', () => {
  it('parses a covariance row', () => {
    expect(parseModel('a ~~ b')).toEqual([row('a', '~~', 'b')]);
  });

  it('parses a variance row (a ~~ a)', () => {
    expect(parseModel('a ~~ a')).toEqual([row('a', '~~', 'a')]);
  });

  it('rejects the symmetric duplicate b ~~ a after a ~~ b', () => {
    expect(() => parseModel('a ~~ b\nb ~~ a')).toThrow(/duplicate/i);
    expect(() => parseModel('a ~~ b\nb ~~ a')).toThrow(/b ~~ a/);
  });
});

describe('parseModel: fixed values and NA*', () => {
  it('fixes the premultiplied term only: f =~ 1*x1 + x2', () => {
    expect(parseModel('f =~ 1*x1 + x2')).toEqual([
      row('f', '=~', 'x1', 1),
      row('f', '=~', 'x2'),
    ]);
  });

  it('handles a fractional fixed covariance: a ~~ 0.5*b', () => {
    expect(parseModel('a ~~ 0.5*b')).toEqual([row('a', '~~', 'b', 0.5)]);
  });

  it('handles negative fixed values: f =~ -1*x3', () => {
    expect(parseModel('f =~ x1 + -1*x3')).toEqual([
      row('f', '=~', 'x1'),
      row('f', '=~', 'x3', -1),
    ]);
  });

  it('NA* marks the term explicitly free: fixed null, freed true', () => {
    expect(parseModel('f =~ NA*x1 + x2')).toEqual([
      row('f', '=~', 'x1', null, true),
      row('f', '=~', 'x2', null, false),
    ]);
  });
});

describe('parseModel: whitespace, comments, separators', () => {
  it('parses the compact whitespace-free form', () => {
    expect(parseModel('f=~x1+x2')).toEqual([row('f', '=~', 'x1'), row('f', '=~', 'x2')]);
    expect(parseModel('y~x')).toEqual([row('y', '~', 'x')]);
    expect(parseModel('a~~b')).toEqual([row('a', '~~', 'b')]);
    expect(parseModel('f=~1*x1+NA*x2')).toEqual([
      row('f', '=~', 'x1', 1),
      row('f', '=~', 'x2', null, true),
    ]);
  });

  it('ignores comments and blank lines', () => {
    const rows = parseModel(`
      # measurement model
      f =~ x1 + x2  # first factor

      # nothing on this line either
      y ~ x1
    `);
    expect(rows).toEqual([row('f', '=~', 'x1'), row('f', '=~', 'x2'), row('y', '~', 'x1')]);
  });

  it('treats ";" as a statement separator', () => {
    const rows = parseModel('f =~ x1 + x2; y ~ f; a ~~ b');
    expect(rows).toEqual([
      row('f', '=~', 'x1'),
      row('f', '=~', 'x2'),
      row('y', '~', 'f'),
      row('a', '~~', 'b'),
    ]);
  });

  it('returns an empty array for empty or comment-only input', () => {
    expect(parseModel('')).toEqual([]);
    expect(parseModel('# just a comment\n\n;;\n')).toEqual([]);
  });
});

describe('parseModel: errors', () => {
  it('rejects statements with no operator, quoting the line', () => {
    expect(() => parseModel('x1 x2')).toThrow(/operator/);
    expect(() => parseModel('x1 x2')).toThrow(/x1 x2/);
  });

  it('rejects statements with multiple operators', () => {
    expect(() => parseModel('y ~ x ~ z')).toThrow(/[Mm]ultiple operators/);
    expect(() => parseModel('f =~ x1 ~~ x2')).toThrow(/[Mm]ultiple operators/);
  });

  it('rejects a missing left-hand side', () => {
    expect(() => parseModel('=~ x1 + x2')).toThrow(/left-hand side/);
    expect(() => parseModel('~ x1')).toThrow(/left-hand side/);
  });

  it('rejects a missing right-hand side', () => {
    expect(() => parseModel('y ~')).toThrow(/right-hand side/);
    expect(() => parseModel('f =~ ')).toThrow(/right-hand side/);
  });

  it('rejects an invalid left-hand-side name', () => {
    expect(() => parseModel('2f =~ x1')).toThrow(/Invalid name "2f"/);
  });

  it('rejects a malformed term like 2x', () => {
    expect(() => parseModel('f =~ 2x')).toThrow(/[Mm]alformed term "2x"/);
    expect(() => parseModel('f =~ 2x')).toThrow(/f =~ 2x/);
  });

  it('rejects a term with a missing premultiplier: *x1', () => {
    expect(() => parseModel('f =~ *x1')).toThrow(/missing premultiplier/);
  });

  it('rejects a term with a missing name: 1*', () => {
    expect(() => parseModel('f =~ 1*')).toThrow(/missing variable name/);
  });

  it('rejects a double premultiplier: a*b*x', () => {
    expect(() => parseModel('f =~ a*b*x')).toThrow(/multiple premultipliers/);
  });

  it('rejects a non-numeric, non-NA premultiplier', () => {
    expect(() => parseModel('f =~ lam*x1')).toThrow(/not a number or NA/);
  });

  it('rejects an empty term from a stray "+"', () => {
    expect(() => parseModel('f =~ x1 + + x2')).toThrow(/empty term/);
    expect(() => parseModel('y ~ x1 +')).toThrow(/empty term/);
  });

  it("rejects '+' on either side of '~~', suggesting one pair per statement", () => {
    expect(() => parseModel('a ~~ b + c')).toThrow(/one pair per statement/);
    expect(() => parseModel('a + b ~~ c')).toThrow(/one pair per statement/);
  });

  it('rejects duplicate identical parameters, mentioning "duplicate"', () => {
    expect(() => parseModel('y ~ x\ny ~ x')).toThrow(/duplicate/i);
    expect(() => parseModel('f =~ x1 + x1')).toThrow(/duplicate/i);
    expect(() => parseModel('f =~ x1\nf =~ 1*x1')).toThrow(/duplicate/i);
  });

  it('rejects non-string input', () => {
    expect(() => parseModel(null)).toThrow(/expects a string/);
  });
});
