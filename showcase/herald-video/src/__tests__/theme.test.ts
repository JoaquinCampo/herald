import {expect, it} from 'vitest';
import {COLORS, SAFE_X, SAFE_Y} from '../theme';

it('uses the approved palette and 1080p safe area', () => {
  expect(COLORS).toEqual({
    alabaster: '#F3EFE6',
    graphite: '#171918',
    blue: '#5D78A6',
    coral: '#D9634F',
    sage: '#91A797',
    paper: '#FFFDF8',
  });
  expect({SAFE_X, SAFE_Y}).toEqual({SAFE_X: 120, SAFE_Y: 86});
});
