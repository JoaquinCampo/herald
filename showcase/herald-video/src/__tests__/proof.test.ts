import {expect, it} from 'vitest';
import {formatPercent, formatQualityPoints, proofCopy} from '../scenes/ProofScene';

it('formats live metrics without renaming them as memory savings', () => {
  expect(formatPercent(0.8092, 1)).toBe('80.9%');
  expect(formatQualityPoints(0.0027)).toBe('0.27 pp');
  expect(proofCopy.toLowerCase()).toContain('compressed-generation fraction');
  expect(proofCopy.toLowerCase()).toContain('live-internal quality cost');
  expect(proofCopy.toLowerCase()).toContain('reverted-attempt wall overhead');
  expect(proofCopy.toLowerCase()).not.toContain('memory savings');
});
