import {expect, it} from 'vitest';
import {CACHE_RESERVE_COPY} from '../components/CacheStack';
import {formatCompressedFraction, formatQualityCost} from '../components/MetricLane';

it('formats evidence metrics with their correct units', () => {
  expect(formatCompressedFraction(0.809)).toBe('80.9%');
  expect(formatQualityCost(0.0027)).toBe('0.27 pp');
});

it('describes the uncompressed cache as the reserve', () => {
  expect(CACHE_RESERVE_COPY).toBe('uncompressed cache held in reserve');
  expect(CACHE_RESERVE_COPY).not.toContain('token');
});
