import {expect, it} from 'vitest';
import {
  CACHE_RESERVE_COPY,
  formatCacheAmount,
  getCacheMetricLabel,
} from '../components/CacheStack';
import {formatCompressedFraction, formatQualityCost} from '../components/MetricLane';

it('formats evidence metrics with their correct units', () => {
  expect(formatCompressedFraction(0.809)).toBe('80.9%');
  expect(formatQualityCost(0.0027)).toBe('0.27 pp');
});

it('describes the uncompressed cache as the reserve', () => {
  expect(CACHE_RESERVE_COPY).toBe('uncompressed cache held in reserve');
  expect(CACHE_RESERVE_COPY).not.toContain('token');
});

it('can express compression as the amount of cache retained', () => {
  expect(getCacheMetricLabel('retained')).toBe('KV CACHE RETAINED');
  expect(formatCacheAmount(0, 'retained')).toBe('100%');
  expect(formatCacheAmount(0.75, 'retained')).toBe('25%');
  expect(formatCacheAmount(0.75, 'compression')).toBe('75%');
});
