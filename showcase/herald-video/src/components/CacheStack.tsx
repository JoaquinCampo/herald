import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, EASE, FONT} from '../theme';

export type CacheStackProps = {
  compression: number;
  heldInReserve: boolean;
  alarmed: boolean;
  metricMode?: CacheMetricMode;
};

export type CacheMetricMode = 'compression' | 'retained';

const LAYERS = 8;
export const CACHE_RESERVE_COPY = 'uncompressed cache held in reserve';

export const getCacheMetricLabel = (metricMode: CacheMetricMode) =>
  metricMode === 'retained' ? 'KV CACHE RETAINED' : 'KV CACHE';

export const formatCacheAmount = (compression: number, metricMode: CacheMetricMode) => {
  const fraction = metricMode === 'retained' ? 1 - compression : compression;
  return `${Math.round(fraction * 100)}%`;
};

export const CacheStack = ({
  compression,
  heldInReserve,
  alarmed,
  metricMode = 'compression',
}: CacheStackProps) => {
  const frame = useCurrentFrame();
  const retained = Math.max(1, Math.round(LAYERS * (1 - compression)));
  const accent = alarmed ? COLORS.coral : COLORS.blue;

  return (
    <div style={{display: 'flex', alignItems: 'center', gap: 36}}>
      <div style={{display: 'flex', width: 310, flexDirection: 'column-reverse', gap: 10}}>
        {Array.from({length: LAYERS}, (_, index) => {
          const isRetained = index < retained;
          return (
            <div
              key={index}
              style={{
                height: 34,
                border: `2px solid ${isRetained ? accent : `${COLORS.graphite}22`}`,
                backgroundColor: isRetained ? `${accent}16` : 'transparent',
                opacity: interpolate(frame, [index * 2, index * 2 + 14], [0, isRetained ? 1 : 0.38], {
                  easing: EASE,
                  extrapolateLeft: 'clamp',
                  extrapolateRight: 'clamp',
                }),
                translate: `${interpolate(frame, [index * 2, index * 2 + 14], [-22, 0], {
                  easing: EASE,
                  extrapolateLeft: 'clamp',
                  extrapolateRight: 'clamp',
                })}px 0`,
              }}
            />
          );
        })}
      </div>
      <div style={{display: 'flex', minWidth: 210, flexDirection: 'column', gap: 13}}>
        <span style={{fontFamily: FONT.mono, fontSize: 20, letterSpacing: '0.08em'}}>
          {getCacheMetricLabel(metricMode)}
        </span>
        <span style={{fontFamily: FONT.display, fontSize: 70, lineHeight: 0.95}}>
          {formatCacheAmount(compression, metricMode)}
        </span>
        <span style={{fontFamily: FONT.sans, fontSize: 25, color: accent}}>
          {heldInReserve ? CACHE_RESERVE_COPY : alarmed ? 'unsafe shift detected' : 'compression active'}
        </span>
      </div>
    </div>
  );
};
