import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, EASE, FONT} from '../theme';

export type CacheStackProps = {
  compression: number;
  heldInReserve: boolean;
  alarmed: boolean;
};

const LAYERS = 8;
export const CACHE_RESERVE_COPY = 'uncompressed cache held in reserve';

export const CacheStack = ({compression, heldInReserve, alarmed}: CacheStackProps) => {
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
        <span style={{fontFamily: FONT.mono, fontSize: 20, letterSpacing: '0.08em'}}>KV CACHE</span>
        <span style={{fontFamily: FONT.display, fontSize: 70, lineHeight: 0.95}}>
          {Math.round(compression * 100)}%
        </span>
        <span style={{fontFamily: FONT.sans, fontSize: 25, color: accent}}>
          {heldInReserve ? CACHE_RESERVE_COPY : alarmed ? 'unsafe shift detected' : 'compression active'}
        </span>
      </div>
    </div>
  );
};
