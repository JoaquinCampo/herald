import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, EASE, FONT} from '../theme';

export type MetricLaneProps = {
  name: string;
  fraction: number;
  qualityCost: number;
  overhead: number;
  delay: number;
};

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

export const MetricLane = ({name, fraction, qualityCost, overhead, delay}: MetricLaneProps) => {
  const frame = useCurrentFrame();
  const animationEnd = delay + 22;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.4fr 3fr 0.8fr 0.8fr',
        alignItems: 'center',
        gap: 28,
        minHeight: 90,
        borderTop: `1px solid ${COLORS.graphite}30`,
        fontFamily: FONT.sans,
        opacity: interpolate(frame, [delay, animationEnd], [0, 1], {
          easing: EASE,
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        }),
        translate: `0 ${interpolate(frame, [delay, animationEnd], [18, 0], {
          easing: EASE,
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        })}px`,
      }}
    >
      <span style={{fontSize: 29, fontWeight: 500}}>{name}</span>
      <div style={{height: 12, backgroundColor: `${COLORS.graphite}12`}}>
        <div
          style={{
            width: `${Math.min(1, Math.max(0, fraction)) * 100}%`,
            height: '100%',
            backgroundColor: COLORS.blue,
            transformOrigin: 'left center',
            scale: `${interpolate(frame, [delay, animationEnd + 10], [0, 1], {
              easing: EASE,
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            })} 1`,
          }}
        />
      </div>
      <span style={{fontFamily: FONT.mono, fontSize: 23, color: COLORS.coral}}>
        {formatPercent(qualityCost)}
      </span>
      <span style={{fontFamily: FONT.mono, fontSize: 23}}>{formatPercent(overhead)}</span>
    </div>
  );
};
