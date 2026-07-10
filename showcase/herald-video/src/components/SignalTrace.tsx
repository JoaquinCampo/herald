import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, EASE, FONT} from '../theme';

export type SignalTraceProps = {
  label: string;
  values: number[];
  accent: 'blue' | 'coral' | 'sage';
  progress: number;
};

const WIDTH = 640;
const HEIGHT = 170;

export const SignalTrace = ({label, values, accent, progress}: SignalTraceProps) => {
  const frame = useCurrentFrame();
  const color = COLORS[accent];
  const safeValues = values.length > 1 ? values : [values[0] ?? 0, values[0] ?? 0];
  const min = Math.min(...safeValues);
  const max = Math.max(...safeValues);
  const range = max - min || 1;
  const points = safeValues
    .map((value, index) => {
      const x = (index / (safeValues.length - 1)) * WIDTH;
      const y = HEIGHT - 18 - ((value - min) / range) * (HEIGHT - 36);
      return `${x},${y}`;
    })
    .join(' ');
  const revealed = Math.min(1, Math.max(0, progress));

  return (
    <div style={{display: 'flex', width: WIDTH, flexDirection: 'column', gap: 14}}>
      <div style={{display: 'flex', alignItems: 'baseline', justifyContent: 'space-between'}}>
        <span style={{fontFamily: FONT.sans, fontSize: 29, fontWeight: 500}}>{label}</span>
        <span style={{fontFamily: FONT.mono, fontSize: 18, color}}>LIVE SIGNAL</span>
      </div>
      <svg width={WIDTH} height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
        <line x1="0" x2={WIDTH} y1={HEIGHT - 1} y2={HEIGHT - 1} stroke={`${COLORS.graphite}2A`} />
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
          pathLength="1"
          strokeDasharray="1"
          strokeDashoffset={1 - revealed * interpolate(frame, [0, 18], [0, 1], {
            easing: EASE,
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          })}
        />
      </svg>
    </div>
  );
};
