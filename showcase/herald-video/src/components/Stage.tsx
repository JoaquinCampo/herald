import type {CSSProperties, ReactNode} from 'react';
import {AbsoluteFill} from 'remotion';
import {COLORS, SAFE_X, SAFE_Y} from '../theme';

export type StageProps = {
  children: ReactNode;
  style?: CSSProperties;
};

export const Stage = ({children, style}: StageProps) => (
  <AbsoluteFill
    style={{
      backgroundColor: COLORS.alabaster,
      color: COLORS.graphite,
      overflow: 'hidden',
    }}
  >
    <AbsoluteFill
      aria-hidden="true"
      style={{
        opacity: 0.015,
        pointerEvents: 'none',
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.88' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
      }}
    />
    <div
      style={{
        position: 'relative',
        display: 'flex',
        flex: 1,
        padding: `${SAFE_Y}px ${SAFE_X}px`,
        ...style,
      }}
    >
      {children}
    </div>
  </AbsoluteFill>
);
