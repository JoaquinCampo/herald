import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS, EASE, FONT} from '../theme';

export type TokenStreamProps = {
  tokens: string[];
  state: 'stable' | 'unstable' | 'probe' | 'committed';
  revealFrame?: number;
};

const stateColor = {
  stable: COLORS.graphite,
  unstable: COLORS.coral,
  probe: COLORS.blue,
  committed: COLORS.sage,
} as const;

export const TokenStream = ({tokens, state, revealFrame = 0}: TokenStreamProps) => {
  const frame = useCurrentFrame();
  const accent = stateColor[state];

  return (
    <div style={{display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 15}}>
      {tokens.map((token, index) => {
        const start = revealFrame + index * 3;
        return (
          <span
            key={`${token}-${index}`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              minHeight: 68,
              padding: '0 18px',
              borderBottom: `3px solid ${accent}`,
              backgroundColor: state === 'probe' ? `${COLORS.blue}12` : 'transparent',
              color: state === 'unstable' && index >= tokens.length - 2 ? COLORS.coral : COLORS.graphite,
              fontFamily: FONT.mono,
              fontSize: 33,
              letterSpacing: '-0.02em',
              opacity: interpolate(frame, [start, start + 12], [0, 1], {
                easing: EASE,
                extrapolateLeft: 'clamp',
                extrapolateRight: 'clamp',
              }),
              translate: `${interpolate(frame, [start, start + 12], [18, 0], {
                easing: EASE,
                extrapolateLeft: 'clamp',
                extrapolateRight: 'clamp',
              })}px 0`,
            }}
          >
            {token}
          </span>
        );
      })}
    </div>
  );
};
