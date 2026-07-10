import {interpolate, useCurrentFrame} from 'remotion';
import {Stage} from '../components/Stage';
import {COLORS, EASE, FONT} from '../theme';

const clamp = {
  easing: EASE,
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

export const EndCardScene = () => {
  const frame = useCurrentFrame();
  const reveal = interpolate(frame, [0, 22], [0, 1], clamp);
  const cursorOpacity = Math.round(frame / 18) % 2 === 0 ? 1 : 0.25;

  return (
    <Stage style={{alignItems: 'center', justifyContent: 'center'}}>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 24,
          opacity: reveal,
          translate: `0 ${interpolate(reveal, [0, 1], [18, 0])}px`,
        }}
      >
        <div style={{display: 'flex', alignItems: 'baseline', gap: 18}}>
          <span style={{fontFamily: FONT.display, fontSize: 162, letterSpacing: '-0.07em', lineHeight: 0.85}}>HERALD</span>
          <span style={{width: 5, height: 106, background: COLORS.blue, opacity: cursorOpacity}} />
        </div>
        <span style={{fontFamily: FONT.sans, fontSize: 34, letterSpacing: '-0.02em'}}>Compression, with an undo button.</span>
      </div>
    </Stage>
  );
};
