import {interpolate, useCurrentFrame} from 'remotion';
import {CacheStack} from '../components/CacheStack';
import {Stage} from '../components/Stage';
import {TokenStream} from '../components/TokenStream';
import {COLORS, EASE, FONT} from '../theme';
import {ON_SCREEN_COPY} from './copy';

const reveal = (frame: number, start: number) =>
  interpolate(frame, [start, start + 18], [0, 1], {
    easing: EASE,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

export const HiddenCostScene = () => {
  const frame = useCurrentFrame();
  const compression = interpolate(frame, [84, 198], [0, 0.75], {
    easing: EASE,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const isUnstable = frame >= 145;

  return (
    <Stage style={{flexDirection: 'column', justifyContent: 'space-between'}}>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 0.9fr', alignItems: 'start', gap: 96}}>
        <div style={{display: 'flex', flexDirection: 'column', gap: 2}}>
          {ON_SCREEN_COPY.slice(0, 3).map((line, index) => {
            const progress = reveal(frame, index * 38);
            return (
              <div
                key={line}
                style={{
                  color: index === 2 ? COLORS.coral : COLORS.graphite,
                  fontFamily: FONT.display,
                  fontSize: 94,
                  lineHeight: 1.02,
                  letterSpacing: '-0.035em',
                  opacity: progress,
                  translate: `0 ${interpolate(progress, [0, 1], [22, 0])}px`,
                }}
              >
                {line}
              </div>
            );
          })}
        </div>

        <div style={{display: 'flex', justifyContent: 'flex-end', paddingTop: 32}}>
          <CacheStack compression={compression} heldInReserve={false} alarmed={isUnstable} />
        </div>
      </div>

      <div style={{display: 'flex', flexDirection: 'column', gap: 24}}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontFamily: FONT.mono,
            fontSize: 20,
            letterSpacing: '0.09em',
          }}
        >
          <span>CAUSAL TOKEN STREAM</span>
          <span style={{color: isUnstable ? COLORS.coral : COLORS.blue}}>
            {isUnstable ? 'STATE DRIFT' : 'STABLE'}
          </span>
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: 15}}>
          <TokenStream
            tokens={['The', 'answer', 'follows', 'from', 'the', 'problem', 'state']}
            state="stable"
            revealFrame={20}
          />
          {isUnstable ? <TokenStream tokens={['boxes']} state="unstable" revealFrame={145} /> : null}
        </div>
      </div>
    </Stage>
  );
};
