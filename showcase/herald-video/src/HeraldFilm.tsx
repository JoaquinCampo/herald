import {AbsoluteFill, Sequence} from 'remotion';
import {HiddenCostScene} from './scenes/HiddenCostScene';
import {FailureScene} from './scenes/FailureScene';
import {COLORS} from './theme';

export const HeraldFilm = () => (
  <AbsoluteFill style={{background: COLORS.alabaster}}>
    <Sequence from={0} durationInFrames={210} premountFor={30}>
      <HiddenCostScene />
    </Sequence>
    <Sequence from={210} durationInFrames={330} premountFor={30}>
      <FailureScene />
    </Sequence>
  </AbsoluteFill>
);
