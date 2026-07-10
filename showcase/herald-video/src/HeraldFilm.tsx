import {Audio} from '@remotion/media';
import {AbsoluteFill, Sequence, staticFile} from 'remotion';
import {HiddenCostScene} from './scenes/HiddenCostScene';
import {FailureScene} from './scenes/FailureScene';
import {MechanismScene} from './scenes/MechanismScene';
import {SignalScene} from './scenes/SignalScene';
import {ProofScene} from './scenes/ProofScene';
import {OpenAIFrameScene} from './scenes/OpenAIFrameScene';
import {EndCardScene} from './scenes/EndCardScene';
import {COLORS} from './theme';
import narrationCues from './data/narration-cues.json';

export const HeraldFilm = () => (
  <AbsoluteFill style={{background: COLORS.alabaster}}>
    <Audio src={staticFile('audio/score.wav')} volume={0.12} />
    {narrationCues.map((cue) => (
      <Sequence key={cue.id} from={cue.from} durationInFrames={cue.duration} premountFor={30}>
        <Audio src={staticFile(cue.file)} volume={1.5} />
      </Sequence>
    ))}
    <Sequence from={0} durationInFrames={210} premountFor={30}>
      <HiddenCostScene />
    </Sequence>
    <Sequence from={210} durationInFrames={330} premountFor={30}>
      <FailureScene />
    </Sequence>
    <Sequence from={540} durationInFrames={450} premountFor={30}>
      <SignalScene />
    </Sequence>
    <Sequence from={990} durationInFrames={570} premountFor={30}>
      <MechanismScene />
    </Sequence>
    <Sequence from={1560} durationInFrames={600} premountFor={30}>
      <ProofScene />
    </Sequence>
    <Sequence from={2160} durationInFrames={270} premountFor={30}>
      <OpenAIFrameScene />
    </Sequence>
    <Sequence from={2430} durationInFrames={120} premountFor={30}>
      <EndCardScene />
    </Sequence>
  </AbsoluteFill>
);
