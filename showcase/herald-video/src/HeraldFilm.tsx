import {Audio} from '@remotion/media';
import {AbsoluteFill, Sequence, staticFile} from 'remotion';
import {HiddenCostScene} from './scenes/HiddenCostScene';
import {FailureScene} from './scenes/FailureScene';
import {MechanismScene} from './scenes/MechanismScene';
import {SignalScene} from './scenes/SignalScene';
import {ProofScene} from './scenes/ProofScene';
import {OpenAIFrameScene} from './scenes/OpenAIFrameScene';
import {EndCardScene} from './scenes/EndCardScene';
import {SCENES} from './scenes/timeline';
import {COLORS} from './theme';
import narrationCues from './data/narration-cues.json';

const SCENE_COMPONENTS = {
  'hidden-cost': HiddenCostScene,
  failure: FailureScene,
  signal: SignalScene,
  mechanism: MechanismScene,
  proof: ProofScene,
  'openai-frame': OpenAIFrameScene,
  'end-card': EndCardScene,
} as const;

export const HeraldFilm = () => (
  <AbsoluteFill style={{background: COLORS.alabaster}}>
    <Audio src={staticFile('audio/score.wav')} volume={0.12} trimAfter={2548} />
    {narrationCues.map((cue) => (
      <Sequence key={cue.id} from={cue.from} durationInFrames={cue.duration} premountFor={30}>
        <Audio src={staticFile(cue.file)} volume={1.3} />
      </Sequence>
    ))}
    {SCENES.map((scene) => {
      const Scene = SCENE_COMPONENTS[scene.id];
      return (
        <Sequence
          key={scene.id}
          from={scene.from}
          durationInFrames={scene.duration}
          premountFor={scene.from === 0 ? 0 : 30}
        >
          <Scene />
        </Sequence>
      );
    })}
  </AbsoluteFill>
);
