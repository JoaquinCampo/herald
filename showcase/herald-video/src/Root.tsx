import {Composition} from 'remotion';
import {HeraldFilm} from './HeraldFilm';

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;
export const DURATION_FRAMES = 2550;

export const RemotionRoot = () => (
  <Composition
    id="HeraldOpenAIShowcase"
    component={HeraldFilm}
    durationInFrames={DURATION_FRAMES}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
  />
);
