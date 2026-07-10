import {describe, expect, it} from 'vitest';
import {DURATION_FRAMES, FPS, HEIGHT, WIDTH} from '../Root';

describe('HERALD film contract', () => {
  it('is an 85-second 1080p composition at 30fps', () => {
    expect({DURATION_FRAMES, FPS, WIDTH, HEIGHT}).toEqual({
      DURATION_FRAMES: 2550,
      FPS: 30,
      WIDTH: 1920,
      HEIGHT: 1080,
    });
  });
});
