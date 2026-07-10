import {expect, it} from 'vitest';
import {ON_SCREEN_COPY} from '../scenes/copy';

it('does not overclaim customer status or memory savings', () => {
  const all = ON_SCREEN_COPY.join(' ').toLowerCase();
  expect(all).not.toContain('openai uses');
  expect(all).not.toContain('deployed at openai');
  expect(all).not.toContain('memory savings');
  expect(all).not.toContain('zero quality loss');
});
