import {expect, it} from 'vitest';
import {evidence} from '../data/evidence';

it('loads only artifact-backed campaign evidence', () => {
  expect(evidence.campaign.episode_count).toBe(552);
  expect(evidence.gsm8k_example).toMatchObject({
    prompt_id: 'gsm8k-0',
    reference_answer: '18',
    compressed_answer: '3',
  });
});
