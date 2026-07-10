import '@fontsource/dm-serif-display/400.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/ibm-plex-mono/400.css';
import {Easing} from 'remotion';

export const COLORS = {
  alabaster: '#F3EFE6',
  graphite: '#171918',
  blue: '#5D78A6',
  coral: '#D9634F',
  sage: '#91A797',
  paper: '#FFFDF8',
} as const;

export const FONT = {
  display: 'DM Serif Display',
  sans: 'Inter',
  mono: 'IBM Plex Mono',
} as const;

export const SAFE_X = 120;
export const SAFE_Y = 86;
export const EASE = Easing.bezier(0.16, 1, 0.3, 1);
