// Charte rr/ : couleurs, polices et courbe d'animation lues dans video/rr-tokens.json, jamais recopiées.
import {loadFont} from '@remotion/fonts';
import {Easing, staticFile} from 'remotion';
import tokens from '../../../video/rr-tokens.json';
import {papier} from '../genere/papier.json';

// Thème : REMOTION_THEME=dark pour le sombre (variables préfixées REMOTION_ seules visibles du navigateur).
const THEME = process.env.REMOTION_THEME === 'dark' ? 'dark' : 'light';

export const C = tokens.color[THEME];
export const G = tokens.chart[THEME];
export const SERIF = tokens.font.serif.stack;
export const MONO = tokens.font.mono.stack;
// Police manuscrite de Windows pour les étiquettes écrites à la main, comme dans la version Manim ; à défaut, la serif.
export const MAIN = `'Ink Free', ${SERIF}`;
export const RAYON = tokens.radius.md;
// Le papier des coupures : l'exception déjà en place, tirée de video/outils/decoupe_portraits.py (ressources.mjs).
export const PAPIER = papier;

const [x1, y1, x2, y2] = tokens.motion.ease.match(/[\d.]+/g)!.map(Number);
export const EASE = Easing.bezier(x1, y1, x2, y2);

export const couleurs = {
	fillon: G.series[0],
	juppe: G.series[1],
	sarkozy: G['mark-muted'],
	macron: G.series[1],
	lepen: G.series[0],
};
// Le gris du graphe est trop pâle pour une écriture : celui des étiquettes est plus soutenu.
export const encre = {...couleurs, sarkozy: `color-mix(in srgb, ${G['mark-muted']} 65%, ${C['text-primary']})`};

export const chargerPolices = () => {
	const nom = (stack: string) => stack.split(',')[0].replaceAll("'", '');
	loadFont({family: nom(SERIF), url: staticFile('fonts/Newsreader-400.ttf'), weight: '400', style: 'normal'});
	loadFont({family: nom(SERIF), url: staticFile('fonts/Newsreader-400-Italic.ttf'), weight: '400', style: 'italic'});
	loadFont({family: nom(MONO), url: staticFile('fonts/JetBrainsMono-400.ttf'), weight: '400', style: 'normal'});
};
