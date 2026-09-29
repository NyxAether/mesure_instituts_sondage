// Échelles des graphes, à la place des `Axes` de Manim : plage de données → pixels de l'écran (repère outils.tsx).
import {scaleLinear, scaleLog} from 'd3-scale';
import {X, Y} from './outils';

/** Échelle linéaire ou logarithmique d'un axe : `[d0, d1]` en données, `[u0, u1]` en unités Manim. */
export const echelle = (domaine: [number, number], unites: [number, number], log = false) => {
	const s = log ? scaleLog().domain(domaine) : scaleLinear().domain(domaine);
	return s.range(unites);
};

export {X, Y};
