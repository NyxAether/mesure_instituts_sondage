// La population (points a et b), la mise en évidence d'un échantillon et l'envol des personnes tirées vers l'axe.
import React from 'react';
import {X, Y} from '../../lib/axes';
import {avance, echelonne, lerp, UNITE} from '../../lib/outils';
import {D, DUREE_LENT, T, debutsRapides} from './temps';
import {D1} from './donnees';
import {AXE_CENTRE, HAUT_PLACE, POP_RAYON, couleurVote, pointPop, xAxe} from './geometrie';

const N = D1.colonnes * D1.lignes;
/** Grossissement des personnes tirées pendant un tirage lent. */
const GROSSIT = 1.8;
const FLEUR = {cible: 0.4, opacite: 0.6}; // réduction et opacité à l'arrivée sur l'axe
const DECALAGE_LENT = 0.004; // lag_ratio des envols lents
const DECALAGE_RAPIDE = 0.003; // lag_ratio des envols rapides
const DECALAGE_ENTREE = 0.0015; // lag_ratio de l'apparition de la population
const DEPART_ECHELLE = 0.3; // les points grandissent depuis 0,3 à leur apparition

/** Le tirage lent en cours à l'instant t : son numéro et le temps écoulé depuis son début (null hors des tirages lents). */
export const lentEnCours = (t: number) => {
	const u = t - T.lents;
	if (u < 0 || u >= D1.nb_lents * DUREE_LENT) return null;
	const i = Math.floor(u / DUREE_LENT);
	return {i, u: u - i * DUREE_LENT};
};

/** Cible des envols (unités) pour un résultat de tirage v. */
const cible = (v: number): [number, number] => [xAxe(v), Y(AXE_CENTRE[1] + HAUT_PLACE)];

export const Population: React.FC<{t: number}> = ({t}) => {
	const sortie = 1 - avance(t, T.populationSortie, D.populationSortie);
	if (t < T.population || sortie <= 0) return null;
	const lent = lentEnCours(t);
	const choisis = lent ? new Set(D1.echantillons[lent.i]) : null;
	// Atténuation du reste de la population pendant qu'un échantillon est mis en évidence.
	const dim = lent ? avance(lent.u, 0, D.lentChoix) - avance(lent.u, D.lentChoix + D.lentPause, D.lentEnvol) : 0;
	const r = POP_RAYON * UNITE;
	const points: React.ReactNode[] = [];
	for (let i = 0; i < D1.lignes; i++) {
		for (let j = 0; j < D1.colonnes; j++) {
			const idx = i * D1.colonnes + j;
			const e = echelonne(t, T.population, D.population, N, idx, DECALAGE_ENTREE);
			const [cx, cy] = pointPop(i, j);
			const estChoisi = choisis?.has(idx) ?? false;
			const grossi = estChoisi ? lerp(1, GROSSIT, dim) : 1;
			points.push(
				<circle
					key={idx}
					cx={X(cx)}
					cy={Y(cy)}
					r={r * lerp(DEPART_ECHELLE, 1, e) * grossi}
					fill={couleurVote(D1.votes[idx])}
					opacity={e * sortie * (choisis && !estChoisi ? lerp(1, 0.15, dim) : 1)}
				/>,
			);
		}
	}
	return <g>{points}</g>;
};

/** Envol de personnes vers l'axe : n copies qui partent de leur point, se réduisent et s'éclaircissent. */
const Envol: React.FC<{indices: readonly number[]; v: number; t: number; debut: number; duree: number; decalage: number; depart: number}> = ({
	indices,
	v,
	t,
	debut,
	duree,
	decalage,
	depart,
}) => {
	const [cx, cy] = cible(v);
	return (
		<g>
			{indices.map((idx, k) => {
				const e = echelonne(t, debut, duree, indices.length, k, decalage);
				const [ox, oy] = pointPop(Math.floor(idx / D1.colonnes), idx % D1.colonnes);
				return (
					<circle
						key={idx}
						cx={lerp(X(ox), cx, e)}
						cy={lerp(Y(oy), cy, e)}
						r={POP_RAYON * UNITE * depart * lerp(1, FLEUR.cible, e)}
						fill={couleurVote(D1.votes[idx])}
						opacity={lerp(1, FLEUR.opacite, e)}
					/>
				);
			})}
		</g>
	);
};

/** Envols du tirage lent ou rapide en cours (rien sinon). */
export const Envols: React.FC<{t: number}> = ({t}) => {
	const lent = lentEnCours(t);
	if (lent && lent.u >= D.lentChoix + D.lentPause && lent.u < D.lentChoix + D.lentPause + D.lentEnvol) {
		return (
			<Envol
				indices={D1.echantillons[lent.i]}
				v={D1.t1000[lent.i]}
				t={lent.u}
				debut={D.lentChoix + D.lentPause}
				duree={D.lentEnvol}
				decalage={DECALAGE_LENT}
				depart={GROSSIT}
			/>
		);
	}
	for (let j = 0; j < D1.durees_rapides.length; j++) {
		if (t >= debutsRapides[j] && t < debutsRapides[j] + D1.durees_rapides[j]) {
			const k = D1.nb_lents + j;
			return <Envol indices={D1.echantillons[k]} v={D1.t1000[k]} t={t} debut={debutsRapides[j]} duree={D1.durees_rapides[j]} decalage={DECALAGE_RAPIDE} depart={1} />;
		}
	}
	return null;
};
