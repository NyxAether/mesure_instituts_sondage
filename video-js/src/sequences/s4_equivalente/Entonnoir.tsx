// L'entonnoir de la séquence 2 (écrit ici en attendant la déduplication), élargi jusqu'à contenir 95 % des écarts.
import React from 'react';
import donnees from '../../../donnees/s4.json';
import {echelle} from '../../lib/axes';
import {C, G, MONO, SERIF} from '../../lib/charte';
import {Titre} from '../../lib/composants';
import {F, NBSP, Txt, UNITE as U, X, Y, avance, echelonne, fr, lerp} from '../../lib/outils';
import {DISCRET, Fondu, LIBELLE, SERIE, SOURCE_Y, TRAIT, largeurLibelle, apparition, centreAuDessus, centreSous, chemin, entier, signeEntier} from './commun';
import {ENTONNOIR as E} from './temps';

const Z95 = donnees.z95;
const CIBLE = donnees.cible;
const Y_MAX = 14; // points ; les écarts exportés sont bornés à ± 14
const GRADUATIONS_N = [300, 1_000, 3_000, 10_000, 30_000, 100_000];
const GRADUATIONS_Y = [-10, -5, 0, 5, 10];
const N_MIN = 200;
const N_MAX = 150_000;
const NB_BORD = 80; // points du bord de l'entonnoir

const pts = donnees.points;
if (pts.length !== donnees.nb_lignes) throw new Error('nombre de lignes différent de celui de la page');
const horsMarge = pts.filter((p) => Math.abs(p.residu) > Z95 * p.sigma).length / pts.length;
if (Math.abs(horsMarge - donnees.part_hors_marge) > 1e-4) throw new Error('part hors marge différente de la page');
const tailleReelle = donnees.median_reel;
const facteur95 = donnees.facteur_95;
const etapes = [...donnees.etapes, facteur95];
if (etapes.length !== E.etapes.length) throw new Error("le nombre d'étapes ne correspond pas au minutage");

// Repère : cadre de 10,6 × 4,4 unités centré en (0,55 ; −0,95) (Axes de Manim), abscisse en échelle log.
const echX = echelle([N_MIN, N_MAX], [0.55 - 5.3, 0.55 + 5.3], true);
const echY = echelle([-Y_MAX, Y_MAX], [-0.95 - 2.2, -0.95 + 2.2]);
const ecrete = (v: number) => Math.max(-Y_MAX, Math.min(Y_MAX, v));
const position = (taille: number, ecart: number): [number, number] => [X(echX(taille)), Y(echY(ecrete(ecart)))];
const bordN = Array.from({length: NB_BORD}, (_, k) => N_MIN * Math.pow(N_MAX / N_MIN, k / (NB_BORD - 1)));

/** Facteur de division de la taille, au temps t (1 : entonnoir de la taille annoncée). */
const facteurA = (t: number) => {
	let f = 1;
	etapes.forEach((cible, k) => {
		f = lerp(f, cible, avance(t, E.etapes[k], E.dureeEtape));
	});
	return f;
};
const marge50 = (taille: number, f: number) => 100 * Z95 * Math.sqrt((0.25 * f) / taille);
const dedans = (f: number) => pts.filter((p) => Math.abs(p.residu) <= Z95 * p.sigma * Math.sqrt(f)).length / pts.length;

const gauche = X(echX(N_MIN));
const hautY = Y(echY(Y_MAX));
const basY = Y(echY(-Y_MAX));
const largeurLabY = Math.max(...GRADUATIONS_Y.map((y) => largeurLibelle(signeEntier(y), 14)));
const titreY = 'écart sondage − résultat (points)';
const xTitreY = gauche - 0.15 * U - largeurLabY; // aligné à gauche sur les graduations
const yTitreY = centreAuDessus(hautY, 0.15, 14);
const nMin = GRADUATIONS_N[0];
const nMax = GRADUATIONS_N[GRADUATIONS_N.length - 1];
const centreX = (X(echX(nMin)) - largeurLibelle(fr(nMin, 0), 14) / 2 + X(echX(nMax)) + largeurLibelle(fr(nMax, 0), 14) / 2) / 2;
const coin = position(N_MAX, Y_MAX);

export const Entonnoir: React.FC<{t: number}> = ({t}) => {
	const f = facteurA(t);
	const graduations = apparition(t, E.axes, 0.8);
	const forme = apparition(t, E.forme, 1.5);

	// Zone de l'entonnoir et contour, tracé seulement dans le cadre (à partir de la taille où la marge atteint le bord).
	const marges = bordN.map((n) => Math.min(marge50(n, f), Y_MAX));
	const zone = chemin([...bordN.map((n, k) => position(n, marges[k])), ...bordN.map((n, k) => position(n, -marges[k])).reverse()], true);
	const nBord = 0.25 * f * Math.pow((100 * Z95) / Y_MAX, 2);
	const tailles = [Math.max(nBord, bordN[0]), ...bordN.filter((n) => n > nBord)];
	const contour = (s: number) => chemin(tailles.map((n) => position(n, s * Math.min(marge50(n, f), Y_MAX))));

	const cadre = f > 1.05 ? `entonnoir d’un sondage ${fr(f, 0)} fois plus petit` : 'entonnoir de la taille annoncée';
	const xCadre = xTitreY + largeurLibelle(titreY, 14) + 0.6 * U;
	const equivalent = tailleReelle / facteur95;
	const [xa, xb] = [position(N_MIN, 0)[0], position(N_MAX, 0)[0]];

	return (
		<div style={{position: 'absolute', inset: 0, opacity: 1 - avance(t, E.sortie, 0.8)}}>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				<g opacity={graduations}>
					{[-10, -5, 5, 10].map((y) => (
						<line key={y} x1={xa} x2={xb} y1={position(N_MIN, y)[1]} y2={position(N_MIN, y)[1]} stroke={G.grid} strokeWidth={TRAIT(1)} />
					))}
				</g>
				<g opacity={forme}>
					<path d={zone} fill={C.accent} fillOpacity={0.15} />
					<path d={contour(1)} fill="none" stroke={C.accent} strokeWidth={TRAIT(2)} />
					<path d={contour(-1)} fill="none" stroke={C.accent} strokeWidth={TRAIT(2)} />
				</g>
				<line x1={xa} x2={xb} y1={position(N_MIN, 0)[1]} y2={position(N_MIN, 0)[1]} stroke={G.axis} strokeWidth={TRAIT(1.5)} opacity={graduations} />
				{pts.map((p, k) => {
					const hors = Math.abs(p.residu) > marge50(p.n, f);
					const [x, y] = position(p.n, p.residu);
					return (
						<circle
							key={k}
							cx={x}
							cy={y}
							r={0.03 * U}
							fill={hors ? SERIE : DISCRET}
							opacity={(hors ? 1 : 0.55) * echelonne(t, E.forme, 1.5, pts.length, k, 0.001)}
						/>
					);
				})}
			</svg>
			<Fondu opacite={graduations}>
				{GRADUATIONS_N.map((v) => (
					<Txt key={v} x={X(echX(v))} y={centreSous(basY, 0.15, 14)} ax={0.5} ay={0.5} taille={14} couleur={LIBELLE}>
						{fr(v, 0)}
					</Txt>
				))}
				{GRADUATIONS_Y.map((y) => (
					<Txt key={y} x={gauche - 0.15 * U} y={position(N_MIN, y)[1]} ax={1} ay={0.5} taille={14} couleur={LIBELLE}>
						{signeEntier(y)}
					</Txt>
				))}
				<Txt x={centreX} y={centreSous(basY, 0.3, 14) + 0.73 * F(14) + 0.035 * U} ax={0.5} ay={0.5} taille={14} couleur={DISCRET}>
					taille de l’échantillon (échelle log)
				</Txt>
				<Txt x={xTitreY} y={yTitreY} ay={0.5} taille={14} couleur={DISCRET}>
					{titreY}
				</Txt>
			</Fondu>
			<Fondu opacite={apparition(t, E.source, 0.6)}>
				<Txt x={0.55 * U} y={SOURCE_Y} taille={14} couleur={DISCRET}>
					{`les mêmes ${fr(donnees.nb_lignes, 0)} intentions de vote qu’à la séquence 2 · dernière semaine avant le vote`}
				</Txt>
			</Fondu>
			<Fondu opacite={apparition(t, E.compteur, 0.6)}>
				<div style={{position: 'absolute', right: 1920 - coin[0], top: coin[1] + 0.1 * U, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 0.08 * U}}>
					<div style={{fontFamily: MONO, fontSize: F(15), lineHeight: 1, color: LIBELLE, whiteSpace: 'pre'}}>écarts dans leur marge</div>
					<div style={{fontFamily: SERIF, fontSize: F(56), lineHeight: 0.85, whiteSpace: 'pre', color: C['text-primary']}}>
						<Titre mot={`${fr(100 * dedans(f))}${NBSP}%`} />
					</div>
					<div style={{fontFamily: MONO, fontSize: F(15), lineHeight: 1, color: DISCRET, whiteSpace: 'pre'}}>{`visé : ${fr(100 * CIBLE, 0)}${NBSP}%`}</div>
				</div>
			</Fondu>
			<Fondu opacite={apparition(t, E.cadre, 0.5)}>
				<Txt x={xCadre} y={yTitreY} ay={0.5} taille={15} couleur={C['text-primary']}>
					{cadre}
				</Txt>
			</Fondu>
			<Fondu opacite={apparition(t, E.verdict, 0.8)} glisse={0.1} avancement={avance(t, E.verdict, 0.8)}>
				<div
					style={{
						position: 'absolute',
						right: 1920 - coin[0] - 0.2 * U,
						top: position(N_MAX, -6)[1] - 0.15 * U - 3,
						padding: `${0.15 * U}px ${0.2 * U}px`,
						background: C['bg-primary'],
						display: 'flex',
						flexDirection: 'column',
						alignItems: 'flex-end',
						gap: 0.1 * U,
					}}
				>
					<div style={{fontFamily: MONO, fontSize: F(15), lineHeight: 1, color: C['text-primary'], whiteSpace: 'pre'}}>
						{`un sondage de ${fr(tailleReelle, 0)} personnes a l’entonnoir d’un tirage de`}
					</div>
					<div style={{fontFamily: SERIF, fontSize: F(48), lineHeight: 1, whiteSpace: 'pre', color: C['text-primary']}}>
						<Titre mot={`${fr(entier(equivalent), 0)} personnes`} />
					</div>
				</div>
			</Fondu>
		</div>
	);
};
