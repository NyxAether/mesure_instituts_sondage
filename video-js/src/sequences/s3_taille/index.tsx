// Séquence 3 : l'excédent d'erreur au-dessus de la théorie ne diminue pas avec la taille de l'échantillon.
// Port de video/scenes/s3_taille.py (classe Taille) ; données : donnees/s3.json (export/exporter_s3.py).
import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import donnees from '../../../donnees/s3.json';
import {C, G, SERIF} from '../../lib/charte';
import {Entete, Titre} from '../../lib/composants';
import {F, FPS, NBSP, Txt, UNITE as U, X, Y, avance, echelonne, fr, largeurMono, lerp} from '../../lib/outils';
import {Grille} from './Grille';
import {
	BAS, DROITE, GAUCHE, GRADUATIONS_N, HAUT, HAUTEUR, N_MIN, TRAIT, Y_ECART, Y_MOYENNE,
	membres, nMed, obs, residu, taille, th, tranches, xu, yu,
} from './repere';
import {T} from './temps';

const SERIE = G.series[0];
const TEXTE_2 = C['text-secondary'];
const DISCRET = C['text-muted'];
const N = residu.length;
const nbTranches = tranches.length;

const exces = obs.map((o, k) => o - th[k]);
const excesMin = Math.min(...exces);
const excesMax = Math.max(...exces);

// Tranche de chaque point.
const trancheDe = new Map<number, number>();
membres.forEach((m, k) => m.forEach((j) => trancheDe.set(j, k)));

// Positions (unités) des points de l'axe des moyennes, une fois le zoom fait.
const sommetsTh = tranches.map((_, k): [number, number] => [xu(nMed[k]), yu(th[k], 0, Y_MOYENNE)]);
const sommetsObs = tranches.map((_, k): [number, number] => [xu(nMed[k]), yu(obs[k], 0, Y_MOYENNE)]);

const ligne = (a: [number, number], b: [number, number], u: number): [number, number] => [lerp(a[0], b[0], u), lerp(a[1], b[1], u)];
const px = (p: [number, number]) => `${X(p[0])},${Y(p[1])}`;
const borne = (v: number) => Math.min(1, Math.max(0, v));
const longueur = (a: [number, number], b: [number, number]) => Math.hypot(b[0] - a[0], b[1] - a[1]);

/** Titre d'axe : le coin gauche du texte à 0,45 unité à gauche de l'axe, 0,15 unité au-dessus du cadre. */
const TitreAxe: React.FC<{opacite: number; children: string}> = ({opacite, children}) => (
	<Txt x={X(GAUCHE - 0.45)} y={Y(HAUT + 0.15)} ay={1} taille={14} couleur={DISCRET} opacite={opacite}>
		{children}
	</Txt>
);

/** Étiquette à côté d'un point : ancre (ax, ay) de la boîte du texte posée sur le point décalé de (dx, dy) unités, y vers le haut. */
const Etiquette: React.FC<{p: [number, number]; dx: number; dy: number; ax: number; ay: number; opacite: number; couleur: string; children: string}> = ({p, dx, dy, ax, ay, opacite, couleur, children}) => (
	<Txt x={X(p[0] + dx)} y={Y(p[1] + dy)} ax={ax} ay={ay} taille={15} couleur={couleur} opacite={opacite}>
		{children}
	</Txt>
);

/** Les 1 553 écarts : pliés vers le haut, puis réduits chacun à la moyenne de sa tranche. */
const Nuage: React.FC<{t: number}> = ({t}) => {
	if (t < T.nuage || t >= T.reduction + 2) return null; // ensuite les 7 moyennes prennent le relais
	const pliage = avance(t, T.pliage, 1.8);
	const reduction = avance(t, T.reduction, 2);
	const couleur = `color-mix(in srgb, ${DISCRET} ${(1 - reduction) * 100}%, ${SERIE})`;
	const rayon = lerp(0.03, 0.075, reduction) * U;
	const opaque = lerp(0.55, 1, reduction);
	return (
		<>
			{residu.map((e, j) => {
				const e14 = Math.min(Y_ECART, Math.max(-Y_ECART, e));
				const k = trancheDe.get(j)!;
				const y0 = lerp(yu(e14, -Y_ECART, Y_ECART), yu(Math.abs(e14), 0, Y_ECART), pliage);
				const x = lerp(xu(taille[j]), xu(nMed[k]), reduction);
				const y = lerp(y0, yu(obs[k], 0, Y_ECART), reduction);
				return <circle key={j} cx={X(x)} cy={Y(y)} r={rayon} fill={couleur} fillOpacity={opaque * echelonne(t, T.nuage, 1.5, N, j, 0.001)} />;
			})}
		</>
	);
};

export const Taille: React.FC = () => {
	const frame = useCurrentFrame();
	const t = frame / FPS;

	// Bornes de l'axe vertical, animées comme les ValueTrackers `bas` (−14 → 0 au pliage) et `haut` (14 → 2,6 au zoom).
	const bas = lerp(-Y_ECART, 0, avance(t, T.pliage, 1.8));
	const haut = lerp(Y_ECART, Y_MOYENNE, avance(t, T.zoom, 2.5));
	const graphique = 1 - avance(t, T.sortie, 0.8); // tout ce qui s'efface avant l'écran final
	const visible = avance(t, T.axes, 0.8) * graphique;
	const apparition = (debut: number, duree: number) => avance(t, debut, duree) * graphique;

	// Les points de moyenne suivent l'axe pendant le zoom.
	const moyennes = nMed.map((_, k): [number, number] => [xu(nMed[k]), yu(obs[k], bas, haut)]);

	// Courbe pointillée attendue (DashedLine de Manim : dash_length 0,08 et moitié pleine, donc une période de 0,16) : les tirets se dessinent l'un après l'autre, sur toute la courbe.
	const tiretsTh = sommetsTh.slice(1).map((b, k) => Math.max(2, Math.ceil((longueur(sommetsTh[k], b) / 0.08) * 0.5)));
	const dessinTh = avance(t, T.theorie, 1.8) * tiretsTh.reduce((s, w) => s + w, 0);
	// Courbe observée : chaque segment prend la même part de la durée.
	const dessinObs = avance(t, T.observe, 1.5) * (nbTranches - 1);
	const trace: [number, number][] = [sommetsObs[0]];
	for (let k = 1; k < nbTranches; k++) {
		const u = borne(dessinObs - (k - 1));
		if (u > 0) trace.push(ligne(sommetsObs[k - 1], sommetsObs[k], u));
	}

	const retraitExces = avance(t, T.exces, 1.5);
	const rapport = (k: number, cote: 'gauche' | 'droite', debut: number) => {
		const a = avance(t, debut, 0.7);
		const mx = (sommetsObs[k][0] + sommetsTh[k][0]) / 2;
		const my = (sommetsObs[k][1] + sommetsTh[k][1]) / 2;
		return (
			<Txt
				x={X(mx + (cote === 'gauche' ? -0.35 : 0.35))}
				y={Y(my) + (1 - a) * 0.1 * U}
				ax={cote === 'gauche' ? 1 : 0}
				ay={0.5}
				taille={40}
				police={SERIF}
				couleur={C['text-primary']}
				opacite={a * (1 - retraitExces)}
			>
				<span style={{fontStyle: 'italic', color: C.accent}}>{`×${NBSP}${fr(obs[k] / th[k])}`}</span>
			</Txt>
		);
	};

	const labelsX = GRADUATIONS_N.map((v) => ({v, largeur: largeurMono(fr(v, 0), 14)}));
	const gaucheLabelsX = Math.min(...labelsX.map(({v, largeur}) => X(xu(v)) - largeur / 2));
	const droiteLabelsX = Math.max(...labelsX.map(({v, largeur}) => X(xu(v)) + largeur / 2));
	const pied = Y(BAS - 0.15) + F(14) + 0.15 * U;

	return (
		<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
			{/* En-tête, puis la source des données */}
			<div style={{position: 'absolute', inset: 0, transform: `translateY(${-(1 - avance(t, T.entete, 1)) * 0.15 * U}px)`, opacity: avance(t, T.entete, 1)}}>
				<Entete numero={3} nom="la taille" avant="L’erreur selon la " mot="taille" apres={`${NBSP}de l’échantillon`} />
			</div>
			<Txt x={0.55 * U} y={0.55 * U + F(17) + 0.18 * U + F(52)} taille={14} couleur={DISCRET} opacite={apparition(T.source, 0.6)}>
				{`les mêmes ${fr(donnees.nb_lignes, 0)} intentions de vote · regroupées en ${nbTranches} tranches de taille`}
			</Txt>

			<Grille bas={bas} haut={haut} visible={visible} />

			<div style={{position: 'absolute', inset: 0, opacity: graphique}}>
				{/* Titres d'axe et graduations de taille */}
				<TitreAxe opacite={avance(t, T.axes, 0.8) * (1 - avance(t, T.pliage, 1.8))}>écart sondage − résultat (points)</TitreAxe>
				<TitreAxe opacite={avance(t, T.pliage, 1.8) * (1 - avance(t, T.retrait, 0.5))}>erreur, sans son signe (points)</TitreAxe>
				<TitreAxe opacite={avance(t, T.zoom, 2.5)}>erreur moyenne de la tranche (points)</TitreAxe>
				{GRADUATIONS_N.map((v) => (
					<Txt key={v} x={X(xu(v))} y={Y(BAS - 0.15)} ax={0.5} taille={14} couleur={TEXTE_2} opacite={avance(t, T.axes, 0.8)}>
						{fr(v, 0)}
					</Txt>
				))}
				<Txt x={(gaucheLabelsX + droiteLabelsX) / 2} y={pied} ax={0.5} taille={14} couleur={DISCRET} opacite={avance(t, T.axes, 0.8)}>
					taille de l’échantillon (échelle log)
				</Txt>

				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					{/* Tranches de taille (une sur deux est grisée) */}
					{tranches.map((tr, k) => {
						if (k % 2 !== 0) return null;
						const gauche = xu(Math.max(tr.n_min, N_MIN));
						const droite = xu(tr.n_max);
						const largeur = Math.max(droite - gauche, 0.04);
						return (
							<rect
								key={k}
								x={X((gauche + droite) / 2 - largeur / 2)}
								y={Y(HAUT)}
								width={largeur * U}
								height={HAUTEUR * U}
								fill={C['text-primary']}
								fillOpacity={0.06 * avance(t, T.bandes, 0.8) * (1 - avance(t, T.retrait, 0.5))}
							/>
						);
					})}

					{/* Aire entre les deux courbes */}
					<polygon
						points={[...sommetsObs, ...[...sommetsTh].reverse()].map(px).join(' ')}
						fill={SERIE}
						fillOpacity={lerp(0.12, 0.05, retraitExces) * avance(t, T.ecart, 1)}
					/>

					{/* Courbe attendue */}
					{sommetsTh.slice(1).map((b, k) => {
						const a = sommetsTh[k];
						const debut = tiretsTh.slice(0, k).reduce((s, w) => s + w, 0);
						const u = borne((dessinTh - debut) / tiretsTh[k]);
						if (u <= 0) return null;
						const fin = ligne(a, b, u);
						const periode = (longueur(a, b) * U) / tiretsTh[k];
						return (
							<line key={k} x1={X(a[0])} y1={Y(a[1])} x2={X(fin[0])} y2={Y(fin[1])} stroke={TEXTE_2} strokeWidth={3 * TRAIT} strokeDasharray={`${periode / 2} ${periode / 2}`} />
						);
					})}
					{sommetsTh.map((s, k) => (
						<circle key={k} cx={X(s[0])} cy={Y(s[1])} r={0.06 * U} fill={TEXTE_2} opacity={avance(t, T.theorie, 1.8)} />
					))}

					{/* Courbe observée, puis le nuage et les moyennes par-dessus */}
					{trace.length > 1 && <polyline points={trace.map(px).join(' ')} fill="none" stroke={SERIE} strokeWidth={4 * TRAIT} strokeLinejoin="round" />}
					<Nuage t={t} />
					{t >= T.reduction + 2 && moyennes.map((m, k) => <circle key={k} cx={X(m[0])} cy={Y(m[1])} r={0.075 * U} fill={SERIE} />)}

					{/* Écarts verticaux entre les deux courbes */}
					{sommetsTh.map((b, k) => {
						const u = echelonne(t, T.exces, 1.5, nbTranches, k, 0.15);
						if (u <= 0) return null;
						const fin = ligne(b, sommetsObs[k], u);
						return <line key={k} x1={X(b[0])} y1={Y(b[1])} x2={X(fin[0])} y2={Y(fin[1])} stroke={SERIE} strokeWidth={5 * TRAIT} />;
					})}
				</svg>

				{/* Étiquettes des courbes */}
				<Etiquette p={sommetsTh[nbTranches - 1]} dx={0} dy={-0.25} ax={0.5} ay={0} couleur={TEXTE_2} opacite={avance(t, T.etiquettesTheorie, 0.8)}>
					attendu en théorie
				</Etiquette>
				<Etiquette p={sommetsTh[0]} dx={-0.1} dy={-0.1} ax={1} ay={0} couleur={TEXTE_2} opacite={avance(t, T.etiquettesTheorie, 0.8)}>
					{fr(th[0])}
				</Etiquette>
				<Etiquette p={sommetsTh[nbTranches - 1]} dx={0.15} dy={0} ax={0} ay={0.5} couleur={TEXTE_2} opacite={avance(t, T.etiquettesTheorie, 0.8)}>
					{fr(th[nbTranches - 1])}
				</Etiquette>
				<Etiquette p={sommetsObs[nbTranches - 1]} dx={0} dy={0.25} ax={0.5} ay={1} couleur={SERIE} opacite={avance(t, T.etiquettesObserve, 0.8)}>
					observé
				</Etiquette>
				<Etiquette p={sommetsObs[0]} dx={-0.1} dy={0.1} ax={1} ay={1} couleur={SERIE} opacite={avance(t, T.etiquettesObserve, 0.8)}>
					{fr(obs[0])}
				</Etiquette>
				<Etiquette p={sommetsObs[nbTranches - 1]} dx={0.15} dy={0} ax={0} ay={0.5} couleur={SERIE} opacite={avance(t, T.etiquettesObserve, 0.8)}>
					{fr(obs[nbTranches - 1])}
				</Etiquette>

				{/* Rapports observé / attendu aux deux extrémités */}
				{rapport(0, 'gauche', T.rapportGauche)}
				{rapport(nbTranches - 1, 'droite', T.rapportDroit)}

				{/* Ce qui dépasse la théorie */}
				<div
					style={{
						position: 'absolute',
						left: X(DROITE),
						top: Y(yu(2.45, 0, Y_MOYENNE)),
						transform: 'translate(-100%, -50%)',
						display: 'flex',
						flexDirection: 'column',
						alignItems: 'flex-end',
						gap: 0.1 * U,
						opacity: avance(t, T.etiquetteExces, 0.7),
					}}
				>
					{[`${fr(excesMin)} à ${fr(excesMax)} point au-dessus de la théorie`, 'à toutes les tailles'].map((texte) => (
						<Txt key={texte} x={0} y={0} taille={15} couleur={SERIE} style={{position: 'relative', transform: 'none'}}>
							{texte}
						</Txt>
					))}
				</div>
			</div>

			{/* Le constat */}
			<div
				style={{
					position: 'absolute',
					left: 960,
					top: Y(-0.4),
					transform: 'translate(-50%, -50%)',
					display: 'flex',
					flexDirection: 'column',
					alignItems: 'center',
					gap: 0.35 * U,
					whiteSpace: 'pre',
					fontFamily: SERIF,
				}}
			>
				<div
					style={{
						fontSize: F(64),
						lineHeight: 1,
						color: C['text-primary'],
						opacity: avance(t, T.constat, 1),
						transform: `translateY(${(1 - avance(t, T.constat, 1)) * 0.15 * U}px)`,
					}}
				>
					<Titre avant="L’excédent d’erreur " mot="ne diminue pas" />
				</div>
				<div style={{fontSize: F(30), lineHeight: 1, color: C['text-primary'], opacity: avance(t, T.detail, 0.6)}}>
					{`${fr(excesMin)} à ${fr(excesMax)} point au-dessus de la théorie, à toutes les tailles`}
				</div>
				<div style={{fontSize: F(28), lineHeight: 1, color: TEXTE_2, opacity: avance(t, T.conclusion, 0.6)}}>
					{`seule la baisse prévue par le hasard a lieu${NBSP}: −${fr(obs[0] - obs[nbTranches - 1])} point, contre −${fr(th[0] - th[nbTranches - 1])} en théorie`}
				</div>
			</div>
		</AbsoluteFill>
	);
};

export const dureeTaille = Math.round(T.fin * FPS);
