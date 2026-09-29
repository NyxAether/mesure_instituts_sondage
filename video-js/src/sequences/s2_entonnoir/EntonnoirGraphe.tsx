// Briques réutilisables du graphe en entonnoir (séquences 2 et 4) : axes à échelle logarithmique en abscisse,
// entonnoir de la marge d'erreur, nuage de points (taille de l'échantillon, écart au résultat) et compteur.
// Rien ici ne dépend du temps : chaque brique reçoit ses avancements (0 à 1) en props.
import React from 'react';
import {C, G, SERIF} from '../../lib/charte';
import {Titre} from '../../lib/composants';
import {echelle} from '../../lib/axes';
import {F, NBSP, Txt, UNITE as U, X, Y, fr, largeurMono, lerp} from '../../lib/outils';

// Un stroke_width de Manim vaut 0,01 unité, soit 1,35 px à 135 px par unité.
export const TRAIT = (largeur: number) => (largeur * U) / 100;

export const GRADUATIONS_N = [300, 1_000, 3_000, 10_000, 30_000, 100_000];
export const GRADUATIONS_ECART = [-10, -5, 0, 5, 10];
export const Y_MAX = 14; // points : les écarts sont bornés à ± Y_MAX à l'affichage
const RAYON_POINT = 0.03; // unités, sur le nuage

/** Repère du graphe : plage des tailles (échelle log) et des écarts, placés dans le cadre (unités Manim). */
export const creerRepere = ({
	centre = [0.55, -0.95],
	largeur = 10.6,
	hauteur = 4.4,
	tailleMin = 200,
	tailleMax = 150_000,
	ecartMax = Y_MAX,
}: {centre?: [number, number]; largeur?: number; hauteur?: number; tailleMin?: number; tailleMax?: number; ecartMax?: number} = {}) => {
	const ex = echelle([tailleMin, tailleMax], [centre[0] - largeur / 2, centre[0] + largeur / 2], true);
	const ey = echelle([-ecartMax, ecartMax], [centre[1] - hauteur / 2, centre[1] + hauteur / 2]);
	/** Écart ramené dans [-ecartMax, ecartMax], comme le `np.clip` de la scène Manim. */
	const borne = (ecart: number) => Math.min(ecartMax, Math.max(-ecartMax, ecart));
	return {
		tailleMin,
		tailleMax,
		ecartMax,
		borne,
		/** Abscisse en px d'une taille d'échantillon. */
		x: (taille: number) => X(ex(taille)),
		/** Ordonnée en px d'un écart (borné). */
		y: (ecart: number) => Y(ey(borne(ecart))),
		gauche: X(ex(tailleMin)),
		droite: X(ex(tailleMax)),
		haut: Y(ey(ecartMax)),
		bas: Y(ey(-ecartMax)),
	};
};
export type Repere = ReturnType<typeof creerRepere>;

/** « +5 », « −5 », « 0 » : écart entier signé, avec le vrai signe moins. */
export const signeEntier = (v: number) => (v > 0 ? `+${fr(v, 0)}` : v === 0 ? '0' : `−${fr(-v, 0)}`);

/** Chemin SVG partiellement tracé : `trace` de 0 à 1 (pathLength normalisé, comme `Create` de Manim). */
export const Trace: React.FC<{d: string; trace: number; stroke: string; largeur: number; opacite?: number}> = ({
	d,
	trace,
	stroke,
	largeur,
	opacite = 1,
}) =>
	trace <= 0 ? null : (
		<path
			d={d}
			pathLength={1}
			fill="none"
			stroke={stroke}
			strokeWidth={largeur}
			strokeLinecap="butt"
			strokeDasharray={`${trace} 1`}
			opacity={opacite}
		/>
	);

/**
 * Axes : grille, ligne du zéro (tracée de gauche à droite), graduations et titres.
 * `apparition` fait entrer grille et textes ; `trace` dessine la ligne du zéro. `children` (éléments SVG) vient entre la grille et le zéro.
 */
export const AxesLog: React.FC<{
	repere: Repere;
	apparition: number;
	trace: number;
	titreX: string;
	titreY: string;
	children?: React.ReactNode;
}> = ({repere, apparition, trace, titreX, titreY, children}) => {
	const {x, y, gauche, droite, haut, bas} = repere;
	const couleurTexte = C['text-secondary'];
	const largeurEtiquetteY = Math.max(...GRADUATIONS_ECART.map((v) => largeurMono(signeEntier(v), 14)));
	const bordGauche = gauche - 0.15 * U - largeurEtiquetteY; // bord gauche du groupe des graduations verticales
	// Graduations horizontales : groupe centré, titre juste dessous.
	const nMin = GRADUATIONS_N[0], nMax = GRADUATIONS_N[GRADUATIONS_N.length - 1];
	const gaucheGrad = x(nMin) - largeurMono(fr(nMin, 0), 14) / 2;
	const droiteGrad = x(nMax) + largeurMono(fr(nMax, 0), 14) / 2;
	const hautGrad = bas + 0.15 * U;
	const hautTitreX = hautGrad + F(14) + 0.15 * U;
	return (
		<>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0, overflow: 'visible'}}>
				<g opacity={apparition}>
					{GRADUATIONS_ECART.filter((v) => v !== 0).map((v) => (
						<line key={v} x1={gauche} x2={droite} y1={y(v)} y2={y(v)} stroke={G.grid} strokeWidth={TRAIT(1)} />
					))}
				</g>
				{children}
				<Trace d={`M${gauche},${y(0)} L${droite},${y(0)}`} trace={trace} stroke={G.axis} largeur={TRAIT(1.5)} />
			</svg>
			{GRADUATIONS_N.map((v) => (
				<Txt key={v} x={x(v)} y={hautGrad} ax={0.5} taille={14} couleur={couleurTexte} opacite={apparition}>
					{fr(v, 0)}
				</Txt>
			))}
			{GRADUATIONS_ECART.map((v) => (
				<Txt key={v} x={gauche - 0.15 * U} y={y(v)} ax={1} ay={0.5} taille={14} couleur={couleurTexte} opacite={apparition}>
					{signeEntier(v)}
				</Txt>
			))}
			<Txt x={(gaucheGrad + droiteGrad) / 2} y={hautTitreX} ax={0.5} taille={14} couleur={C['text-muted']} opacite={apparition}>
				{titreX}
			</Txt>
			<Txt x={bordGauche} y={haut - 0.15 * U} ay={1} taille={14} couleur={C['text-muted']} opacite={apparition}>
				{titreY}
			</Txt>
		</>
	);
};

/** Point d'un nuage : `hors` (0 à 1) fait passer du gris discret à la couleur de série ; `apparition` grossit le point. */
export type PointNuage = {taille: number; ecart: number; hors: number; apparition: number};

/** Nuage de points de taille (échelle log) et d'écart, un cercle par point. */
export const Nuage: React.FC<{repere: Repere; points: PointNuage[]; opacite?: number}> = ({repere, points, opacite = 1}) => (
	<svg width={1920} height={1080} style={{position: 'absolute', inset: 0, overflow: 'visible', opacity: opacite}}>
		{points.map((p, k) =>
			p.apparition <= 0 ? null : (
				<circle
					key={k}
					cx={repere.x(p.taille)}
					cy={repere.y(p.ecart)}
					r={RAYON_POINT * U * lerp(0.3, 1, p.apparition)}
					fill={p.hors >= 1 ? G.series[0] : p.hors <= 0 ? C['text-muted'] : `color-mix(in srgb, ${G.series[0]} ${p.hors * 100}%, ${C['text-muted']})`}
					fillOpacity={lerp(0.55, 1, p.hors) * p.apparition}
				/>
			),
		)}
	</svg>
);

/**
 * Entonnoir : zone et contour de la marge d'erreur (± `marges[i].m` points à la taille `marges[i].taille`), élargi de `facteur`.
 * `apparition` fond la zone, `trace` dessine les deux contours l'un après l'autre. À placer dans les `children` d'`AxesLog`.
 */
export const Entonnoir: React.FC<{
	repere: Repere;
	marges: {taille: number; m: number}[];
	facteur?: number;
	apparition: number;
	trace: number;
}> = ({repere, marges, facteur = 1, apparition, trace}) => {
	const haut = marges.map((b) => [repere.x(b.taille), repere.y(b.m * facteur)] as const);
	const bas = marges.map((b) => [repere.x(b.taille), repere.y(-b.m * facteur)] as const);
	const chemin = (pts: (readonly [number, number])[]) => pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px},${py}`).join(' ');
	return (
		<g>
			<polygon
				points={[...haut, ...[...bas].reverse()].map(([px, py]) => `${px},${py}`).join(' ')}
				fill={C.accent}
				fillOpacity={0.15 * apparition}
			/>
			<Trace d={chemin(haut)} trace={Math.min(1, trace * 2)} stroke={C.accent} largeur={TRAIT(2)} />
			<Trace d={chemin(bas)} trace={Math.max(0, trace * 2 - 1)} stroke={C.accent} largeur={TRAIT(2)} />
		</g>
	);
};

/** Légende de l'entonnoir : petit carré teinté + texte, calée à droite sur le bord droit du repère, à l'écart `ecart`. */
export const LegendeEntonnoir: React.FC<{repere: Repere; ecart: number; texte: string; opacite: number}> = ({repere, ecart, texte, opacite}) => {
	const carre = 0.2 * U;
	const largeur = carre + 0.15 * U + largeurMono(texte, 14);
	const x0 = repere.droite - largeur;
	return (
		<div style={{position: 'absolute', left: 0, top: 0, opacity: opacite}}>
			<div
				style={{
					position: 'absolute',
					left: x0,
					top: repere.y(ecart) - carre / 2,
					width: carre,
					height: carre,
					boxSizing: 'border-box',
					border: `${TRAIT(2)}px solid ${C.accent}`,
					background: `color-mix(in srgb, ${C.accent} 15%, transparent)`,
				}}
			/>
			<Txt x={x0 + carre + 0.15 * U} y={repere.y(ecart)} ay={0.5} taille={14} couleur={C['text-secondary']}>
				{texte}
			</Txt>
		</div>
	);
};

/** Compteur en haut à droite du repère : libellé, puis grand pourcentage en italique prune. `pourcent` en points de pourcentage. */
export const Compteur: React.FC<{repere: Repere; libelle: string; pourcent: number; opacite: number}> = ({repere, libelle, pourcent, opacite}) => (
	<div
		style={{
			position: 'absolute',
			right: 1920 - repere.droite,
			top: repere.haut + 0.1 * U,
			display: 'flex',
			flexDirection: 'column',
			alignItems: 'flex-end',
			gap: 0.08 * U,
			opacity: opacite,
		}}
	>
		<Txt x={0} y={0} taille={15} couleur={C['text-secondary']} style={{position: 'static'}}>
			{libelle}
		</Txt>
		<div style={{fontFamily: SERIF, fontSize: F(56), lineHeight: 1, whiteSpace: 'pre', color: C['text-primary']}}>
			<Titre mot={`${fr(pourcent)}${NBSP}%`} />
		</div>
	</div>
);
