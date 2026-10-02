// Axe gradué, vraie valeur, traits des tirages, barres de l'histogramme et bande à 95 %.
import React from 'react';
import {C, G, SERIF} from '../../lib/charte';
import {F, NBSP, Txt, UNITE, avance, fr, lerp} from '../../lib/outils';
import {X, Y} from '../../lib/axes';
import {D, DUREE_LENT, T, debutsRapides} from './temps';
import {D1, HAUTEUR, M_GRAND, M_PETIT, bornes, centre, comptes, t1000, t4000, tranche, unite} from './donnees';
import {AXE_CENTRE, AXE_LONGUEUR, HAUT_TRAIT, couleurCote, trait, xAxe, yAxe} from './geometrie';

const borne = (x: number) => Math.min(1, Math.max(0, x));
const ECART_BARRES = 0.04; // espace entre deux barres voisines (unités)
const BARRE_VIDE = 0.002; // hauteur d'une barre sans tirage (unités)
const OPACITE_ESTOMPEE = 0.3; // barres hors de la marge
const MEL = (a: string, b: string, u: number) => `color-mix(in srgb, ${a} ${(1 - u) * 100}%, ${b})`;

// --- Axe ------------------------------------------------------------------
const GRADUATIONS = Array.from({length: (bornes[bornes.length - 1] - bornes[0]) / 1 + 1}, (_, i) => bornes[0] + i); // tous les points de %
const GRADUATIONS_ECRITES = GRADUATIONS.filter((v) => (v - bornes[0]) % 2 === 0); // un libellé tous les 2 points
const TAILLE_TICK = 0.05; // demi-longueur d'une graduation (unités)
const HAUT_VERITE = HAUTEUR + 0.55; // hauteur de la ligne de la vraie valeur (unités)
const TAILLE_LIBELLE = 15;
// JetBrains Mono : le haut de la boîte CSS (line-height 1) est à 0,13 em au-dessus du haut des chiffres, qui font 0,73 em.
const HAUT_MONO = 0.13;
const CAPITALE_MONO = 0.73;

/** Axe, graduations et vraie valeur (partie graphique). Create(axe) dessine le trait puis les graduations une à une. */
export const AxeSvg: React.FC<{t: number}> = ({t}) => {
	if (t < T.axe) return null;
	const v = avance(t, T.axe, D.axe) * (GRADUATIONS.length + 1);
	const tick = TAILLE_TICK * UNITE;
	const eVerite = avance(t, T.verite, D.verite);
	const x0 = xAxe(bornes[0]);
	const longueur = AXE_LONGUEUR * UNITE;
	return (
		<g>
			<line x1={x0} x2={x0 + longueur * borne(v)} y1={yAxe} y2={yAxe} stroke={G.axis} strokeWidth={trait(2)} />
			{GRADUATIONS.map((g, j) => (
				<line key={g} x1={xAxe(g)} x2={xAxe(g)} y1={yAxe + tick} y2={yAxe + tick - 2 * tick * borne(v - 1 - j)} stroke={G.axis} strokeWidth={trait(2)} />
			))}
			{eVerite > 0 && (
				<line
					x1={xAxe(50)}
					x2={xAxe(50)}
					y1={yAxe}
					y2={yAxe - HAUT_VERITE * UNITE * eVerite}
					stroke={C['text-secondary']}
					strokeWidth={trait(1.5)}
					strokeDasharray={`${0.08 * UNITE} ${0.08 * UNITE}`}
				/>
			)}
		</g>
	);
};

/** Libellés de l'axe : graduations, titre, vraie valeur. */
export const AxeTexte: React.FC<{t: number}> = ({t}) => {
	const eAxe = avance(t, T.axe, D.axe);
	const eVerite = avance(t, T.verite, D.verite);
	// La vraie valeur s'efface avec le compteur quand la bande apparaît.
	const oVerite = eVerite * (1 - avance(t, T.zone, D.zone));
	return (
		<>
			{GRADUATIONS_ECRITES.map((v) => (
				<Txt key={v} x={xAxe(v)} y={Y(AXE_CENTRE[1] - 0.18) - HAUT_MONO * F(TAILLE_LIBELLE)} taille={TAILLE_LIBELLE} ax={0.5} couleur={C['text-secondary']} opacite={eAxe}>
					{`${v}${NBSP}%`}
				</Txt>
			))}
			<Txt x={xAxe(50)} y={Y(AXE_CENTRE[1] - 0.18) + (CAPITALE_MONO - HAUT_MONO) * F(TAILLE_LIBELLE) + 0.2 * UNITE + 4} taille={TAILLE_LIBELLE} ax={0.5} couleur={C['text-muted']} opacite={eAxe}>
				résultat du tirage (part du vote a)
			</Txt>
			<Txt x={xAxe(50)} y={Y(AXE_CENTRE[1] + HAUT_VERITE + 0.1)} taille={TAILLE_LIBELLE} ax={0.5} ay={1} couleur={C['text-secondary']} opacite={oVerite}>
				{`vraie valeur · 50${NBSP}%`}
			</Txt>
		</>
	);
};

// --- Tirages et compteur --------------------------------------------------
/** Nombre de tirages affiché par le compteur : (avant, après, avancement) d'un saut entre deux valeurs. */
const compteurEn = (t: number): [number, number, number] => {
	const nbLents = D1.nb_lents;
	let n = nbLents;
	for (let j = 0; j < D1.durees_rapides.length; j++) if (t >= debutsRapides[j] + D1.durees_rapides[j]) n = nbLents + j + 1;
	let m: [number, number, number] = [n, n, 0];
	D1.lots.forEach((lot, i) => {
		const p = avance(t, T.lots + i * D.lot, D.lot);
		if (t >= T.lots + i * D.lot) m = [i === 0 ? D1.nb_rapides : D1.lots[i - 1], lot, p];
	});
	return m;
};

export const Compteur: React.FC<{t: number}> = ({t}) => {
	if (t < T.compteur) return null;
	const o = avance(t, T.compteur, D.compteur) * (1 - avance(t, T.zone, D.zone));
	// Pendant un saut, le compteur défile de l'ancienne valeur à la nouvelle, comme un compteur qui tourne.
	const [avant, apres, u] = compteurEn(t);
	const haut = 0.9 + HAUTEUR + 6 / UNITE; // bas du texte au-dessus de l'axe (6 px de plus : encre de Manim)
	return (
		<Txt x={xAxe(bornes[bornes.length - 1])} y={Y(AXE_CENTRE[1] + haut)} taille={17} ax={1} ay={1} couleur={C['text-primary']} opacite={o}>
			{`tirages · ${Math.round(lerp(avant, apres, u))}`}
		</Txt>
	);
};

/** Un trait de tirage (avant la fusion) : de la valeur sur l'axe jusqu'à 0.35 au-dessus. */
export const Traits: React.FC<{t: number}> = ({t}) => {
	const fin = T.fusion + D.fusion;
	if (t >= fin) return null;
	const eFusion = avance(t, T.fusion, D.fusion);
	const elements: React.ReactNode[] = [];
	const valeurs = t1000.slice(0, D1.nb_rapides);
	valeurs.forEach((v, k) => {
		// Avancement du trait : Create pour les tirages lents, apparition d'un coup à la fin de l'envol pour les rapides.
		const j = k - D1.nb_lents;
		const p =
			k < D1.nb_lents
				? avance(t - T.lents - k * DUREE_LENT, D.lentChoix + D.lentPause + D.lentEnvol, D.lentTrait)
				: t >= debutsRapides[j] + D1.durees_rapides[j]
					? 1
					: 0;
		if (p <= 0) return;
		if (eFusion <= 0) {
			elements.push(
				<line key={k} x1={xAxe(v)} x2={xAxe(v)} y1={yAxe} y2={yAxe - HAUT_TRAIT * UNITE * p} stroke={couleurCote(v)} strokeWidth={trait(4)} />,
			);
			return;
		}
		// Fusion : le trait devient la barre de sa tranche (même rectangle pour tous les traits de la tranche).
		const k2 = tranche(v);
		const w = trait(4);
		const xa = lerp(xAxe(v) - w / 2, xAxe(bornes[k2]) + (ECART_BARRES / 2) * UNITE, eFusion);
		const xb = lerp(xAxe(v) + w / 2, xAxe(bornes[k2 + 1]) - (ECART_BARRES / 2) * UNITE, eFusion);
		const h = lerp(HAUT_TRAIT, ETAT_40.h[k2], eFusion) * UNITE;
		elements.push(<rect key={k} x={xa} y={yAxe - h} width={xb - xa} height={h} fill={MEL(couleurCote(v), couleurCote(centre(k2)), eFusion)} />);
	});
	return <g>{elements}</g>;
};

// --- Barres ---------------------------------------------------------------
type Etat = {h: number[]; o: number[]};

/** Hauteurs (unités) et opacités des barres pour ces tirages, cette unité de hauteur et cette marge (points de %, ou null). */
const barres = (valeurs: number[], echelleH: number, marge: number | null): Etat => {
	const c = comptes(valeurs);
	return {
		h: c.map((n) => Math.max(n * echelleH, BARRE_VIDE)),
		o: c.map((_, k) => (marge !== null && Math.abs(centre(k) - 50) >= marge ? OPACITE_ESTOMPEE : 1)),
	};
};

const ETAT_40 = barres(t1000.slice(0, D1.nb_rapides), unite(t1000.slice(0, D1.nb_rapides)), null);
const ECHELLE_600 = unite(t1000);
const ECHELLE_4000 = HAUTEUR / Math.max(...comptes(t4000)); // pas de plafond à UNITE_MAX pour n = 4 000
const ETAT_MARGE_PETIT = barres(t1000, ECHELLE_600, M_PETIT);
const ETAT_MARGE_GRAND = barres(t4000, ECHELLE_4000, M_GRAND);
const IMAGES_CLES: {debut: number; duree: number; etat: Etat}[] = [
	...D1.lots.map((lot, i) => ({debut: T.lots + i * D.lot, duree: D.lot, etat: barres(t1000.slice(0, lot), unite(t1000.slice(0, lot)), null)})),
	{debut: T.bande, duree: D.bande, etat: ETAT_MARGE_PETIT},
	{debut: T.quatre, duree: D.quatre, etat: ETAT_MARGE_GRAND},
];

const melange = (a: Etat, b: Etat, u: number): Etat => ({h: a.h.map((x, k) => lerp(x, b.h[k], u)), o: a.o.map((x, k) => lerp(x, b.o[k], u))});

/** État des barres à l'instant t : transformations successives (Transform) d'un état à l'autre. */
const etatBarres = (t: number): Etat => {
	let etat = ETAT_40;
	for (const cle of IMAGES_CLES) {
		if (t < cle.debut) break;
		etat = melange(etat, cle.etat, avance(t, cle.debut, cle.duree));
	}
	return etat;
};

export const Barres: React.FC<{t: number}> = ({t}) => {
	if (t < T.fusion + D.fusion) return null;
	const etat = etatBarres(t);
	return (
		<g>
			{etat.h.map((h, k) => (
				<rect
					key={k}
					x={xAxe(bornes[k]) + (ECART_BARRES / 2) * UNITE}
					y={yAxe - h * UNITE}
					width={xAxe(bornes[k + 1]) - xAxe(bornes[k]) - ECART_BARRES * UNITE}
					height={h * UNITE}
					fill={couleurCote(centre(k))}
					opacity={etat.o[k]}
				/>
			))}
		</g>
	);
};

// --- Bande à 95 % ---------------------------------------------------------
const HAUT_BANDE = HAUTEUR + 0.25;
const DECAL_FLECHE = 0.12; // la flèche flotte au-dessus de la bande
const TAILLE_POINTE = 0.14;
const AJUSTE_BANDE = 3; // px : la flèche et son texte sont 3 px plus bas en Manim

const margeEn = (t: number) => lerp(M_PETIT, M_GRAND, avance(t, T.quatre, D.quatre));

/** Zone accentuée derrière l'histogramme (à dessiner avant les barres). */
export const ZoneSvg: React.FC<{t: number}> = ({t}) => {
	if (t < T.zone) return null;
	const m = margeEn(t);
	const a = xAxe(50 - m);
	const b = xAxe(50 + m);
	return <rect x={a} y={yAxe - HAUT_BANDE * UNITE} width={b - a} height={HAUT_BANDE * UNITE} fill={C.accent} opacity={D1.opacite_aire * avance(t, T.zone, D.zone)} />;
};

/** Flèche double au-dessus de la bande : Create dessine le trait puis chaque pointe. */
export const FlecheSvg: React.FC<{t: number}> = ({t}) => {
	if (t < T.bande) return null;
	const m = margeEn(t);
	const a = xAxe(50 - m);
	const b = xAxe(50 + m);
	const y = yAxe - (HAUT_BANDE + DECAL_FLECHE) * UNITE + AJUSTE_BANDE;
	const v = avance(t, T.bande, D.bande) * 3;
	const L = TAILLE_POINTE * UNITE;
	return (
		<g fill={C['text-primary']} stroke={C['text-primary']}>
			<line x1={a + L} x2={a + L + (b - a - 2 * L) * borne(v)} y1={y} y2={y} strokeWidth={trait(2)} />
			<polygon points={`${b},${y} ${b - L},${y - L / 2} ${b - L},${y + L / 2}`} opacity={borne(v - 1)} strokeWidth={0} />
			<polygon points={`${a},${y} ${a + L},${y - L / 2} ${a + L},${y + L / 2}`} opacity={borne(v - 2)} strokeWidth={0} />
		</g>
	);
};

/** Valeur de la marge et « 95 % des tirages » au-dessus de la flèche. */
export const BandeTexte: React.FC<{t: number}> = ({t}) => {
	if (t < T.bande) return null;
	const e = avance(t, T.bande, D.bande);
	const u = avance(t, T.quatre, D.quatre);
	const bas = AXE_CENTRE[1] + HAUT_BANDE + DECAL_FLECHE + TAILLE_POINTE / 2 + 0.1 - AJUSTE_BANDE / UNITE;
	const valeur = (m: number, o: number) => (
		<div style={{fontFamily: SERIF, fontSize: F(34), lineHeight: 1, color: C['text-primary'], whiteSpace: 'pre', opacity: o, position: 'absolute', left: '50%', transform: 'translateX(-50%)', bottom: 0}}>
			{`±${NBSP}${fr(m)}${NBSP}points`}
		</div>
	);
	return (
		<div style={{position: 'absolute', left: X(AXE_CENTRE[0]), top: Y(bas), transform: 'translate(-50%, -100%)', opacity: e}}>
			<div style={{fontFamily: SERIF, fontSize: F(28), lineHeight: 1, color: C['text-secondary'], whiteSpace: 'pre', textAlign: 'center', marginBottom: 0.08 * UNITE}}>
				{`95${NBSP}% des tirages`}
			</div>
			<div style={{position: 'relative', height: F(34)}}>
				{valeur(M_PETIT, 1 - u)}
				{valeur(M_GRAND, u)}
			</div>
		</div>
	);
};
