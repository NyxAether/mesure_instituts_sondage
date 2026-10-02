// Séquence 2 : l'entonnoir. 45 % des écarts hors de leur marge, contre 5 % attendus en théorie.
// Port de video/scenes/s2_entonnoir.py ; chiffres et tirages exportés par video-js/export/exporter_s2.py.
import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import donnees from '../../../donnees/s2.json';
import {C, G, SERIF} from '../../lib/charte';
import {SousEntete, Titre} from '../../lib/composants';
import {F, FPS, NBSP, Txt, UNITE as U, Y, avance, echelonne, fr, largeurMono, lerp} from '../../lib/outils';
import {AxesLog, Compteur, Entonnoir as ZoneEntonnoir, LegendeEntonnoir, Nuage, TRAIT, Trace, creerRepere, type PointNuage} from './EntonnoirGraphe';
import {T} from './temps';

const {points, marge: marges, exemple, z95} = donnees;
const N = points.length;
const POINTS_DE_POURCENT = 100; // un écart s'exprime en points de pourcentage, une part en %
const P_MOITIE = 0.5; // l'entonnoir tracé est celui d'un résultat à 50 %, la marge la plus large

// --- Calculs sur les données (fonctions pures des données exportées) ---
const horsMarge = (ecarts: number[]) => ecarts.map((e, i) => Math.abs(e) > points[i].marge);
const ecartsReels = points.map((p) => p.reel);
const ecartsSimules = points.map((p) => p.simule);
const horsReel = horsMarge(ecartsReels);
const horsSimule = horsMarge(ecartsSimules);
const part = (hors: boolean[]) => (POINTS_DE_POURCENT * hors.filter(Boolean).length) / hors.length;
// Simplification visuelle : la couleur suit l'entonnoir tracé (marge à p = 50 %), alors que les pourcentages
// affichés comparent chaque écart à sa propre marge (celle de l'export).
const marge50 = points.map((p) => POINTS_DE_POURCENT * z95 * Math.sqrt((P_MOITIE * (1 - P_MOITIE)) / p.n));
const colore = (ecarts: number[]) => ecarts.map((e, i) => (Math.abs(e) > marge50[i] ? 1 : 0));
const coloreReel = colore(ecartsReels);
const coloreSimule = colore(ecartsSimules);
const PART_SIMULEE = part(horsSimule);
const PART_REELLE = part(horsReel);

// Gardes de cohérence (les assert de la scène Manim).
if (Math.abs(PART_REELLE / POINTS_DE_POURCENT - donnees.part_hors_marge) > 0.5 / N) throw new Error("part hors marge différente de l'export");
const ECART_EXEMPLE = POINTS_DE_POURCENT * (exemple.poll - exemple.vote);
if (!points.some((p) => p.n === exemple.n && Math.abs(p.reel - ECART_EXEMPLE) < 1e-4)) throw new Error("point d'exemple absent des données");

const repere = creerRepere();
const MARGES_PTS = marges.map((m) => ({taille: m.n, m: POINTS_DE_POURCENT * m.m}));

const signeDecimal = (v: number) => fr(v).replace('-', '−');

/** Un point de l'échantillon, expliqué : point, barre de sa marge, note. */
const Exemple: React.FC<{t: number}> = ({t}) => {
	const marge = POINTS_DE_POURCENT * z95 * Math.sqrt((exemple.vote * (1 - exemple.vote)) / exemple.n);
	const px = repere.x(exemple.n), py = repere.y(ECART_EXEMPLE);
	const dot = avance(t, T.pointExemple, 0.6);
	const note = avance(t, T.noteExemple, 0.9);
	const barre = avance(t, T.barreExemple, 0.8);
	const sortie = 1 - avance(t, T.retraitExemple, 0.6);
	const rayon = 0.08 * U;
	const haut = repere.y(marge), bas = repere.y(-marge);
	const gauche = px + rayon + 0.5 * U; // bord gauche de la note
	const centreNote = py + 0.4 * U;
	const decalage = -0.1 * U * (1 - note);
	const libelle = {taille: 15};
	return (
		<div style={{position: 'absolute', inset: 0, opacity: sortie}}>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				{note > 0 && (
					<line
						x1={px + rayon}
						y1={py}
						x2={lerp(px + rayon, gauche, note)}
						y2={lerp(py, centreNote, note)}
						stroke={C['text-muted']}
						strokeWidth={TRAIT(1)}
						strokeDasharray={`${0.025 * U} ${0.025 * U}`}
					/>
				)}
				<Trace d={`M${px},${bas} L${px},${lerp(bas, haut, barre)}`} trace={barre > 0 ? 1 : 0} stroke={C['text-secondary']} largeur={TRAIT(3)} />
				<g opacity={barre} stroke={C['text-secondary']} strokeWidth={TRAIT(3)}>
					{[haut, bas].map((y) => (
						<line key={y} x1={px - rayon} x2={px + rayon} y1={y} y2={y} />
					))}
				</g>
				<circle cx={px} cy={py} r={rayon * lerp(0.5, 1, dot)} fill={G.series[0]} opacity={dot} />
			</svg>
			<div
				style={{
					position: 'absolute',
					left: gauche + decalage,
					top: centreNote,
					transform: 'translateY(-50%)',
					display: 'flex',
					flexDirection: 'column',
					gap: 0.1 * U,
					opacity: note,
				}}
			>
				<Txt x={0} y={0} {...libelle} couleur={C['text-primary']} style={{position: 'static'}}>
					{`${exemple.pays.toLowerCase()} · ${exemple.annee} · ${fr(exemple.n, 0)} sondés`}
				</Txt>
				<Txt x={0} y={0} {...libelle} couleur={C['text-secondary']} style={{position: 'static'}}>
					{`sondage ${fr(POINTS_DE_POURCENT * exemple.poll)}${NBSP}% · résultat ${fr(POINTS_DE_POURCENT * exemple.vote)}${NBSP}%`}
				</Txt>
				<Txt x={0} y={0} {...libelle} couleur={C['text-secondary']} style={{position: 'static'}}>
					{`écart ${signeDecimal(ECART_EXEMPLE)} pts · marge ±${NBSP}${fr(marge)} pts`}
				</Txt>
			</div>
		</div>
	);
};

/** Dernier plan : le chiffre, seul. */
const Constat: React.FC<{t: number}> = ({t}) => {
	const chiffre = avance(t, T.constat, 1);
	const phrase = avance(t, T.phrase, 0.6);
	const rappel = avance(t, T.rappel, 0.6);
	return (
		<div
			style={{
				position: 'absolute',
				left: 0,
				right: 0,
				top: Y(-0.4),
				transform: 'translateY(-50%)',
				display: 'flex',
				flexDirection: 'column',
				alignItems: 'center',
				gap: 0.35 * U,
				fontFamily: 'inherit',
			}}
		>
			<div style={{fontFamily: SERIF, fontSize: F(110), lineHeight: 1, marginBottom: -0.28 * F(110), marginTop: 0.1 * F(110), whiteSpace: 'pre', opacity: chiffre, transform: `translateY(${0.15 * U * (1 - chiffre)}px)`}}>
				<Titre mot={`${fr(PART_REELLE, 0)}${NBSP}%`} />
			</div>
			<div style={{fontFamily: SERIF, fontSize: F(40), lineHeight: 1, color: C['text-primary'], whiteSpace: 'pre', opacity: phrase}}>
				des écarts sortent de leur marge d’erreur
			</div>
			<div style={{fontFamily: SERIF, fontSize: F(32), lineHeight: 1, color: C['text-secondary'], whiteSpace: 'pre', opacity: rappel}}>
				{`au lieu des 5${NBSP}% attendus en théorie · presque un sur deux`}
			</div>
		</div>
	);
};

export const Entonnoir: React.FC = () => {
	const t = useCurrentFrame() / FPS;
	const graphe = 1 - avance(t, T.sortieGraphe, 0.8);
	const apparitionAxes = avance(t, T.axes, 1.2);
	const zone = avance(t, T.entonnoir, 1.2);
	const realite = avance(t, T.realite, 3);
	const cadreReel = avance(t, T.cadreReel, 0.6);
	const compteur = avance(t, T.compteur, 0.6);

	const nuage: PointNuage[] = points.map((p, k) => ({
		taille: p.n,
		ecart: lerp(repere.borne(ecartsSimules[k]), repere.borne(ecartsReels[k]), realite),
		hors: lerp(coloreSimule[k], coloreReel[k], realite),
		apparition: echelonne(t, T.tirages, 2.5, N, k, 0.0015),
	}));

	const droiteTitreY = 0.6 * U + largeurMono('écart sondage − résultat (points)', 14);
	const texteCadre = 'en théorie : des tirages aléatoires de même taille';
	const texteReel = 'la réalité : les sondages publiés';
	// « attendu » : sous le compteur, aligné à droite du repère.
	const hautAttendu = repere.haut + 0.1 * U + 1.15 * U;

	const ligneSource = `jennings & wlezien · ${fr(donnees.nb_lignes, 0)} intentions de vote · ${fr(donnees.nb_sondages, 0)} sondages · ${donnees.nb_pays} pays · dernière semaine · depuis ${donnees.annee_min}`;

	return (
		<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
			<div style={{opacity: graphe}}>
				<SousEntete opacite={avance(t, T.source, 0.8)}>
					{ligneSource}
				</SousEntete>
				{t >= T.axes && (
					<AxesLog
						repere={repere}
						apparition={apparitionAxes}
						trace={apparitionAxes}
						titreX="taille de l’échantillon (échelle log)"
						titreY="écart sondage − résultat (points)"
					>
						{t >= T.entonnoir && <ZoneEntonnoir repere={repere} marges={MARGES_PTS} apparition={zone} trace={zone} />}
					</AxesLog>
				)}
				{t >= T.entonnoir && <LegendeEntonnoir repere={repere} ecart={-9} texte={`marge d’erreur à 95${NBSP}%`} opacite={zone} />}
				{t >= T.cadreTheorie && (
					<>
						<Txt
							x={repere.gauche - 0.15 * U - largeurMono('+10', 14) + droiteTitreY}
							y={repere.haut - 0.15 * U - F(14) / 2}
							ay={0.5}
							taille={15}
							couleur={C['text-primary']}
							opacite={avance(t, T.cadreTheorie, 0.6) * (1 - cadreReel)}
						>
							{texteCadre}
						</Txt>
						<Txt
							x={repere.gauche - 0.15 * U - largeurMono('+10', 14) + droiteTitreY}
							y={repere.haut - 0.15 * U - F(14) / 2}
							ay={0.5}
							taille={15}
							couleur={C['text-primary']}
							opacite={cadreReel}
						>
							{texteReel}
						</Txt>
					</>
				)}
				{t >= T.tirages && <Nuage repere={repere} points={nuage} />}
				{t >= T.compteur && (
					<>
						<Compteur repere={repere} libelle="hors de leur marge d’erreur" pourcent={lerp(PART_SIMULEE, PART_REELLE, realite)} opacite={compteur} />
						<Txt x={repere.droite} y={hautAttendu} ax={1} taille={15} couleur={C['text-muted']} opacite={cadreReel}>
							{`attendu : 5${NBSP}%`}
						</Txt>
					</>
				)}
			</div>
			{t < T.retraitExemple + 0.6 && t >= T.pointExemple && <Exemple t={t} />}
			{t >= T.constat && <Constat t={t} />}
		</AbsoluteFill>
	);
};

export const dureeEntonnoir = Math.round(T.fin * FPS);
