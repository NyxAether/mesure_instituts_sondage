// Primaire de la droite 2016 : photomontage des candidats, banderole, mème, puis graphe des sondages et résultat.
import React from 'react';
import {Img, OffthreadVideo, Sequence, staticFile, useCurrentFrame} from 'remotion';
import cous from '../../../../video/externe/corps/cous.json';
import donnees from '../../../donnees/s0.json';
import {C, G, MAIN, MONO, SERIF, couleurs, encre} from '../../lib/charte';
import {Papier, RECTANGLE, Terminal} from '../../lib/papier';
import {programmeSauts, etatSauts} from '../../lib/sautiller';
import {DUREE_MEME, T} from './temps';
import {F, FPS, NBSP, Txt, UNITE as U, X, Y, avance, degres, echelonne, largeurMono, lerp, pourCent, signe} from '../../lib/outils';

type Cle = 'fillon' | 'juppe' | 'sarkozy';
const {sondages, resultat} = donnees.primaire_2016;
const CANDIDATS: Cle[] = ['fillon', 'juppe', 'sarkozy'];
const NOMS: Record<Cle, string> = {fillon: 'Fillon', juppe: 'Juppé', sarkozy: 'Sarkozy'};

// --- Photomontage (unités Manim) ---
// Dans l'ordre où le narrateur les nomme, de gauche à droite : Juppé, Sarkozy, Fillon.
const PLACES: Record<Cle, {x: number; prenom: string; rang: number}> = {
	juppe: {x: -4.5, prenom: 'Alain', rang: 0},
	sarkozy: {x: 0, prenom: 'Nicolas', rang: 1},
	fillon: {x: 4.6, prenom: 'François', rang: 2},
};
const HAUTEUR_TETE = 1.4; // sur les corps
const HAUTEUR_GRAPHE = 0.8; // sur le graphe, où Juppé et Sarkozy ne sont qu'à 0,8 unité l'un de l'autre
const COU_BAS = 4.6; // du col au bas de la photo : le bas des corps sort du cadre, même quand ils sautent
// Sous le cadre (−4), avec la marge d'un saut (0,3) et de la bascule qui lève un coin. Manim prenait −4,3 :
// au sommet de certains sauts, le bas de Fillon se décollait de quelques pixels.
const BAS_CORPS = -4.5;
const ECHELLE_CORPS: Partial<Record<Cle, number>> = {sarkozy: 0.85}; // bras levé et cadrage serré : il paraîtrait plus massif
const INCLINAISONS: Record<Cle, number> = {fillon: -0.12, juppe: 0.08, sarkozy: -0.03};
const DECALAGE_TAS: Record<Cle, number> = {fillon: -0.45, juppe: 0.05, sarkozy: 0.5};

const col = (cle: Cle) => BAS_CORPS + COU_BAS * (ECHELLE_CORPS[cle] ?? 1);
const SAUTS = Object.fromEntries(
	CANDIDATS.map((cle) => [cle, programmeSauts(100 + PLACES[cle].rang, T.sauts, T.graphe, 0.04)]),
) as Record<Cle, ReturnType<typeof programmeSauts>>;

// --- Graphe (unités Manim) ---
const GAUCHE = -5.2, DROITE = 2.6, BAS = -2.6, HAUT = 2.3, X_RES = 4.6, Y_MAX = 50;
const gx = (i: number) => GAUCHE + (i / (sondages.length - 1)) * (DROITE - GAUCHE);
const gy = (v: number) => BAS + (v / Y_MAX) * (HAUT - BAS);
const GRADUATIONS = [0, 10, 20, 30, 40, 50];
const DERNIER = sondages.length - 1;
const sommets = (cle: Cle) => sondages.map((s, i) => [gx(i), gy(s[cle])] as const);
const DESSUS = HAUTEUR_GRAPHE / 2 + 0.1; // les têtes suivent la pointe un peu au-dessus

type Point = readonly [number, number];
/** Pointe d'une courbe tracée à l'avancement a, segment après segment, en temps égal. */
const pointe = (points: readonly Point[], a: number): Point => {
	const u = a * (points.length - 1);
	const i = Math.min(Math.floor(u), points.length - 2);
	return [lerp(points[i][0], points[i + 1][0], u - i), lerp(points[i][1], points[i + 1][1], u - i)];
};

/** Position (unités), échelle et angle (radians) d'une tête, à tout instant de la primaire. */
const etatTete = (cle: Cle, t: number) => {
	const inclinaison = INCLINAISONS[cle];
	// Sur le corps : le groupe corps + tête sautille, en tournant autour du bas de la photo.
	const surCorps = (t: number) => {
		const {dx, dy, angle} = etatSauts(SAUTS[cle], t);
		const [px, py] = [PLACES[cle].x, BAS_CORPS];
		const [rx, ry] = [0, col(cle) + HAUTEUR_TETE / 2 - 0.2 - py];
		return {
			x: px + dx + rx * Math.cos(angle) - ry * Math.sin(angle),
			y: py + dy + rx * Math.sin(angle) + ry * Math.cos(angle),
			echelle: 1,
			angle: inclinaison + angle,
		};
	};
	// Trajet qui tangue comme un papier qu'on promène.
	const trajet = (a: number, depart: Point, arrivee: Point, arc: number, dandinement: number, oscillations: number, echelle: [number, number]) => ({
		x: lerp(depart[0], arrivee[0], a),
		y: lerp(depart[1], arrivee[1], a) + arc * Math.sin(Math.PI * a),
		echelle: lerp(echelle[0], echelle[1], a),
		angle: inclinaison + dandinement * Math.sin(a * oscillations * 2 * Math.PI),
	});
	const petite = HAUTEUR_GRAPHE / HAUTEUR_TETE;
	const p = sommets(cle);
	const surCorpsFin = surCorps(T.graphe);
	const surAxe: Point = [GAUCHE - 1.3, p[0][1]];
	const tas: Point = [p[DERNIER][0] + DECALAGE_TAS[cle], p[DERNIER][1] + DESSUS];

	if (t < T.tetes + 0.8) {
		const a = avance(t, T.tetes, 0.8);
		return {...surCorps(t), echelle: lerp(1.4, 1, a), opacite: a};
	}
	if (t < T.graphe) return {...surCorps(t), opacite: 1};
	if (t < T.courbes) {
		const a = avance(t, T.graphe, 1.8);
		return {...trajet(a, [surCorpsFin.x, surCorpsFin.y], surAxe, 0.8, 0.07, 3, [1, petite]), opacite: 1};
	}
	if (t < T.saut) {
		// Chaque tête rejoint la pointe de sa courbe et la suit ; elles finissent en tas sur le dernier sondage.
		const a = avance(t, T.courbes, 3.5);
		const [qx, qy] = pointe(p, a);
		const debut = (1 - a) ** 3, fin = a ** 3;
		return {
			x: qx + (surAxe[0] - p[0][0]) * debut + (tas[0] - p[DERNIER][0]) * fin,
			y: qy + DESSUS + (surAxe[1] - p[0][1] - DESSUS) * debut + (tas[1] - p[DERNIER][1] - DESSUS) * fin,
			echelle: petite,
			angle: inclinaison + 0.07 * Math.sin(a * 3 * 2 * Math.PI),
			opacite: 1,
		};
	}
	// Envol vers le résultat, juste à droite de son point ; tout s'efface à la fin de la primaire.
	const a = avance(t, T.saut, 1.6);
	return {
		...trajet(a, tas, [X_RES + 0.62, gy(resultat[cle])], 0.6, 0.1, 2, [petite, petite]),
		opacite: 1 - avance(t, T.sortiePrimaire, 0.8),
	};
};

const Tete: React.FC<{nom: string; x: number; y: number; hauteur: number; angle: number; opacite: number}> = ({nom, x, y, hauteur, angle, opacite}) => (
	<Img
		src={staticFile(`portraits/${nom}.png`)}
		style={{
			position: 'absolute',
			left: X(x),
			top: Y(y),
			height: hauteur * U,
			transform: `translate(-50%, -50%) rotate(${degres(angle)}deg)`,
			opacity: opacite,
		}}
	/>
);
export {Tete};

const Corps: React.FC<{cle: Cle; t: number; opacite: number}> = ({cle, t, opacite}) => {
	const [fx, fy] = cous[cle];
	const hauteur = (COU_BAS * (ECHELLE_CORPS[cle] ?? 1)) / (1 - fy);
	const entree = avance(t, T.corps, 0.9);
	const {dx, dy, angle} = etatSauts(SAUTS[cle], t);
	const pivot = [X(PLACES[cle].x), Y(BAS_CORPS)];
	return (
		<div
			style={{
				position: 'absolute',
				inset: 0,
				transformOrigin: `${pivot[0]}px ${pivot[1]}px`,
				transform: `translate(${dx * U}px, ${(-dy + 1.5 * (1 - entree)) * U}px) rotate(${degres(angle)}deg)`,
				opacity: opacite * entree,
			}}
		>
			{/* Le col de la photo vient au point (x, col) */}
			<Img
				src={staticFile(`corps/${cle}.png`)}
				style={{
					position: 'absolute',
					left: X(PLACES[cle].x),
					top: Y(col(cle)),
					height: hauteur * U,
					transform: `translate(${-fx * 100}%, ${-fy * 100}%)`,
				}}
			/>
		</div>
	);
};

const Etiquette: React.FC<{cle: Cle; opacite: number}> = ({cle, opacite}) => {
	const {x, prenom, rang} = PLACES[cle];
	return (
		<div
			style={{
				position: 'absolute',
				left: X(x),
				top: Y(-3.45),
				transform: `translate(-50%, -50%) rotate(${degres(0.05 * (-1) ** rang)}deg)`,
				padding: `${0.12 * U}px ${0.2 * U}px ${0.1 * U}px`,
				opacity: opacite,
			}}
		>
			<Papier points={RECTANGLE} graine={20 + rang} bruit={[1.2, 3]} />
			<span style={{position: 'relative', fontFamily: MAIN, fontSize: F(30), lineHeight: 1, color: encre[cle]}}>{prenom}</span>
		</div>
	);
};

/** Banderole : ruban de papier découpé à pans fourchus, qui descend du haut ; le titre s'écrit lettre à lettre. */
const Banderole: React.FC<{t: number; opacite: number}> = ({t, opacite}) => {
	const a = avance(t, T.banderole, 0.7);
	const morceaux: [string, boolean][] = [['Primaire de la ', false], ['droite', true], [' et du centre', false]];
	const n = morceaux.reduce((s, [m]) => s + m.length, 0);
	let k = 0;
	const pan = (cote: 'left' | 'right', graine: number) => (
		<div style={{position: 'absolute', [cote]: -0.9 * U, width: 1.2 * U, top: 0.25 * U, bottom: -0.18 * U}}>
			<Papier
				vue="0 0 120 100"
				bruit={[2, 2]}
				graine={graine}
				points={
					cote === 'left'
						? [[120, 0], [0, 0], [35, 47], [0, 100], [120, 100]]
						: [[0, 0], [120, 0], [85, 47], [120, 100], [0, 100]]
				}
			/>
		</div>
	);
	return (
		<div
			style={{
				position: 'absolute',
				left: X(0),
				top: Y(3.05),
				transform: `translate(-50%, -50%) translateY(${-0.8 * U * (1 - a)}px)`,
				padding: `${0.22 * U}px ${0.4 * U}px`,
				opacity: opacite * a,
			}}
		>
			{pan('left', 31)}
			{pan('right', 32)}
			<Papier points={[[0, 100], [0, 0], [100, -3], [100, 102]]} graine={33} bruit={[0.25, 2]} />
			<div style={{position: 'relative', fontFamily: SERIF, fontSize: F(40), lineHeight: 1, whiteSpace: 'pre', color: C['text-primary']}}>
				{morceaux.map(([mot, italique]) => (
					<span key={mot} style={italique ? {fontStyle: 'italic', color: C.accent} : undefined}>
						{[...mot].map((lettre, i) => (
							<span key={i} style={{opacity: echelonne(t, T.ecriture, 1.5, n, k++, 0.1)}}>{lettre}</span>
						))}
					</span>
				))}
			</div>
		</div>
	);
};

/** Le mème, en coupe franche, avec son son, dans une fenêtre terminal de la charte. */
const Meme: React.FC<{t: number}> = ({t}) => {
	const largeur = 9.6 * U; // l'extrait est en 16/9
	const marge = 0.16 * U;
	return (
		<div style={{position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: avance(t, T.meme, 0.3)}}>
			<Terminal dossier="~/sondages" commande="play quelle_indignite.mp4" statut="france 2 · 17 nov. 2016" taille={F(15)} marge={marge}>
				<div style={{width: largeur, height: (largeur * 9) / 16, background: C['bg-secondary']}}>
					<Sequence from={Math.round(T.extrait * FPS)}durationInFrames={Math.round(DUREE_MEME * FPS)} layout="none">
						<OffthreadVideo src={staticFile('meme/quelle_indignite.mp4')} style={{width: largeur, height: (largeur * 9) / 16, display: 'block'}} />
					</Sequence>
				</div>
			</Terminal>
		</div>
	);
};

const Graphe: React.FC<{t: number}> = ({t}) => {
	const entree = avance(t, T.graphe, 1.8);
	const labX = (k: number) => echelonne(t, T.graphe, 1.8, sondages.length, k, 0.1);
	const courbes = avance(t, T.courbes, 3.5);
	const saut = avance(t, T.saut, 1.6);
	const points = avance(t, T.points, 0.8);
	const premier = sondages[0], dernier = sondages[DERNIER];
	const trait = {strokeLinecap: 'round', strokeLinejoin: 'round', fill: 'none'} as const;

	// Libellés sous l'axe : deux lignes mono, puis la source et la légende des écarts.
	const hautLabX = Y(BAS) + 0.12 * U;
	const basLabX = hautLabX + 2 * F(11) + 0.04 * U;
	const gaucheLabX = X(gx(0)) - Math.max(largeurMono('harris', 11), largeurMono('09 nov.', 11)) / 2;
	const hautSource = basLabX + 0.15 * U;
	const droiteRes = X(X_RES) + largeurMono('résultat 20 nov.', 12) / 2;
	const yNomFillon = Y(gy(premier.fillon)) + 0.15 * U;
	const nomEcart = `+/− : écart au dernier sondage (${dernier.institut.toLowerCase()}, ${dernier.terrain})`;

	return (
		<>
			<Txt x={0.55 * U} y={0.55 * U} taille={16} couleur={C['text-muted']} opacite={entree}>
				primaire de la droite et du centre · 1er tour · novembre 2016
			</Txt>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				<g opacity={entree}>
					{GRADUATIONS.map((v) => (
						<line key={v} x1={X(GAUCHE - 0.3)} x2={X(X_RES + 0.4)} y1={Y(gy(v))} y2={Y(gy(v))} stroke={G.grid} strokeWidth={1.5} />
					))}
				</g>
				{CANDIDATS.map((cle) => {
					const p = sommets(cle);
					const u = courbes * DERNIER;
					const trace = [...p.slice(0, Math.floor(u) + 1), pointe(p, courbes)];
					return (
						<g key={cle} stroke={couleurs[cle]} fill={couleurs[cle]}>
							<polyline points={trace.map(([x, y]) => `${X(x)},${Y(y)}`).join(' ')} strokeWidth={3} {...trait} stroke={couleurs[cle]} />
							{p.map(([x, y], k) => (
								<circle
									key={k}
									cx={X(x)}
									cy={Y(y)}
									r={0.06 * U}
									stroke="none"
									opacity={k === 0 ? avance(t, T.noms, 0.6) : Math.min(1, Math.max(0, (u - k + 0.15) * 4))}
								/>
							))}
							{/* Saut en pointillé vers le résultat, tracé en même temps que l'envol des têtes */}
							{saut > 0 && (
								<line
									x1={X(p[DERNIER][0])}
									y1={Y(p[DERNIER][1])}
									x2={X(lerp(p[DERNIER][0], X_RES, saut))}
									y2={Y(lerp(p[DERNIER][1], gy(resultat[cle]), saut))}
									stroke={couleurs[cle]}
									strokeWidth={2.5}
									strokeDasharray="11 9"
								/>
							)}
							<circle
								cx={X(X_RES)}
								cy={Y(gy(resultat[cle]))}
								r={0.11 * U * lerp(1.5, 1, points)}
								stroke="none"
								opacity={points}
							/>
						</g>
					);
				})}
			</svg>

			{GRADUATIONS.map((v) => (
				<Txt key={v} x={X(GAUCHE - 0.3 - 0.12)} y={Y(gy(v))} ax={1} ay={0.5} taille={13} couleur={C['text-secondary']} opacite={entree}>
					{`${v}${NBSP}%`}
				</Txt>
			))}
			{sondages.map((s, i) => (
				<div
					key={i}
					style={{
						position: 'absolute',
						left: X(gx(i)),
						top: hautLabX,
						transform: 'translateX(-50%)',
						fontFamily: MONO,
						fontSize: F(11),
						lineHeight: 1,
						textAlign: 'center',
						whiteSpace: 'pre',
						opacity: labX(i),
					}}
				>
					<div style={{color: C['text-secondary'], marginBottom: 0.04 * U}}>{s.institut.split(' ')[0].toLowerCase()}</div>
					<div style={{color: C['text-muted']}}>{`${s.fin.slice(-2)} nov.`}</div>
				</div>
			))}
			<Txt x={gaucheLabX} y={hautSource} taille={12} couleur={C['text-muted']} opacite={entree}>
				sondages publiés après le deuxième débat · source : wikipédia
			</Txt>

			{/* Noms des courbes et annotation du premier sondage */}
			{CANDIDATS.map((cle) => {
				const [x, y] = sommets(cle)[0];
				return (
					<Txt key={cle} x={X(x)} y={cle === 'fillon' ? yNomFillon : Y(y) - 0.15 * U} ax={0.5} ay={cle === 'fillon' ? 0 : 1} taille={14} couleur={couleurs[cle]} opacite={avance(t, T.noms, 0.6)}>
						{NOMS[cle].toLowerCase()}
					</Txt>
				);
			})}
			<Txt x={X(GAUCHE)} y={yNomFillon + F(14) + 0.2 * U} taille={26} police={SERIF} couleur={couleurs.fillon} opacite={avance(t, T.fillon17, 0.6)}>
				{`Fillon troisième, ${pourCent(premier.fillon)}`}
			</Txt>

			{/* Résultat du premier tour */}
			<Txt x={X(X_RES)} y={hautLabX} ax={0.5} taille={12} couleur={C['text-primary']} opacite={avance(t, T.saut, 1.6)}>
				résultat 20 nov.
			</Txt>
			{CANDIDATS.map((cle) => {
				const y = Y(gy(resultat[cle]));
				const x = X(X_RES + 0.11 + 1.15);
				return (
					<React.Fragment key={cle}>
						<Txt x={x} y={y} ay={0.5} taille={15} couleur={couleurs[cle]} opacite={points}>
							{pourCent(resultat[cle])}
						</Txt>
						<Txt x={x} y={y + F(15) / 2 + 0.06 * U} taille={13} couleur={couleurs[cle]} opacite={echelonne(t, T.ecarts, 1, 3, CANDIDATS.indexOf(cle), 0.2)}>
							{`${signe(resultat[cle] - dernier[cle])} pts`}
						</Txt>
					</React.Fragment>
				);
			})}
			<Txt
				x={X(X_RES + 0.11 + 0.3)}
				y={Y(gy(resultat.fillon) + 0.11 + 0.35)}
				ax={1}
				ay={1}
				taille={30}
				police={SERIF}
				couleur={couleurs.fillon}
				opacite={avance(t, T.soir, 0.6)}
				style={{translate: `0 ${0.1 * U * (1 - avance(t, T.soir, 0.6))}px`}}
			>
				{`le soir du vote${NBSP}: ${pourCent(resultat.fillon)}`}
			</Txt>
			<Txt x={droiteRes} y={hautSource + F(12) + 0.08 * U} ax={1} taille={12} couleur={C['text-muted']} opacite={avance(t, T.ecarts, 1)}>
				{nomEcart}
			</Txt>
		</>
	);
};

export const Primaire: React.FC = () => {
	const t = useCurrentFrame() / FPS;
	const pendantMeme = t >= T.meme && t < T.retour;
	const depart = 1 - avance(t, T.graphe, 1.8); // corps, étiquettes et banderole s'effacent quand les têtes partent
	const sortie = 1 - avance(t, T.sortiePrimaire, 0.8);
	if (t >= T.accident) return null;
	return (
		<>
			{!pendantMeme && depart > 0 && (
				<>
					{CANDIDATS.map((cle) => (
						<Corps key={cle} cle={cle} t={t} opacite={depart} />
					))}
					{CANDIDATS.map((cle) => (
						<Etiquette key={cle} cle={cle} opacite={depart * echelonne(t, T.tetes, 0.8, 3, PLACES[cle].rang, 0.3)} />
					))}
					{t >= T.banderole && <Banderole t={t} opacite={depart} />}
				</>
			)}
			{t >= T.graphe && (
				<div style={{position: 'absolute', inset: 0, opacity: sortie}}>
					<Graphe t={t} />
				</div>
			)}
			{/* Les têtes, toujours devant les courbes */}
			{!pendantMeme &&
				t >= T.tetes &&
				CANDIDATS.map((cle) => {
					const e = etatTete(cle, t);
					return <Tete key={cle} nom={cle} x={e.x} y={e.y} hauteur={HAUTEUR_TETE * e.echelle} angle={e.angle} opacite={e.opacite} />;
				})}
			{pendantMeme && <Meme t={t} />}
		</>
	);
};
