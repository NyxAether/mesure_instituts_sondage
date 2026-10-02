// Séquence 6 : « Une prédiction plus qu'une photographie, un présage plus qu'une prédiction ».
// Port de video/scenes/s6_presage.py ; les chiffres viennent de donnees/s6.json (export/exporter_s6.py).
import React from 'react';
import {AbsoluteFill} from 'remotion';
import donnees from '../../../donnees/s6.json';
import {C, G, SERIF} from '../../lib/charte';
import {BAS_ENTETE, Titre} from '../../lib/composants';
import {avance, echelonne, F, fr, FPS, NBSP, Txt, UNITE, X, Y} from '../../lib/outils';
import {CRITIQUE, DUREE_TOTALE, FONDU_SORTIE, FORMULE, HASARD, MOTS, PANELS, PRECAUTIONS} from './temps';
import {Apparition, Citation, Colonne, Libelle, Serif, useTemps} from './elements';

const {base, france, par_taille: tailles} = donnees;

// Gardes de cohérence (les assert de la scène Manim).
if (tailles.length === 0) throw new Error('aucune tranche de taille dans s6.json');
if (france.sondages > base.sondages) throw new Error('plus de sondages français que de sondages au total');
const E_MAX = Math.max(...tailles.map((t) => t.obs));
if (!(E_MAX > 0)) throw new Error('erreur observée maximale nulle');

// Bas de l'en-tête (en px) : tout ce qui se place « sous l'en-tête » part de là.
const GAUCHE = 0.55 * UNITE;

/** Bloc de texte qui commence sous l'en-tête, aligné à gauche. */
const SousTete: React.FC<{ecart: number; gap: number; opacite?: number; children: React.ReactNode}> = ({ecart, gap, opacite = 1, children}) => (
	<div style={{position: 'absolute', left: GAUCHE, top: BAS_ENTETE + ecart * UNITE, opacity: opacite}}>
		<Colonne gap={gap}>{children}</Colonne>
	</div>
);

/** Opacité d'un plan qui s'efface : fondu de sortie de 0,8 s à partir de `sortie`. */
const sortie = (t: number, debut: number) => 1 - avance(t, debut, FONDU_SORTIE);

const Precautions: React.FC<{t: number}> = ({t}) => {
	const P = PRECAUTIONS;
	const partFrance = france.sondages / base.sondages;
	const largeur = 9.0 * UNITE;
	const limites = [
		`une base qui s’arrête en ${base.fin}`,
		'peu de sondages français',
		'des sondages d’un même jour parfois fusionnés en une moyenne',
	];
	return (
		<SousTete ecart={0.7} gap={0.55} opacite={sortie(t, P.sortie)}>
			<Colonne gap={0.12}>
				<Apparition t={t} debut={P.base} duree={0.8}>
					<Libelle taille={15}>
						{`${fr(base.sondages, 0)} sondages · ${base.pays} pays · élections de ${base.annee_min} à ${base.fin}`}
					</Libelle>
				</Apparition>
				<div style={{position: 'relative', width: largeur, height: 0.35 * UNITE}}>
					<Apparition t={t} debut={P.base} duree={0.8}>
						<div style={{width: largeur, height: 0.35 * UNITE, background: G['mark-muted'], opacity: 0.6}} />
					</Apparition>
					<Apparition t={t} debut={P.france} duree={0.8} absolu>
						<div style={{width: largeur * partFrance, height: 0.35 * UNITE, background: G.series[0]}} />
					</Apparition>
				</div>
				<Apparition t={t} debut={P.france} duree={0.8}>
					<Libelle taille={15} couleur={G.series[0]}>
						{`france : ${fr(france.sondages, 0)} sondages, ${fr(100 * partFrance, 0)}${NBSP}%`}
					</Libelle>
				</Apparition>
			</Colonne>
			<Colonne gap={0.22}>
				{limites.map((texte, k) => (
					<Apparition key={texte} t={t} debut={P.limites[k]} duree={0.5} decalage={[0, 0.1 * UNITE]}>
						<Serif taille={26}>{texte}</Serif>
					</Apparition>
				))}
			</Colonne>
			<Apparition t={t} debut={P.conclusion} duree={0.6}>
				<Serif taille={26} couleur={C['text-secondary']}>
					{`des tendances solides sur ${base.pays} pays, indicatives pour un pays pris seul`}
				</Serif>
			</Apparition>
		</SousTete>
	);
};

const Hasard: React.FC<{t: number}> = ({t}) => {
	const H = HASARD;
	const hautMax = 2.7;
	const baseY = -2.9;
	const x0 = -5.0;
	const pas = 1.45;
	const lb = 0.42;
	const ecartBarres = 0.04;
	const centre = (i: number) => x0 + i * pas + (lb + ecartBarres) / 2; // centre de la paire de barres i
	const gauche = X(x0 - 0.5);
	const droite = X(x0 + (tailles.length - 1) * pas + lb + 0.5);
	const milieuX = X((centre(0) + centre(tailles.length - 1)) / 2);
	const yBas = Y(baseY);
	return (
		<AbsoluteFill style={{opacity: sortie(t, H.sortie)}}>
			{/* légende et titre du graphique, sous l'en-tête */}
			<div style={{position: 'absolute', left: GAUCHE, top: BAS_ENTETE + 0.3 * UNITE}}>
				<Colonne gap={0.2}>
					<Apparition t={t} debut={H.axes} duree={0.8}>
						<Libelle taille={15} couleur={C['text-primary']}>erreur typique d’un sondage, selon sa taille</Libelle>
					</Apparition>
					<Colonne gap={0.12}>
						<Apparition t={t} debut={H.axes} duree={0.8}>
							<LigneLegende couleur={C.accent} opacite={0.6}>hasard du tirage : ce que mesure la marge</LigneLegende>
						</Apparition>
						<Apparition t={t} debut={H.observe} duree={1.2}>
							<LigneLegende couleur={C['text-secondary']} opacite={0.85}>erreur observée</LigneLegende>
						</Apparition>
					</Colonne>
					<Apparition t={t} debut={H.reste} duree={0.6}>
						<Serif taille={24}>le reste ne vient pas du hasard : il vient de la fabrication du sondage</Serif>
					</Apparition>
				</Colonne>
			</div>

			<Apparition t={t} debut={H.axes} duree={0.8} absolu>
				<div style={{position: 'absolute', left: gauche, top: yBas - 1, width: droite - gauche, height: 2, background: G.axis}} />
				{tailles.map((tr, i) => (
					<Txt key={tr.n} x={X(centre(i))} y={yBas + 0.12 * UNITE} taille={13} ax={0.5} couleur={C['text-secondary']}>
						{fr(tr.n, 0)}
					</Txt>
				))}
				<Txt x={milieuX} y={yBas + 0.12 * UNITE + F(13) + 0.1 * UNITE} taille={13} ax={0.5} couleur={C['text-muted']}>
					taille du sondage
				</Txt>
			</Apparition>

			{tailles.map((tr, i) => {
				const x = x0 + i * pas;
				const hTh = (tr.th / E_MAX) * hautMax;
				const hObs = (tr.obs / E_MAX) * hautMax;
				// Les barres poussent depuis l'axe, l'une après l'autre.
				const hThVu = hTh * echelonne(t, H.hasard, 1, tailles.length, i, 0.1);
				const hObsVu = hObs * echelonne(t, H.observe, 1.2, tailles.length, i, 0.1);
				return (
					<React.Fragment key={tr.n}>
						<div style={{position: 'absolute', left: X(x - lb / 2), top: Y(baseY + hThVu), width: lb * UNITE, height: hThVu * UNITE, background: C.accent, opacity: 0.6}} />
						<div
							style={{
								position: 'absolute',
								left: X(x + lb + ecartBarres - lb / 2),
								top: Y(baseY + hObsVu),
								width: lb * UNITE,
								height: hObsVu * UNITE,
								background: C['text-secondary'],
								opacity: 0.85,
							}}
						/>
						<div style={{opacity: echelonne(t, H.rapports, 1, tailles.length, i, 0.1)}}>
							<Txt x={X(centre(i))} y={Y(baseY + hObs) - 0.1 * UNITE} taille={22} ax={0.5} ay={1} police={SERIF} couleur={C['text-primary']}>
								{`×${fr(tr.obs / tr.th)}`}
							</Txt>
						</div>
					</React.Fragment>
				);
			})}
		</AbsoluteFill>
	);
};

const LigneLegende: React.FC<{couleur: string; opacite: number; children: React.ReactNode}> = ({couleur, opacite, children}) => (
	<div style={{display: 'flex', alignItems: 'center', gap: 0.15 * UNITE}}>
		<div style={{width: 0.3 * UNITE, height: 0.2 * UNITE, background: couleur, opacity: opacite}} />
		<Libelle taille={14} couleur={C['text-primary']}>{children}</Libelle>
	</div>
);

const Critique: React.FC<{t: number}> = ({t}) => {
	const K = CRITIQUE;
	const questions = ['tout le monde peut avoir une opinion ?', 'toutes les opinions se valent ?', 'tout le monde s’accorde sur les questions à poser ?'];
	return (
		<SousTete ecart={0.5} gap={0.6} opacite={sortie(t, K.sortie)}>
			<Colonne gap={0.3}>
				<Apparition t={t} debut={K.bourdieu} duree={0.6}>
					<Libelle taille={16} couleur={C.accent}>pierre bourdieu · « l’opinion publique n’existe pas » · 1973</Libelle>
				</Apparition>
				<Colonne gap={0.15}>
					{questions.map((q, k) => (
						<Apparition key={q} t={t} debut={K.questions[k]} duree={0.5} decalage={[0, 0.1 * UNITE]}>
							<Serif taille={26}>{q}</Serif>
						</Apparition>
					))}
				</Colonne>
				<Apparition t={t} debut={K.artefact} duree={0.6}>
					<Citation auteur="Les Temps modernes, n° 318, 1973">un artefact pur et simple</Citation>
				</Apparition>
			</Colonne>
			<Colonne gap={0.18}>
				<Apparition t={t} debut={K.deze} duree={0.8}>
					<Libelle taille={16} couleur={C.accent}>alexandre dézé · 10 leçons sur les sondages politiques · 2022</Libelle>
				</Apparition>
				<Apparition t={t} debut={K.deze} duree={0.8}>
					<Serif taille={26}>échantillons par quotas, redressements, formulation des questions</Serif>
				</Apparition>
				<Apparition t={t} debut={K.photo} duree={0.8}>
					<Serif taille={30}>une « photographie de l’opinion » ?</Serif>
				</Apparition>
				<Apparition t={t} debut={K.photo} duree={0.8}>
					<Libelle taille={13} couleur={C['text-muted']}>la formule des instituts, mise en question (leçon 3)</Libelle>
				</Apparition>
			</Colonne>
		</SousTete>
	);
};

const Panels: React.FC<{t: number}> = ({t}) => {
	const K = PANELS;
	return (
		<SousTete ecart={0.6} gap={0.4} opacite={sortie(t, K.sortie)}>
			<Apparition t={t} debut={K.nom} duree={0.8}>
				<Serif taille={34}>Un angle mort récent : les panels en ligne</Serif>
			</Apparition>
			<Colonne gap={0.45}>
				<Apparition t={t} debut={K.citations[0]} duree={0.6} decalage={[0, 0.1 * UNITE]}>
					<Citation auteur="Alexandre Dézé, 2022 · inscrit sous une fausse identité" taille={26}>
						inscription […] sans condition et sans contrôle
					</Citation>
				</Apparition>
				<Apparition t={t} debut={K.citations[1]} duree={0.6} decalage={[0, 0.1 * UNITE]}>
					<Citation auteur="Mathieu Gallard, Ipsos, 2022 · sur l’infiltration des panels" taille={26}>
						pas impossible, mais […] si décourageante que ce doit être extrêmement rare
					</Citation>
				</Apparition>
			</Colonne>
			<Apparition t={t} debut={K.limite} duree={0.6}>
				<Libelle taille={15} couleur={C['text-muted']}>cette étude ne permet ni de détecter ni d’exclure une manipulation</Libelle>
			</Apparition>
		</SousTete>
	);
};

/** Un mot de la triade, avec sa note dessous (centrée, sans élargir la boîte du mot). */
const Mot: React.FC<{mot: string; note: string; opacite: number; children?: React.ReactNode}> = ({mot, note, opacite, children}) => (
	<div style={{position: 'relative'}}>
		<div style={{fontFamily: SERIF, fontSize: F(60), lineHeight: 1, color: C['text-primary'], whiteSpace: 'pre', opacity: opacite}}>
			<Titre mot={mot} />
		</div>
		{children}
		<div style={{position: 'absolute', left: '50%', top: '100%', marginTop: 0.25 * UNITE, transform: 'translateX(-50%)'}}>
			<Libelle taille={14} couleur={C['text-muted']}>{note}</Libelle>
		</div>
	</div>
);

const Mots: React.FC<{t: number}> = ({t}) => {
	const K = MOTS;
	const u = (debut: number) => avance(t, debut, 0.8);
	const barre = avance(t, K.barre, 0.6);
	return (
		<AbsoluteFill style={{opacity: sortie(t, K.sortie)}}>
			<SousTete ecart={0.6} gap={0}>
				<Apparition t={t} debut={K.photographie} duree={0.8}>
					<Citation auteur="Mathieu Gallard, Ipsos, 2022" taille={24}>un sondage n’est pas une prédiction</Citation>
				</Apparition>
			</SousTete>
			<div
				style={{
					position: 'absolute',
					left: '50%',
					top: Y(0.6 - 1.1),
					transform: 'translate(-50%, -50%)',
					display: 'flex',
					gap: 1.1 * UNITE,
				}}
			>
				<div style={{opacity: u(K.photographie)}}>
					<Mot mot="photographie" note="selon les instituts" opacite={1 - 0.6 * barre}>
						<svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: barre, overflow: 'visible'}} preserveAspectRatio="none" viewBox="0 0 1 1">
							<line x1="0" y1="0" x2="1" y2="1" stroke={C['text-muted']} strokeWidth={3} vectorEffect="non-scaling-stroke" />
							<line x1="0" y1="1" x2="1" y2="0" stroke={C['text-muted']} strokeWidth={3} vectorEffect="non-scaling-stroke" />
						</svg>
					</Mot>
				</div>
				<div style={{opacity: u(K.prediction), transform: `translateX(${(1 - u(K.prediction)) * 0.15 * UNITE}px)`}}>
					<Mot mot="prédiction" note="jugée le soir du vote" opacite={1} />
				</div>
				<div style={{opacity: u(K.presage), transform: `translateX(${(1 - u(K.presage)) * 0.15 * UNITE}px)`}}>
					<Mot mot="présage" note="quand l’erreur est commune" opacite={1} />
				</div>
			</div>
		</AbsoluteFill>
	);
};

const Fin: React.FC<{t: number}> = ({t}) => {
	const K = FORMULE;
	const u = (debut: number, duree: number) => avance(t, debut, duree);
	const serif = (taille: number): React.CSSProperties => ({fontFamily: SERIF, fontSize: F(taille), lineHeight: 1, color: C['text-primary'], whiteSpace: 'pre'});
	return (
		<div
			style={{
				position: 'absolute',
				left: '50%',
				top: Y(-0.4),
				transform: 'translate(-50%, -50%)',
				display: 'flex',
				flexDirection: 'column',
				alignItems: 'flex-start',
				gap: 0.55 * UNITE,
			}}
		>
			<Colonne gap={0.2}>
				<div style={{...serif(46), opacity: u(K.premiere, 1), transform: `translateY(${(1 - u(K.premiere, 1)) * 0.15 * UNITE}px)`}}>
					<Titre avant="Une prédiction plus qu’une " mot="photographie" apres="," />
				</div>
				<div style={{...serif(46), opacity: u(K.seconde, 1), transform: `translateY(${(1 - u(K.seconde, 1)) * 0.15 * UNITE}px)`}}>
					<Titre avant="un présage plus qu’une " mot="prédiction" apres="." />
				</div>
			</Colonne>
			<Apparition t={t} debut={K.ensemble} duree={0.8}>
				<Serif taille={28} couleur={C['text-secondary']}>quand tous les présages disent la même chose, ils peuvent se tromper ensemble</Serif>
			</Apparition>
			<Apparition t={t} debut={K.lien} duree={0.6}>
				<Libelle taille={16} couleur={C['text-muted']}>calculs, données et graphiques interactifs : lien sous la vidéo</Libelle>
			</Apparition>
		</div>
	);
};

export const Presage: React.FC = () => {
	const t = useTemps();
	return (
		<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
			{t < HASARD.axes && <Precautions t={t} />}
			{t >= HASARD.axes && t < CRITIQUE.bourdieu && <Hasard t={t} />}
			{t >= CRITIQUE.bourdieu && t < PANELS.nom && <Critique t={t} />}
			{t >= PANELS.nom && t < MOTS.photographie && <Panels t={t} />}
			{t >= MOTS.photographie && t < FORMULE.premiere && <Mots t={t} />}
			{t >= FORMULE.premiere && <Fin t={t} />}
		</AbsoluteFill>
	);
};

export const dureePresage = Math.round(DUREE_TOTALE * FPS);
