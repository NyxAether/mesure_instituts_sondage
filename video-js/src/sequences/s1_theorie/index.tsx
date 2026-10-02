// Séquence 1 : la théorie, ce que veut dire « ± 3 points » (port de video/scenes/s1_theorie.py).
import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, MONO, SERIF} from '../../lib/charte';
import {F, FPS, NBSP, Txt, UNITE, avance, fr} from '../../lib/outils';
import {X, Y} from '../../lib/axes';
import {Conclusion} from './Conclusion';
import {D1, M_GRAND, M_PETIT, entier, texEntier, texNombre} from './donnees';
import {Calque, Fondu} from './Fondu';
import {Formule} from './Formule';
import {AXE_CENTRE, POP_GAUCHE, POP_HAUTEUR, POP_CENTRE, couleurVote, xAxe} from './geometrie';
import {AxeSvg, AxeTexte, Barres, BandeTexte, Compteur, FlecheSvg, Traits, ZoneSvg} from './Histogramme';
import {Envols, Population, lentEnCours} from './Population';
import {D, DUREE_LENT, T} from './temps';

// JetBrains Mono : le haut de la boîte CSS (line-height 1) est 0,13 em au-dessus du haut des chiffres.
const HAUT_MONO = 0.13;
const CAPITALE_MONO = 0.73;
// Écarts constatés entre l'encre de Manim et les boîtes CSS (px, mesurés sur les images extraites des deux rendus).
const AJUSTE_ETIQUETTE = 9; // « vote a dans l'échantillon » est plus proche du résultat en Manim
const AJUSTE_ECHANTILLON = 3;
// Espaces (unités) entre les boîtes des formules : ceux de la scène (buff 0.45, 0.45, 0.35, 0.5) mesurent l'encre, pas la boîte KaTeX.
const ECART_FORMULES = [0.394, 0.283, 0.172, 0.422];

const Puce: React.FC<{couleur: string; contenu: string}> = ({couleur, contenu}) => (
	<div style={{display: 'flex', alignItems: 'center', gap: 0.12 * UNITE}}>
		<div style={{width: 0.12 * UNITE, height: 0.12 * UNITE, borderRadius: '50%', background: couleur}} />
		<div>{contenu}</div>
	</div>
);

/** Titre, légende et taille de l'échantillon autour de la population, puis résultat d'un tirage lent. */
const TextesPopulation: React.FC<{t: number}> = ({t}) => {
	const sortie = 1 - avance(t, T.populationSortie, D.populationSortie);
	const bas = POP_CENTRE[1] - POP_HAUTEUR / 2;
	const haut = POP_CENTRE[1] + POP_HAUTEUR / 2;
	const yLegende = bas - 0.3; // haut de la légende (encre)
	const yEchantillon = yLegende - (CAPITALE_MONO * F(16)) / UNITE - 0.25;
	const lent = lentEnCours(t);
	return (
		<Fondu o={sortie}>
			<Txt x={X(POP_GAUCHE)} y={Y(haut + 0.25)} ay={1} taille={17} couleur={C['text-secondary']} opacite={avance(t, T.population, D.titrePopulation)}>
				population · des millions d’électeurs
			</Txt>
			<div
				style={{
					position: 'absolute',
					left: X(POP_GAUCHE),
					top: Y(yLegende) - HAUT_MONO * F(16),
					display: 'flex',
					gap: 0.5 * UNITE,
					fontFamily: MONO,
					fontSize: F(16),
					lineHeight: 1,
					whiteSpace: 'pre',
					color: C['text-secondary'],
					opacity: avance(t, T.legende, D.legende),
				}}
			>
				<Puce couleur={couleurVote(0)} contenu={`vote a · ${Math.round(D1.p * 100)}${NBSP}%`} />
				<Puce couleur={couleurVote(1)} contenu="vote b" />
			</div>
			<Txt x={X(POP_GAUCHE)} y={Y(yEchantillon) - HAUT_MONO * F(17) + AJUSTE_ECHANTILLON} taille={17} couleur={C['text-primary']} opacite={avance(t, T.echantillon, D.echantillon)}>
				{`échantillon · ${entier(D1.taille_petit)} personnes`}
			</Txt>
			{lent && lent.u < D.lentChoix + D.lentPause + D.lentEnvol && <Resultat i={lent.i} u={lent.u} />}
		</Fondu>
	);
};

/** Résultat du tirage lent i, à gauche du haut de la vraie valeur ; entre en montant, sort en descendant. */
const Resultat: React.FC<{i: number; u: number}> = ({i, u}) => {
	const entree = avance(u, 0, D.lentChoix);
	const sortie = avance(u, D.lentChoix + D.lentPause, D.lentEnvol);
	return (
		<div
			style={{
				position: 'absolute',
				left: xAxe(50) - 0.3 * UNITE,
				top: Y(AXE_CENTRE[1] + 2.3),
				transform: `translate(-100%, calc(-50% + ${(1 - entree) * 0.1 * UNITE + sortie * 0.5 * UNITE}px))`,
				opacity: entree * (1 - sortie),
				display: 'flex',
				flexDirection: 'column',
				alignItems: 'center',
				gap: 0.1 * UNITE,
				whiteSpace: 'pre',
			}}
		>
			<div style={{fontFamily: MONO, fontSize: F(15), lineHeight: 1, color: C['text-secondary'], position: 'relative', top: AJUSTE_ETIQUETTE}}>vote a dans l’échantillon</div>
			<div style={{fontFamily: SERIF, fontSize: F(56), lineHeight: 1, color: C['text-primary']}}>{`${fr(D1.t1000[i])}${NBSP}%`}</div>
		</div>
	);
};

/** Formules à gauche : σ, marge à 95 %, puis les deux calculs (n = 1 000 puis n = 4 000) et la règle. */
const Formules: React.FC<{t: number}> = ({t}) => {
	if (t < T.formule) return null;
	const decale = (e: number, sens: number) => sens * (1 - e) * 0.1 * UNITE; // FadeIn(shift = 0.1 vers le bas / le haut)
	const eMarge = avance(t, T.margeTex, D.margeTex);
	const eCalcul = avance(t, T.calcul, D.calcul);
	const eQuatre = avance(t, T.quatre, D.quatre);
	const eRegle = avance(t, T.regle, D.regle);
	const enfant = (haut: number): React.CSSProperties => ({position: 'absolute', left: 0, top: '100%', marginTop: haut * UNITE});
	return (
		<div style={{position: 'absolute', left: X(-3.75), top: Y(0.25), transform: 'translate(-50%, -50%)', display: 'flex', flexDirection: 'column', alignItems: 'flex-start'}}>
			<Formule latex={String.raw`\sigma = \sqrt{\frac{p\,(1-p)}{n}}`} taille={46} ecriture={avance(t, T.formule, D.formule)} />
			<div style={{marginTop: ECART_FORMULES[0] * UNITE, transform: `translateY(${-decale(eMarge, 1)}px)`, opacity: eMarge}}>
				<Formule latex={String.raw`\text{marge à 95\,\%} = ${texNombre(D1.z95, 2)}\,\sigma`} taille={40} />
			</div>
			<div style={{position: 'relative', marginTop: ECART_FORMULES[1] * UNITE, transform: `translateY(${-decale(eCalcul, 1)}px)`, opacity: eCalcul}}>
				<Formule latex={String.raw`n = ${texEntier(D1.taille_petit)} \;\Rightarrow\; \pm\,${texNombre(M_PETIT)}\ \text{points}`} taille={40} />
				<div style={{...enfant(ECART_FORMULES[2]), transform: `translateY(${-decale(eQuatre, 1)}px)`, opacity: eQuatre}}>
					<Formule latex={String.raw`n = ${texEntier(D1.taille_grand)} \;\Rightarrow\; \pm\,${texNombre(M_GRAND)}\ \text{points}`} taille={40} couleur={C.accent} />
					<div style={{...enfant(ECART_FORMULES[3]), transform: `translateY(${decale(eRegle, 1)}px)`, opacity: eRegle, fontFamily: SERIF, fontSize: F(28), lineHeight: 1, whiteSpace: 'pre', color: C['text-primary']}}>
						{`4 × plus de monde${NBSP}: marge ÷${NBSP}2`}
					</div>
				</div>
			</div>
		</div>
	);
};

export const Theorie: React.FC = () => {
	const t = useCurrentFrame() / FPS;
	// « tout » (axe, histogramme, formules...) s'efface avant l'écran final ; l'en-tête reste.
	const sortie = 1 - avance(t, T.sortie, D.sortie);
	return (
		<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
			<Calque>
				<Population t={t} />
				<g opacity={sortie}>
					<ZoneSvg t={t} />
					<AxeSvg t={t} />
					<Traits t={t} />
					<Barres t={t} />
					<FlecheSvg t={t} />
				</g>
				<Envols t={t} />
			</Calque>
			<TextesPopulation t={t} />
			<Fondu o={sortie}>
				<AxeTexte t={t} />
				<Compteur t={t} />
				<BandeTexte t={t} />
				<Formules t={t} />
			</Fondu>
			<Conclusion t={t} />
		</AbsoluteFill>
	);
};

export const dureeTheorie = Math.round(T.fin * FPS);
