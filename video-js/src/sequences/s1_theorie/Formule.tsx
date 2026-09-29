// Formules en KaTeX (à la place des MathTex de Manim, en LaTeX / Computer Modern).
import katex from 'katex';
import 'katex/dist/katex.min.css';
import React, {useEffect, useState} from 'react';
import {continueRender, delayRender} from 'remotion';
import {C, SERIF} from '../../lib/charte';
import {UNITE} from '../../lib/outils';

/** Taille de la fonte en px pour un `font_size` de MathTex : LaTeX compose en 10 pt, et Manim ramène chaque point de font_size à 1/960 d'unité. */
export const PX_PAR_TAILLE_TEX = (10 / 960) * UNITE;

/**
 * Harmonisation avec Newsreader : les blocs \text{...} passent en Newsreader (les lettres et symboles mathématiques
 * restent en KaTeX_Math / KaTeX_Main). Désactivée par défaut, pour rester fidèle au rendu LaTeX validé.
 */
export const HARMONISER_NEWSREADER = false;

// KaTeX pose lui-même sa taille (1,21 em) sur .katex : on la ramène à 1 pour que la taille vienne de l'élément parent.
const CSS = `.formule-tex .katex{font-size:1em;} .formule-tex-harmonisee .katex .text{font-family:${SERIF};}`;

/** Attend le chargement des fontes KaTeX avant d'accepter l'image (les formules sont toujours dans le DOM, même invisibles). */
const useFontesKatex = () => {
	const [poignee] = useState(() => delayRender('fontes KaTeX'));
	useEffect(() => {
		Promise.all(['KaTeX_Main', 'KaTeX_Math', 'KaTeX_Size1', 'KaTeX_Size2', 'KaTeX_Size3', 'KaTeX_Size4'].map((f) => document.fonts.load(`1em ${f}`)))
			.then(() => document.fonts.ready)
			.then(() => continueRender(poignee));
	}, [poignee]);
};

export const Formule: React.FC<{
	latex: string;
	taille: number; // font_size de MathTex
	couleur?: string;
	/** Écriture (Write) de 0 à 1 : révélation de gauche à droite. */
	ecriture?: number;
	opacite?: number;
	style?: React.CSSProperties;
}> = ({latex, taille, couleur = C['text-primary'], ecriture = 1, opacite = 1, style}) => {
	useFontesKatex();
	// \displaystyle : MathTex compose en mode display (align*), donc les fractions gardent la taille et les espaces du texte.
	const html = katex.renderToString(String.raw`\displaystyle ${latex}`, {throwOnError: true, output: 'html', displayMode: false});
	const bord = 8; // largeur du dégradé du bord de révélation, en % de la formule
	const masque = `linear-gradient(to right, #000 ${ecriture * (100 + bord) - bord}%, transparent ${ecriture * (100 + bord)}%)`;
	return (
		<div
			className={`formule-tex${HARMONISER_NEWSREADER ? ' formule-tex-harmonisee' : ''}`}
			style={{
				fontSize: taille * PX_PAR_TAILLE_TEX,
				color: couleur,
				opacity: opacite,
				whiteSpace: 'nowrap',
				WebkitMaskImage: ecriture < 1 ? masque : undefined,
				maskImage: ecriture < 1 ? masque : undefined,
				...style,
			}}
		>
			<style>{CSS}</style>
			<span dangerouslySetInnerHTML={{__html: html}} />
		</div>
	);
};
