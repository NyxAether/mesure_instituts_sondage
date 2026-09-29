// Petites briques propres à la séquence 5 : boîte ancrée à plusieurs lignes et calque qui apparaît / disparaît.
import React from 'react';
import {MONO} from '../../lib/charte';
import {F} from '../../lib/outils';

/** Calque plein cadre : opacité, et décalage en px (positif = vers le bas). */
export const Calque: React.FC<{opacite?: number; dy?: number; children: React.ReactNode}> = ({opacite = 1, dy = 0, children}) => (
	<div style={{position: 'absolute', inset: 0, opacity: opacite, transform: dy ? `translateY(${dy}px)` : undefined, pointerEvents: 'none'}}>{children}</div>
);

/** Boîte de texte à plusieurs lignes placée en px, ancrée comme `Txt` ; `align` aligne les lignes entre elles. */
export const Boite: React.FC<{
	x: number;
	y: number;
	ax?: number;
	ay?: number;
	ecart?: number;
	align?: 'flex-start' | 'center' | 'flex-end';
	opacite?: number;
	children: React.ReactNode;
}> = ({x, y, ax = 0, ay = 0, ecart = 0, align = 'flex-start', opacite = 1, children}) => (
	<div
		style={{
			position: 'absolute',
			left: x,
			top: y,
			transform: `translate(${-ax * 100}%, ${-ay * 100}%)`,
			display: 'flex',
			flexDirection: 'column',
			alignItems: align,
			gap: ecart,
			opacity: opacite,
			whiteSpace: 'pre',
			lineHeight: 1,
		}}
	>
		{children}
	</div>
);

/** Une ligne de texte à l'intérieur d'une `Boite`. */
export const Ligne: React.FC<{taille: number; police?: string; couleur?: string; hauteur?: number; children: React.ReactNode}> = ({taille, police = MONO, couleur, hauteur = 1, children}) => (
	<div style={{fontFamily: police, fontSize: F(taille), color: couleur, lineHeight: hauteur}}>{children}</div>
);
