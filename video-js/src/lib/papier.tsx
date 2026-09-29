// Papier découpé aux ciseaux (étiquettes, banderole) et fenêtre terminal de la charte.
import React from 'react';
import {C, MONO, PAPIER, RAYON} from './charte';
import {hasard} from './outils';

/**
 * Morceau de papier : polygone aux sommets un peu déplacés, bord line-strong. Il remplit son parent
 * (coordonnées en % de sa boîte, ou dans `vue` si on la donne) ; `bruit` est l'écart des sommets, dans les mêmes unités.
 */
export const Papier: React.FC<{
	points: [number, number][];
	graine: number;
	bruit?: [number, number];
	vue?: string;
	style?: React.CSSProperties;
}> = ({points, graine, bruit = [0.8, 2.5], vue = '0 0 100 100', style}) => {
	const r = hasard(graine);
	const sommets = points.map(([x, y]) => `${x + r.normale(0, bruit[0])},${y + r.normale(0, bruit[1])}`).join(' ');
	return (
		<svg
			viewBox={vue}
			preserveAspectRatio="none"
			style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible', ...style}}
		>
			<polygon
				points={sommets}
				fill={PAPIER}
				stroke={C['line-strong']}
				strokeWidth={1.5}
				vectorEffect="non-scaling-stroke"
				strokeLinejoin="round"
			/>
		</svg>
	);
};

export const RECTANGLE: [number, number][] = [
	[0, 100],
	[0, 0],
	[100, 0],
	[100, 100],
];

/** Fenêtre terminal (guide/motifs.md, rr-components.css) : rayon 4 px, bordure line-strong 1 px, fond bg-card,
 * barre « dossier $ commande » séparée du corps par un filet tireté, statut à droite, sans pastilles. */
export const Terminal: React.FC<{
	dossier: string;
	commande: string;
	statut: string;
	taille: number;
	marge: number;
	children: React.ReactNode;
}> = ({dossier, commande, statut, taille, marge, children}) => (
	<div
		style={{
			border: `1px solid ${C['line-strong']}`,
			borderRadius: RAYON,
			background: C['bg-card'],
			fontFamily: MONO,
			fontSize: taille,
			overflow: 'hidden',
		}}
	>
		<div
			style={{
				display: 'flex',
				justifyContent: 'space-between',
				alignItems: 'baseline',
				gap: '1em',
				padding: '0.6em 1.2em',
				borderBottom: `1px dashed ${C.line}`,
				color: C['text-muted'],
				lineHeight: 1.2,
			}}
		>
			<span>
				{dossier} $ {commande}
			</span>
			<span>
				<span style={{color: C.sage}}>●</span> {statut}
			</span>
		</div>
		<div style={{padding: marge, lineHeight: 0}}>{children}</div>
	</div>
);
