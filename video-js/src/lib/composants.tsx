// Briques de mise en page reprises de video/theme.py : titre à un mot italique, en-tête de section, curseur.
import React from 'react';
import {C, MONO, SERIF} from './charte';
import {F, Txt, UNITE} from './outils';

/** Titre en Newsreader 400 dont un seul mot est en italique prune. Taille Manim 52 par défaut. */
export const Titre: React.FC<{avant?: string; mot: string; apres?: string; taille?: number}> = ({avant = '', mot, apres = '', taille = 52}) => (
	<>
		{avant}
		<span style={{fontStyle: 'italic', color: C.accent}}>{mot}</span>
		{apres}
	</>
);

/** En-tête « // 0N · nom » + titre, calé en haut à gauche (to_corner(UL, buff=0.55) de Manim). */
export const Entete: React.FC<{numero: number; nom: string; avant?: string; mot: string; apres?: string; opacite?: number}> = ({
	numero,
	nom,
	avant,
	mot,
	apres,
	opacite = 1,
}) => (
	<div style={{position: 'absolute', left: 0.55 * UNITE, top: 0.55 * UNITE, opacity: opacite, display: 'flex', flexDirection: 'column', gap: 0.18 * UNITE}}>
		<div style={{fontFamily: MONO, fontSize: F(17), lineHeight: 1, color: C['text-muted'], whiteSpace: 'pre'}}>
			{`// ${String(numero).padStart(2, '0')} · ${nom}`}
		</div>
		<div style={{fontFamily: SERIF, fontSize: F(52), lineHeight: 1, color: C['text-primary'], whiteSpace: 'pre'}}>
			<Titre avant={avant} mot={mot} apres={apres} />
		</div>
	</div>
);

/** Curseur de terminal qui clignote (période en secondes), à la suite d'une invite. */
export const Curseur: React.FC<{x: number; y: number; taille: number; t: number; periode?: number; couleur?: string}> = ({x, y, taille, t, periode = 1, couleur}) => (
	<Txt x={x} y={y} taille={taille} ay={0.5} couleur={couleur ?? C['text-primary']} opacite={Math.floor((t / periode) * 2) % 2 === 0 ? 1 : 0}>
		█
	</Txt>
);
