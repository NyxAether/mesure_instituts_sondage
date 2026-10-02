// Briques de mise en page reprises de video/theme.py : titre à un mot italique, en-tête de section, curseur.
import React from 'react';
import {C, MONO, SERIF, V} from './charte';
import {Txt} from './outils';

/** Titre en Newsreader 400 dont un seul mot est en italique prune. Taille Manim 52 par défaut. */
export const Titre: React.FC<{avant?: string; mot: string; apres?: string; taille?: number}> = ({avant = '', mot, apres = '', taille = 52}) => (
	<>
		{avant}
		<span style={{fontStyle: 'italic', color: C.accent}}>{mot}</span>
		{apres}
	</>
);

const H = V.header;
/** Interligne du titre d'en-tête (gabarit templates/video/titre.html). */
export const INTERLIGNE_TITRE = 1.02;
/** Bas de l'en-tête rangé, en px : le contenu placé sous lui part de là. */
export const BAS_ENTETE = V.margin.top + H.tag + H.gap + H.title * INTERLIGNE_TITRE;

/** Ligne facultative sous l'en-tête rangé (source, précision) : une seule, en mono discret. */
export const SousEntete: React.FC<{opacite?: number; children: React.ReactNode}> = ({opacite = 1, children}) => (
	<div style={{position: 'absolute', left: V.margin.x, top: BAS_ENTETE + H.subtitle.gap, fontFamily: MONO, fontSize: H.subtitle.size, lineHeight: 1, color: C['text-muted'], whiteSpace: 'pre', opacity: opacite}}>
		{children}
	</div>
);

/** En-tête « // 0N · nom » + titre, rangé dans le coin haut gauche aux marges de la charte (`video.header`). */
export const Entete: React.FC<{numero: number; nom: string; avant?: string; mot: string; apres?: string; opacite?: number; style?: React.CSSProperties}> = ({
	numero,
	nom,
	avant,
	mot,
	apres,
	opacite = 1,
	style,
}) => (
	<div style={{position: 'absolute', left: V.margin.x, top: V.margin.top, opacity: opacite, display: 'flex', flexDirection: 'column', gap: H.gap, ...style}}>
		<div style={{fontFamily: MONO, fontSize: H.tag, lineHeight: 1, color: C['text-muted'], whiteSpace: 'pre'}}>
			{`// ${String(numero).padStart(2, '0')} · ${nom}`}
		</div>
		<div style={{fontFamily: SERIF, fontSize: H.title, lineHeight: INTERLIGNE_TITRE, letterSpacing: '-.02em', color: C['text-primary'], whiteSpace: 'pre'}}>
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
