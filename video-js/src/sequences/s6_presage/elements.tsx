// Petits éléments de mise en page propres à la séquence 6 (texte serif, libellé mono, citation, apparition).
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {C, MONO, SERIF} from '../../lib/charte';
import {avance, F, FPS, UNITE} from '../../lib/outils';

/** Temps courant en secondes. */
export const useTemps = () => useCurrentFrame() / FPS;

/** Colonne alignée à gauche, espacement en unités Manim (VGroup.arrange(DOWN, aligned_edge=LEFT)). */
export const Colonne: React.FC<{gap: number; children: React.ReactNode}> = ({gap, children}) => (
	<div style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: gap * UNITE}}>{children}</div>
);

/** Texte éditorial en Newsreader (sous_titre de theme.py). */
export const Serif: React.FC<{taille: number; couleur?: string; children: React.ReactNode}> = ({taille, couleur = C['text-primary'], children}) => (
	<div style={{fontFamily: SERIF, fontSize: F(taille), lineHeight: 1, color: couleur, whiteSpace: 'pre'}}>{children}</div>
);

/** Libellé terminal en JetBrains Mono (libelle de theme.py). */
export const Libelle: React.FC<{taille: number; couleur?: string; children: React.ReactNode}> = ({taille, couleur = C['text-secondary'], children}) => (
	<div style={{fontFamily: MONO, fontSize: F(taille), lineHeight: 1, color: couleur, whiteSpace: 'pre'}}>{children}</div>
);

/** Citation en Newsreader, auteur en libellé dessous, filet prune à gauche. */
export const Citation: React.FC<{auteur: string; taille?: number; children: string}> = ({auteur, taille = 26, children}) => (
	<div style={{display: 'flex', alignItems: 'stretch', gap: 0.25 * UNITE}}>
		<div style={{width: 3, background: C.accent}} />
		<Colonne gap={0.1}>
			<Serif taille={taille}>{`« ${children} »`}</Serif>
			<Libelle taille={13} couleur={C['text-muted']}>{auteur}</Libelle>
		</Colonne>
	</div>
);

/** FadeIn de Manim : fondu de `duree` s à partir de `debut`, avec un décalage [dx, dy] en px résorbé pendant le fondu. */
export const Apparition: React.FC<{
	t: number;
	debut: number;
	duree: number;
	decalage?: [number, number];
	absolu?: boolean;
	children: React.ReactNode;
}> = ({t, debut, duree, decalage = [0, 0], absolu = false, children}) => {
	const u = avance(t, debut, duree);
	return (
		<div
			style={{
				opacity: u,
				transform: `translate(${(1 - u) * decalage[0]}px, ${(1 - u) * decalage[1]}px)`,
				...(absolu ? {position: 'absolute', left: 0, top: 0} : {}),
			}}
		>
			{children}
		</div>
	);
};
