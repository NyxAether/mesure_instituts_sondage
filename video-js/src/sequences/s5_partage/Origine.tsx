// D'où vient l'erreur commune ? Deux colonnes : ce qu'avancent les instituts, ce que pointent les chercheurs.
import React from 'react';
import {C, G, MONO, SERIF} from '../../lib/charte';
import {F, UNITE as U, Y, avance} from '../../lib/outils';
import {Calque} from './commun';
import {DEBUT_LIGNES, ORIGINE, PAS_LIGNE} from './temps';

type Colonne = {nom: string; couleur: string; debut: number; lignes: [string, string][]};

// Textes de l'utilisateur, repris tels quels de la scène Manim.
const COLONNES: Colonne[] = [
	{
		nom: 'ce qu’avancent les instituts',
		couleur: G.series[1],
		debut: ORIGINE.instituts,
		lignes: [
			['l’opinion bouge jusqu’au dernier moment', 'Mathieu Gallard, Ipsos, 2022'],
			['l’abstention est mal anticipée', 'Mathieu Gallard, Ipsos, 2022'],
			['la méthode a parfois une élection de retard', 'Mathieu Gallard, Ipsos, 2022'],
		],
	},
	{
		nom: 'ce que pointent les chercheurs',
		couleur: G.series[0],
		debut: ORIGINE.chercheurs,
		lignes: [
			['tout le monde n’a pas d’opinion sur tout', 'Pierre Bourdieu, 1973'],
			['des questions que les sondés ne se posent pas', 'Alexandre Dézé, 2022'],
			['échantillons douteux, redressements opaques', 'Alexandre Dézé, 2022'],
		],
	},
];

const Bloc: React.FC<{col: Colonne; t: number}> = ({col, t}) => (
	<div style={{display: 'flex', flexDirection: 'column', gap: 0.25 * U}}>
		<div style={{fontFamily: MONO, fontSize: F(16), color: col.couleur, opacity: avance(t, col.debut, 0.5)}}>{col.nom}</div>
		<div style={{display: 'flex', gap: 0.25 * U, alignItems: 'stretch'}}>
			<div style={{width: 3, background: col.couleur, opacity: avance(t, col.debut, 0.5)}} />
			<div style={{display: 'flex', flexDirection: 'column', gap: 0.3 * U}}>
				{col.lignes.map(([texte, source], k) => {
					const u = avance(t, col.debut + DEBUT_LIGNES + k * PAS_LIGNE, 0.5);
					return (
						<div key={texte} style={{display: 'flex', flexDirection: 'column', gap: 0.06 * U, opacity: u, transform: `translateY(${0.1 * U * (1 - u)}px)`}}>
							<div style={{fontFamily: SERIF, fontSize: F(24), color: C['text-primary']}}>{texte}</div>
							<div style={{fontFamily: MONO, fontSize: F(13), color: C['text-muted']}}>{source}</div>
						</div>
					);
				})}
			</div>
		</div>
	</div>
);

export const Origine: React.FC<{t: number}> = ({t}) => (
	<Calque opacite={1 - avance(t, ORIGINE.sortie, 0.8)}>
		<div
			style={{
				position: 'absolute',
				left: '50%',
				top: Y(-0.6),
				transform: 'translate(-50%, -50%)',
				display: 'flex',
				flexDirection: 'column',
				alignItems: 'flex-start',
				gap: 0.3 * U,
				whiteSpace: 'pre',
				lineHeight: 1,
			}}
		>
			<div style={{fontFamily: SERIF, fontSize: F(36), color: C['text-primary'], opacity: avance(t, ORIGINE.question, 0.8)}}>D’où vient cette erreur commune ?</div>
			<div style={{fontFamily: MONO, fontSize: F(15), color: C['text-muted'], opacity: avance(t, ORIGINE.question, 0.8), marginTop: -0.12 * U}}>
				des explications, pas des résultats : l’étude mesure l’erreur, pas son origine
			</div>
			<div style={{display: 'flex', alignItems: 'flex-start', gap: 0.55 * U}}>
				{COLONNES.map((col) => (
					<Bloc key={col.nom} col={col} t={t} />
				))}
			</div>
			<div
				style={{
					position: 'absolute',
					top: '100%',
					left: 0,
					marginTop: 0.45 * U,
					fontFamily: SERIF,
					fontSize: F(26),
					color: C['text-secondary'],
					opacity: avance(t, ORIGINE.commun, 0.6),
				}}
			>
				dans tous les cas, une erreur que multiplier les sondages ne dilue pas
			</div>
		</div>
	</Calque>
);
