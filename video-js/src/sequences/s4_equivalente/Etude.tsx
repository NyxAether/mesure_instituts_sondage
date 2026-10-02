// La mesure de l'étude (taille annoncée, équivalente, témoin) et l'écran final.
import React from 'react';
import donnees from '../../../donnees/s4.json';
import {C, MONO, SERIF} from '../../lib/charte';
import {Titre} from '../../lib/composants';
import {F, UNITE as U, X, Y, avance, fr} from '../../lib/outils';
import {DISCRET, Fondu, NBSP, SERIE, apparition, entier} from './commun';
import {SousEntete} from '../../lib/composants';
import {CONSTAT, ETUDE} from './temps';

const nReel = donnees.median_reel;
const nKl = donnees.medianes.optimal_kl;
const nTemoin = donnees.medianes.oneshot;
const jours = donnees.jours;
if (jours.reduce((s, b) => s + b.effectif, 0) !== donnees.effectif_30_jours) throw new Error('boîtes incomplètes');

const LIGNES = [
	{nom: 'taille annoncée', valeur: nReel, couleur: C['text-primary'], detail: 'sondages réels'},
	{nom: 'taille équivalente', valeur: nKl, couleur: SERIE, detail: `÷${NBSP}${fr(nReel / nKl, 0)}`},
	{nom: 'témoin', valeur: nTemoin, couleur: C['text-secondary'], detail: 'un vrai tirage aléatoire par sondage : la méthode retrouve la taille'},
];

export const Etude: React.FC<{t: number}> = ({t}) => (
	<div style={{position: 'absolute', inset: 0, opacity: 1 - avance(t, ETUDE.sortie, 0.8)}}>
		<Fondu opacite={apparition(t, ETUDE.titre, 0.6)}>
			<SousEntete>
				{`la mesure de l’étude, sondage par sondage · ${fr(donnees.nb_proches, 0)} sondages des 14 derniers jours · médianes`}
			</SousEntete>
		</Fondu>
		<div style={{position: 'absolute', left: X(-0.5) + 8, top: Y(-0.6) + 8, transform: 'translate(-50%, -50%)', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 0.405 * U}}>
			{LIGNES.map((l, k) => {
				const a = avance(t, ETUDE.lignes[k], 0.7);
				return (
					<div
						key={l.nom}
						style={{display: 'flex', alignItems: 'baseline', gap: 0.3 * U, opacity: a, transform: `translateY(${0.1 * U * (1 - a)}px)`, whiteSpace: 'pre'}}
					>
						<span style={{fontFamily: MONO, fontSize: F(16), lineHeight: 1, color: C['text-primary']}}>{l.nom}</span>
						<span style={{fontFamily: SERIF, fontSize: F(40), lineHeight: 0.8, color: l.couleur}}>{fr(l.valeur, 0)}</span>
						<span style={{fontFamily: MONO, fontSize: F(14), lineHeight: 1, color: DISCRET}}>{l.detail}</span>
					</div>
				);
			})}
		</div>
	</div>
);

export const Constat: React.FC<{t: number}> = ({t}) => {
	const premier = jours[0];
	const dernier = jours[jours.length - 1];
	return (
		<div style={{position: 'absolute', left: X(0), top: Y(-0.4), transform: 'translate(-50%, -50%)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.35 * U, whiteSpace: 'pre', fontFamily: SERIF, lineHeight: 1}}>
			<div style={{fontSize: F(64), color: C['text-primary'], opacity: avance(t, CONSTAT.titre, 1), transform: `translateY(${0.15 * U * (1 - avance(t, CONSTAT.titre, 1))}px)`}}>
				<Titre avant={`${fr(nReel, 0)} sondés, la précision de `} mot={fr(nKl, 0)} />
			</div>
			<div style={{fontSize: F(28), color: C['text-primary'], opacity: avance(t, CONSTAT.detail, 0.6)}}>
				{`un sondage se comporte comme un tirage aléatoire de ${fr(nKl, 0)} personnes · ${fr(nReel / nKl, 0)} fois moins`}
			</div>
			<div style={{fontSize: F(28), color: C['text-secondary'], opacity: avance(t, CONSTAT.temps, 0.6)}}>
				{`${fr(entier(premier.med), 0)} dans les ${premier.tranche.split('–')[1]} derniers jours, ${fr(entier(dernier.med), 0)} un mois avant`}
			</div>
		</div>
	);
};
