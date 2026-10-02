import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C} from '../../lib/charte';
import {FPS} from '../../lib/outils';
import {Courbes} from './Courbes';
import {Entonnoir} from './Entonnoir';
import {Constat, Etude} from './Etude';
import {Jours} from './Jours';
import {CONSTAT, ENTONNOIR, ETUDE, JOURS, LISSAGE} from './temps';

// Plans successifs : chacun disparaît (0,8 s) avant que le suivant n'apparaisse  (l’en-tête est posé par lib/ouverture.tsx).
const visible = (t: number, debut: number, fin: number) => t >= debut && t < fin + 0.8;

export const Equivalente: React.FC = () => {
	const t = useCurrentFrame() / FPS;
	return (
		<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
			{visible(t, ENTONNOIR.entete, ENTONNOIR.sortie) && <Entonnoir t={t} />}
			{visible(t, ETUDE.titre, ETUDE.sortie) && <Etude t={t} />}
			{visible(t, LISSAGE.grille, LISSAGE.sortie) && <Courbes t={t} />}
			{visible(t, JOURS.grille, JOURS.sortie) && <Jours t={t} />}
			{t >= CONSTAT.titre && <Constat t={t} />}
		</AbsoluteFill>
	);
};

export const dureeEquivalente = Math.round(CONSTAT.fin * FPS);
