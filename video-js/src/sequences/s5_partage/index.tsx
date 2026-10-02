import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C} from '../../lib/charte';
import {FPS} from '../../lib/outils';
import {Constat} from './Constat';
import {Exemple} from './Exemple';
import {Mesures} from './Mesures';
import {Origine} from './Origine';
import {CONSENSUS, CONSTAT, FIN, HASARD, MIMETISME, ORIGINE, REEL} from './temps';

// Chaque plan n'est monté que pendant sa durée : hors de là, rien à peindre.
const dans = (t: number, debut: number, fin: number) => t >= debut && t < fin;

export const Partage: React.FC = () => {
	const t = useCurrentFrame() / FPS;
	return (
		<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
			{dans(t, HASARD.axes - 0.6, REEL.fin) && <Exemple t={t} />}
			{dans(t, CONSENSUS.axe, MIMETISME.fin) && <Mesures t={t} />}
			{dans(t, ORIGINE.question, ORIGINE.fin) && <Origine t={t} />}
			{dans(t, CONSTAT.constat, FIN) && <Constat t={t} />}
		</AbsoluteFill>
	);
};

export const dureePartage = Math.round(FIN * FPS);
