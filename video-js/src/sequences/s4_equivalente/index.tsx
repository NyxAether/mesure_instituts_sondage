import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C} from '../../lib/charte';
import {Entete} from '../../lib/composants';
import {FPS, NBSP, UNITE as U, avance} from '../../lib/outils';
import {Courbes} from './Courbes';
import {Entonnoir} from './Entonnoir';
import {Constat, Etude} from './Etude';
import {Jours} from './Jours';
import {CONSTAT, ENTONNOIR, ETUDE, JOURS, LISSAGE} from './temps';

// Plans successifs : chacun disparaît (0,8 s) avant que le suivant n'apparaisse ; l'en-tête reste tout du long.
const visible = (t: number, debut: number, fin: number) => t >= debut && t < fin + 0.8;

export const Equivalente: React.FC = () => {
	const t = useCurrentFrame() / FPS;
	const a = avance(t, ENTONNOIR.entete, 1);
	return (
		<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
			<div style={{position: 'absolute', inset: 0, transform: `translateY(${-0.15 * U * (1 - a)}px)`}}>
				<Entete numero={4} nom="la taille équivalente" avant="Combien vaut " mot="vraiment" apres={`${NBSP}un sondage${NBSP}?`} opacite={a} />
			</div>
			{visible(t, ENTONNOIR.entete, ENTONNOIR.sortie) && <Entonnoir t={t} />}
			{visible(t, ETUDE.titre, ETUDE.sortie) && <Etude t={t} />}
			{visible(t, LISSAGE.grille, LISSAGE.sortie) && <Courbes t={t} />}
			{visible(t, JOURS.grille, JOURS.sortie) && <Jours t={t} />}
			{t >= CONSTAT.titre && <Constat t={t} />}
		</AbsoluteFill>
	);
};

export const dureeEquivalente = Math.round(CONSTAT.fin * FPS);
