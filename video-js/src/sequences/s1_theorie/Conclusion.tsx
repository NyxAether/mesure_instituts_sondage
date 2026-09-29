// Écran final : « ± 3 points, 19 fois sur 20 », la précision, l'exception et l'invite « > vérifions ».
import React from 'react';
import {Titre} from '../../lib/composants';
import {C, MONO, SERIF} from '../../lib/charte';
import {F, NBSP, UNITE, avance} from '../../lib/outils';
import {X, Y} from '../../lib/axes';
import {D, NB_CLIGNOTEMENTS, T} from './temps';
import {D1, M_PETIT, entier} from './donnees';
import {Fondu} from './Fondu';

/** Opacité du curseur █ : NB_CLIGNOTEMENTS fois, il disparaît puis revient (deux animations d'un quart de seconde). */
const opaciteCurseur = (t: number) => {
	let o = 1;
	for (let k = 0; k < NB_CLIGNOTEMENTS; k++) {
		const debut = T.clignote + k * 2 * D.clignote;
		o = Math.min(o, 1 - avance(t, debut, D.clignote) + avance(t, debut + D.clignote, D.clignote));
	}
	return o;
};

export const Conclusion: React.FC<{t: number}> = ({t}) => {
	if (t < T.prevision) return null;
	const ePrevision = avance(t, T.prevision, D.prevision);
	const ePrecision = avance(t, T.precision, D.precision);
	const eException = avance(t, T.exception, D.exception);
	const eInvite = avance(t, T.invite, D.invite);
	const ligne: React.CSSProperties = {fontFamily: SERIF, lineHeight: 1, whiteSpace: 'pre', color: C['text-secondary'], fontSize: F(34)};
	const marge = Math.round(M_PETIT);
	const pourcent = Math.round(D1.p * 100);
	return (
		<Fondu>
			<div
				style={{
					position: 'absolute',
					left: X(0),
					top: Y(0.2),
					transform: 'translate(-50%, -50%)',
					display: 'flex',
					flexDirection: 'column',
					alignItems: 'center',
					gap: 0.35 * UNITE,
				}}
			>
				<div style={{fontFamily: SERIF, fontSize: F(72), lineHeight: 1, whiteSpace: 'pre', color: C['text-primary'], opacity: ePrevision, transform: `translateY(${(1 - ePrevision) * 0.15 * UNITE}px)`}}>
					<Titre avant={`±${NBSP}${marge} points, `} mot="19 fois" apres=" sur 20" />
				</div>
				<div style={{...ligne, opacity: ePrecision}}>{`pour ${entier(D1.taille_petit)} personnes interrogées et un candidat à ${pourcent}${NBSP}%`}</div>
				<div style={{...ligne, opacity: eException, position: 'relative'}}>
					{`1 fois sur 20${NBSP}: plus de ${marge} points d’écart`}
					<div
						style={{
							position: 'absolute',
							left: '50%',
							top: '100%',
							marginTop: 0.8 * UNITE,
							transform: 'translateX(-50%)',
							fontFamily: MONO,
							fontSize: F(30),
							lineHeight: 1,
							whiteSpace: 'pre',
							color: C['text-primary'],
							opacity: eInvite,
						}}
					>
						<span style={{color: C.accent}}>{'> '}</span>
						vérifions
						<span style={{color: C.accent, opacity: opaciteCurseur(t)}}>{' █'}</span>
					</div>
				</div>
			</div>
		</Fondu>
	);
};
