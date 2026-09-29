// Le constat : les sondages se trompent ensemble, sans mimétisme.
import React from 'react';
import {C, SERIF} from '../../lib/charte';
import {Titre} from '../../lib/composants';
import {F, NBSP, UNITE as U, Y, avance, fr} from '../../lib/outils';
import {Calque} from './commun';
import {D, MIM} from './donnees';
import {CONSTAT} from './temps';

const Apparition: React.FC<{t: number; debut: number; duree: number; monte?: boolean; children: React.ReactNode}> = ({t, debut, duree, monte = false, children}) => {
	const u = avance(t, debut, duree);
	return <div style={{opacity: u, transform: monte ? `translateY(${0.15 * U * (1 - u)}px)` : undefined}}>{children}</div>;
};

export const Constat: React.FC<{t: number}> = ({t}) => (
	<Calque>
		<div
			style={{
				position: 'absolute',
				left: '50%',
				top: Y(-0.5),
				transform: 'translate(-50%, -50%)',
				display: 'flex',
				flexDirection: 'column',
				alignItems: 'center',
				gap: 0.8 * U,
				whiteSpace: 'pre',
				lineHeight: 1,
				fontFamily: SERIF,
				color: C['text-primary'],
			}}
		>
			<div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.3 * U}}>
				<Apparition t={t} debut={CONSTAT.constat} duree={1} monte>
					<div style={{fontSize: F(64)}}>
						<Titre avant="Les sondages se trompent " mot="ensemble" />
					</div>
				</Apparition>
				<Apparition t={t} debut={CONSTAT.detail} duree={0.6}>
					<div style={{fontSize: F(28)}}>
						{`consensus d’erreur anormal dans ${fr(100 * MIM.part_consensus, 0)}${NBSP}% des élections, contre ${fr(100 * D.seuil, 0)}${NBSP}% au hasard`}
					</div>
				</Apparition>
				<Apparition t={t} debut={CONSTAT.moyenne} duree={0.6}>
					<div style={{fontSize: F(28), color: C['text-secondary']}}>faire la moyenne des sondages ne corrige pas leur erreur commune</div>
				</Apparition>
			</div>
			{/* Manim espace de 0,3 u l'encre des glyphes ; la boîte CSS d'un grand corps est plus haute que son encre, d'où un écart réduit. */}
			<div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.12 * U}}>
				<Apparition t={t} debut={CONSTAT.sans} duree={1} monte>
					<div style={{fontSize: F(52)}}>
						<Titre mot="Pas" apres=" de mimétisme" />
					</div>
				</Apparition>
				<Apparition t={t} debut={CONSTAT.detailSans} duree={0.6}>
					<div style={{fontSize: F(28)}}>
						{`sondages trop semblables dans ${fr(100 * MIM.part_resserrement, 0)}${NBSP}% des élections, pas plus qu’au hasard (${fr(100 * D.seuil, 0)}${NBSP}%)`}
					</div>
				</Apparition>
			</div>
		</div>
	</Calque>
);
