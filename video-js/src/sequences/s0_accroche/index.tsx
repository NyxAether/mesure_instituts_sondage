import React from 'react';
import {AbsoluteFill} from 'remotion';
import {C} from '../../lib/charte';
import {FPS} from '../../lib/outils';
import {Fin} from './Fin';
import {Primaire} from './Primaire';
import {T} from './temps';

export const Accroche: React.FC = () => (
	<AbsoluteFill style={{background: C['bg-primary'], overflow: 'hidden'}}>
		<Primaire />
		<Fin />
	</AbsoluteFill>
);

export const dureeAccroche = Math.round(T.fin * FPS);
