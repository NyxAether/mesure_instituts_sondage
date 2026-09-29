// Liste ordonnée des séquences de la vidéo : chacune est une composition à part (relecture isolée)
// et un morceau de la composition « Video » qui les enchaîne.
import type React from 'react';
import {Accroche, dureeAccroche} from './sequences/s0_accroche';
import {Theorie, dureeTheorie} from './sequences/s1_theorie';
import {Entonnoir, dureeEntonnoir} from './sequences/s2_entonnoir';
import {Taille, dureeTaille} from './sequences/s3_taille';
import {Equivalente, dureeEquivalente} from './sequences/s4_equivalente';
import {Partage, dureePartage} from './sequences/s5_partage';
import {Presage, dureePresage} from './sequences/s6_presage';

export type Sequence = {id: string; composant: React.FC; images: number};

export const SEQUENCES: Sequence[] = [
	{id: 'Sequence0', composant: Accroche, images: dureeAccroche},
	{id: 'Sequence1', composant: Theorie, images: dureeTheorie},
	{id: 'Sequence2', composant: Entonnoir, images: dureeEntonnoir},
	{id: 'Sequence3', composant: Taille, images: dureeTaille},
	{id: 'Sequence4', composant: Equivalente, images: dureeEquivalente},
	{id: 'Sequence5', composant: Partage, images: dureePartage},
	{id: 'Sequence6', composant: Presage, images: dureePresage},
];
