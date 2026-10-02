// Liste ordonnée des séquences de la vidéo : chacune est une composition à part (relecture isolée)
// et un morceau de la composition « Video » qui les enchaîne. Les séquences 1 à 6 s'ouvrent sur leur écran de chapitre.
import type React from 'react';
import {avecOuverture} from './lib/ouverture';
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
	{id: 'Sequence1', ...avecOuverture(1, Theorie, dureeTheorie)},
	{id: 'Sequence2', ...avecOuverture(2, Entonnoir, dureeEntonnoir)},
	{id: 'Sequence3', ...avecOuverture(3, Taille, dureeTaille)},
	{id: 'Sequence4', ...avecOuverture(4, Equivalente, dureeEquivalente)},
	{id: 'Sequence5', ...avecOuverture(5, Partage, dureePartage)},
	{id: 'Sequence6', ...avecOuverture(6, Presage, dureePresage)},
];
