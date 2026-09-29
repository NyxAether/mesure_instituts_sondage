// Grille horizontale à bornes animées : les lignes fines apparaissent quand elles s'espacent, celles qui sortent du cadre s'effacent.
import React from 'react';
import {C, G} from '../../lib/charte';
import {Txt, X, Y, fr} from '../../lib/outils';
import {DROITE, GAUCHE, HAUTEUR, TRAIT, Y_ECART, yu} from './repere';

const borne = (v: number) => Math.min(1, Math.max(0, v));
const signe = (v: number) => (v > 0 ? `+${fr(v, 0)}` : v === 0 ? '0' : `−${fr(-v, 0)}`);
const pas = (y: number) => (y % 5 === 0 ? 5 : y % 1 === 0 ? 1 : 0.5);

const VALEURS = Array.from({length: Math.round(4 * Y_ECART) + 1}, (_, k) => -Y_ECART + k * 0.5);

/** `bas` et `haut` sont les bornes de l'axe vertical à cet instant, `visible` l'opacité générale de la grille. */
export const Grille: React.FC<{bas: number; haut: number; visible: number}> = ({bas, haut, visible}) => {
	const etendue = haut - bas;
	const signesVisibles = borne(-bas / 3); // graduations « +5 / −5 » tant que l'axe descend sous zéro
	const lignes = VALEURS.map((y) => {
		const bord = borne(1 - Math.max(y - haut, bas - y) / (0.05 * etendue));
		const espace = (pas(y) * HAUTEUR) / etendue;
		return {y, bord, espace, opacite: visible * bord * (y === 0 ? 1 : borne((espace - 0.3) / 0.3))};
	});
	return (
		<>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				{lignes.map(({y, opacite}) => (
					<line
						key={y}
						x1={X(GAUCHE)}
						x2={X(DROITE)}
						y1={Y(yu(y, bas, haut))}
						y2={Y(yu(y, bas, haut))}
						stroke={y === 0 ? G.axis : G.grid}
						strokeWidth={(y === 0 ? 1.5 : 1) * TRAIT}
						opacity={opacite}
					/>
				))}
			</svg>
			{lignes
				.filter(({y}) => y % 1 === 0)
				.map(({y, bord, espace}) => {
					const opacite = visible * bord * borne((espace - 0.5) / 0.3);
					return [
						[signe(y), signesVisibles],
						[fr(y, 0), 1 - signesVisibles],
					].map(([texte, poids], i) => (
						<Txt key={`${y}-${i}`} x={X(GAUCHE - 0.15)} y={Y(yu(y, bas, haut))} ax={1} ay={0.5} taille={14} couleur={C['text-secondary']} opacite={opacite * (poids as number)}>
							{texte as string}
						</Txt>
					));
				})}
		</>
	);
};
