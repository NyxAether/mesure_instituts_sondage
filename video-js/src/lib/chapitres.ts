// Les six chapitres de la vidéo : le nom, au sommaire et dans le tag de l'en-tête, et le titre, tapé à l'écran
// de chapitre puis repris par l'en-tête des plans de contenu.
import {NBSP} from './outils';

/** Titre dont un mot (ou une courte expression) est en italique prune. */
export type TitreUnMot = {avant?: string; mot: string; apres?: string};

export type Chapitre = {nom: string; titre: TitreUnMot};

export const CHAPITRES: Chapitre[] = [
	{nom: 'la théorie', titre: {avant: 'Ce que prévoit la ', mot: 'marge', apres: `${NBSP}d’erreur`}},
	{nom: 'l’entonnoir', titre: {avant: 'Les sondages face aux ', mot: 'résultats'}},
	{nom: 'la taille', titre: {avant: 'L’erreur selon la ', mot: 'taille', apres: ' d’échantillon'}},
	{nom: 'la taille équivalente', titre: {avant: 'Combien vaut ', mot: 'vraiment', apres: `${NBSP}un sondage${NBSP}?`}},
	{nom: 'l’erreur partagée', titre: {avant: 'Une erreur ', mot: 'partagée'}},
	{nom: 'le présage', titre: {avant: 'Un ', mot: 'présage', apres: ' plus qu’une prédiction'}},
];
