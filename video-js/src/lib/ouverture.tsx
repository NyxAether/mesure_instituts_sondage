// Ouverture d'une séquence selon la charte (guide/video.md) : l'écran de chapitre « sommaire », dont le titre tapé
// et la ligne choisie glissent ensuite dans le coin pour devenir l'en-tête du premier plan. Mesures et durées :
// tokens `video`, transposés du gabarit templates/video/chapitre.html de la charte.
import React from 'react';
import {AbsoluteFill, Sequence, useCurrentFrame} from 'remotion';
import {CHAPITRES, type TitreUnMot} from './chapitres';
import {C, MONO, RAYON_FIN, SERIF, V} from './charte';
import {Entete, INTERLIGNE_TITRE} from './composants';
import {FPS, avance, lerp} from './outils';

const K = V.chapter;
const TK = K.timing;
const H = V.header;
const E = H.enter;

/** Les séquences ouvraient sur 1 s d'en-tête seul : leur contenu commence à cette seconde. */
const ENTETE_INTERNE = 1;

const texte = (titre: TitreUnMot) => `${titre.avant ?? ''}${titre.mot}${titre.apres ?? ''}`;

/** Approche des lettres et interligne du titre tapé (gabarit templates/video/chapitre.html). */
const APPROCHE = -0.02;
const INTERLIGNE = 1.05;

/** Largeur (px) d'un titre à un corps donné, mesurée avec les polices chargées. */
const largeur = (titre: TitreUnMot, corps: number) => {
	const ctx = document.createElement('canvas').getContext('2d')!;
	const mesure = (bout: string, italique: boolean) => {
		ctx.font = `${italique ? 'italic ' : ''}400 ${corps}px ${SERIF}`;
		return ctx.measureText(bout).width + [...bout].length * APPROCHE * corps;
	};
	return mesure(titre.avant ?? '', false) + mesure(titre.mot, true) + mesure(titre.apres ?? '', false);
};

/** Corps du titre tapé : celui de la charte, réduit juste assez pour que titre et curseur bloc tiennent avant la marge droite. */
const corpsTitre = (titre: TitreUnMot) => {
	const corps = K.title.size;
	const dispo = V.size[0] - V.margin.x - K.title.x - (K.title.cursor.gap + K.title.cursor.width) * corps;
	return Math.min(corps, (corps * dispo) / largeur(titre, corps));
};

/** Instants (s) de l'écran du chapitre n (1 à 6). */
const minutage = (n: number) => {
	const selection = TK.cursor + (n - 1) * TK.step + TK['select-after'];
	const ouverture = selection + TK['open-after'];
	const frappe = ouverture + TK.open + TK['type-after'];
	const tape = frappe + [...texte(CHAPITRES[n - 1].titre)].length * TK.char;
	const sortie = tape + TK.hold;
	return {selection, ouverture, frappe, tape, sortie, fin: sortie + E.move};
};

/** Durée de l'écran de chapitre, glissement du titre compris, en images. */
export const imagesChapitre = (n: number) => Math.round(minutage(n).fin * FPS);

/** Le titre tapé : les `nb` premiers caractères, le mot en italique prune compris. */
const Tape: React.FC<{titre: TitreUnMot; nb: number}> = ({titre, nb}) => {
	const morceaux: [string, boolean][] = [[titre.avant ?? '', false], [titre.mot, true], [titre.apres ?? '', false]];
	let reste = nb;
	return (
		<>
			{morceaux.map(([bout, italique], k) => {
				const vu = [...bout].slice(0, Math.max(0, reste)).join('');
				reste -= [...bout].length;
				return italique ? <span key={k} style={{fontStyle: 'italic', color: C.accent}}>{vu}</span> : <span key={k}>{vu}</span>;
			})}
		</>
	);
};

/** Écran de chapitre : compteur, sommaire, curseur, sélection et titre tapé, puis le glissement vers l'en-tête. Fond transparent : le contenu entre dessous. */
export const EcranChapitre: React.FC<{numero: number}> = ({numero}) => {
	const t = useCurrentFrame() / FPS;
	const m = minutage(numero);
	const choisi = numero - 1;
	const haut = (i: number) => K.list.top + i * K.list.pitch;

	// 1. La liste apparaît ligne après ligne, puis s'ouvre sous le chapitre choisi.
	const ouvre = avance(t, m.ouverture, TK.open);
	// 2. Le curseur descend d'une ligne à chaque pas ; chaque déplacement finit à l'instant du pas.
	let pos = 0;
	for (let k = 1; k < numero; k++) pos += avance(t, TK.cursor + k * TK.step - TK.move, TK.move);
	const vu = avance(t, TK.cursor - TK['cursor-fade'], TK['cursor-fade']);
	// 3. Le compteur suit le curseur : l'ancien chiffre sort vers le haut, le nouveau entre par le bas.
	const affiche = Math.min(choisi, Math.round(pos));
	const frac = pos - Math.floor(pos);
	const glisse = frac > 0 && frac < 1 ? (frac < 0.5 ? -frac : 1 - frac) : 0;
	// 4. Sélection : la bande se pose derrière la ligne.
	const bande = avance(t, m.selection, TK['band-fade']);
	// 5. Le titre s'écrit derrière le curseur bloc, qui clignote une fois le titre fini.
	const titre = CHAPITRES[choisi].titre;
	const nb = Math.max(0, Math.floor((t - m.frappe) / TK.char));
	const fini = nb >= [...texte(titre)].length;
	const allume = t >= m.frappe - 0.05 && (!fini || Math.floor(((t - m.tape) / TK.blink) * 2) % 2 === 0);
	// 7. Sortie : le décor s'efface pendant que le titre et la ligne choisie glissent vers l'en-tête.
	const decor = 1 - avance(t, m.sortie, TK.exit);
	const g = avance(t, m.sortie, E.move);
	const sorti = t >= m.sortie;
	const corps = corpsTitre(titre);
	const hautTitre = haut(choisi + 1) + K.title.offset;
	// Titre rangé : l'interligne passe de 1,05 à 1,02, on compense la demi-différence pour que le texte tombe à sa place.
	const hautRange = V.margin.top + H.tag + H.gap - ((INTERLIGNE - INTERLIGNE_TITRE) / 2) * H.title;
	const numero2 = String(numero).padStart(2, '0');

	return (
		<AbsoluteFill>
			{/* Compteur « 0N/ », centré verticalement */}
			<div style={{position: 'absolute', left: K.counter.x, top: '50%', transform: 'translateY(-50%)', fontFamily: SERIF, fontSize: K.counter.size, lineHeight: 0.8, letterSpacing: '-.04em', color: C['text-primary'], whiteSpace: 'pre', opacity: vu * decor}}>
				<span style={{display: 'inline-block', transform: `translateY(${glisse * K.counter.travel * 100}%)`, opacity: 1 - Math.abs(glisse) * 1.8}}>
					{String(affiche + 1).padStart(2, '0')}
				</span>
				<span style={{color: C.accent}}>/</span>
			</div>

			{/* Bande de sélection, centrée sur la boîte de la ligne choisie */}
			<div
				style={{
					position: 'absolute',
					left: K.band.x,
					top: haut(choisi) + (K.list.size - K.band.height) / 2,
					width: K.band.width,
					height: K.band.height,
					borderRadius: RAYON_FIN,
					background: C['bg-secondary'],
					opacity: bande * decor,
					transform: `scaleX(${lerp(K.band['scale-from'], 1, bande)})`,
					transformOrigin: '0 50%',
				}}
			/>

			{/* Sommaire */}
			{CHAPITRES.map((c, i) => {
				const a = avance(t, i * TK['row-stagger'], TK['row-fade']);
				const opacite = i === choisi ? (sorti ? 0 : a) : a * decor;
				const couleur = i === choisi && t >= m.selection ? C['text-primary'] : i < choisi ? C['text-secondary'] : C['text-muted'];
				return (
					<div key={i} style={{position: 'absolute', left: K.list.x, top: haut(i) + (i > choisi ? ouvre * K.title.opening : 0) + (1 - a) * K.list.rise, opacity: opacite, fontFamily: MONO, fontSize: K.list.size, lineHeight: 1, color: couleur, whiteSpace: 'pre'}}>
						{`${i === CHAPITRES.length - 1 ? '└─' : '├─'} ${String(i + 1).padStart(2, '0')} ${c.nom}`}
					</div>
				);
			})}

			{/* Curseur de sélection, dans sa propre colonne */}
			<div style={{position: 'absolute', left: K['cursor-x'], top: haut(pos), fontFamily: MONO, fontSize: K.list.size, lineHeight: 1, color: C.accent, whiteSpace: 'pre', opacity: vu * decor}}>
				{'>'}
			</div>

			{/* 7. La ligne choisie glisse avec le titre et devient le tag : ├─ laisse place à //, le point apparaît, le nom se décale.
			    Chasse fixe : chaque morceau est placé en ch, le numéro ne bouge pas dans la ligne. */}
			{sorti && (
				<div
					style={{
						position: 'absolute',
						left: lerp(K.list.x, V.margin.x, g),
						top: lerp(haut(choisi), V.margin.top, g),
						height: '1em',
						fontFamily: MONO,
						fontSize: K.list.size,
						lineHeight: 1,
						whiteSpace: 'pre',
						color: `color-mix(in oklab, ${C['text-muted']} ${g * 100}%, ${C['text-primary']})`,
						transform: `scale(${lerp(1, H.tag / K.list.size, g)})`,
						transformOrigin: '0 0',
					}}
				>
					<span style={{position: 'absolute', top: 0, left: 0, opacity: 1 - g}}>{choisi === CHAPITRES.length - 1 ? '└─' : '├─'}</span>
					<span style={{position: 'absolute', top: 0, left: 0, opacity: g}}>//</span>
					<span style={{position: 'absolute', top: 0, left: '3ch'}}>{numero2}</span>
					<span style={{position: 'absolute', top: 0, left: `${3 + numero2.length + 1}ch`, opacity: g}}>·</span>
					<span style={{position: 'absolute', top: 0, left: `${3 + numero2.length + 1 + 2 * g}ch`}}>{CHAPITRES[choisi].nom}</span>
				</div>
			)}

			{/* Titre tapé, sous la ligne choisie, suivi du curseur bloc ; 7. il glisse dans le coin et prend le corps de l'en-tête */}
			<div
				style={{
					position: 'absolute',
					left: lerp(K.title.x, V.margin.x, g),
					top: lerp(hautTitre, hautRange, g),
					fontFamily: SERIF,
					fontSize: corps,
					lineHeight: INTERLIGNE,
					letterSpacing: `${APPROCHE}em`,
					color: C['text-primary'],
					whiteSpace: 'pre',
					opacity: ouvre,
					transform: `scale(${lerp(1, H.title / corps, g)})`,
					transformOrigin: '0 0',
				}}
			>
				<Tape titre={titre} nb={nb} />
				{/* Curseur bloc posé sur la ligne de base sans hauteur propre : il ne change pas la boîte de la ligne,
				    sinon le titre ne tomberait pas exactement sur l'en-tête rangé à la fin du glissement. */}
				<span style={{display: 'inline-block', position: 'relative', width: `${K.title.cursor.width}em`, height: 0, marginLeft: `${K.title.cursor.gap}em`}}>
					<span
						style={{
							position: 'absolute',
							left: 0,
							bottom: '-.12em',
							width: '100%',
							height: `${K.title.cursor.height}em`,
							background: C.accent,
							opacity: allume ? decor : 0,
						}}
					/>
				</span>
			</div>
		</AbsoluteFill>
	);
};

/** Séquence de contenu précédée de son écran de chapitre : le titre du chapitre y glisse en en-tête du premier plan,
 * et le contenu entre en fondu `content-delay` après le début du glissement. */
export const avecOuverture = (numero: number, Contenu: React.FC, images: number) => {
	const m = minutage(numero);
	const chapitre = imagesChapitre(numero);
	const debutContenu = m.sortie + E['content-delay'];
	// Le contenu commence là où finissait son ancien en-tête seul (ENTETE_INTERNE) : à l'instant voulu par la charte.
	const depart = Math.round((debutContenu - ENTETE_INTERNE) * FPS);
	const c = CHAPITRES[numero - 1];
	const Ouverte: React.FC = () => {
		const t = useCurrentFrame() / FPS;
		return (
			<AbsoluteFill style={{background: C['bg-primary']}}>
				{/* Le contenu est monté sous l'écran de chapitre : sa première seconde, vide, passe dessous. */}
				<Sequence from={depart}>
					<AbsoluteFill style={{opacity: avance(t, debutContenu, E['content-fade'])}}>
						<Contenu />
					</AbsoluteFill>
				</Sequence>
				{/* Une fois le glissement fini, l'en-tête rangé prend le relais, au même endroit. */}
				<Sequence from={chapitre}>
					<Entete numero={numero} nom={c.nom} {...c.titre} />
				</Sequence>
				<Sequence durationInFrames={chapitre}>
					<EcranChapitre numero={numero} />
				</Sequence>
			</AbsoluteFill>
		);
	};
	return {composant: Ouverte, images: depart + images};
};
