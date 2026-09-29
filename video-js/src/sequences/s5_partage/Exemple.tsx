// L'exemple du Royaume-Uni 2015 : d'abord des sondages simulés (les erreurs se compensent), puis les vrais (tous du même côté).
import React from 'react';
import {C, G, SERIF} from '../../lib/charte';
import {NBSP, Txt, UNITE as U, X, Y, avance, echelonne, fr, lerp, signe} from '../../lib/outils';
import {Boite, Calque, Ligne} from './commun';
import {D, J_MAX, PARTIS, Parti, REELS, RESULTAT, SIMULES, SONDAGES} from './donnees';
import {HASARD, REEL, TETE} from './temps';

const GAUCHE = -4.8;
const DROITE = 3.2;
const BAS = -2.9;
const HAUT = 1.2;
const Y_MIN = 27;
const Y_MAX = 41;
const GRADUATIONS = [30, 35, 40];
const JOURS_LIBELLES = [J_MAX, 10, 5, 1];
const COULEURS: Record<Parti, string> = {conservateurs: G.series[0], travaillistes: G.series[1]};
const DECALAGE: Record<Parti, number> = {conservateurs: -0.07, travaillistes: 0.07};

/** Point de l'écran (unités) pour un nombre de jours avant l'élection et un score en %. */
const pt = (jours: number, score: number): [number, number] => [
	GAUCHE + ((J_MAX - jours) / (J_MAX - 1)) * (DROITE - GAUCHE),
	BAS + ((score - Y_MIN) / (Y_MAX - Y_MIN)) * (HAUT - BAS),
];
const marge = (v: number, n: number) => 100 * D.z95 * Math.sqrt(((v / 100) * (1 - v / 100)) / n);

const moyenne = (valeurs: number[]) => valeurs.reduce((a, b) => a + b, 0) / valeurs.length;
const ligneX = (a: [number, number], b: [number, number]) => ({x1: X(a[0]), y1: Y(a[1]), x2: X(b[0]), y2: Y(b[1])});

/** Moyennes des sondages par parti : un trait épais, et avec `ecarts` l'écart au résultat et son texte. */
const Moyennes: React.FC<{valeurs: Record<Parti, number[]>; ecarts?: boolean; opacite: number}> = ({valeurs, ecarts = false, opacite}) => (
	<Calque opacite={opacite}>
		<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
			{PARTIS.map((parti) => {
				const m = moyenne(valeurs[parti]);
				const v = RESULTAT[parti];
				const dx = 0.2;
				return (
					<g key={parti} stroke={COULEURS[parti]} fill="none">
						<line {...ligneX(pt(J_MAX + 0.5, m), pt(0.5, m))} strokeWidth={5} />
						{ecarts && (
							<>
								<line {...ligneX([pt(0.5, m)[0] + dx, pt(0.5, m)[1]], [pt(0.5, v)[0] + dx, pt(0.5, v)[1]])} strokeWidth={2.5} />
								<line {...ligneX(pt(0.5, m), [pt(0.5, m)[0] + dx, pt(0.5, m)[1]])} strokeWidth={2.5} />
							</>
						)}
					</g>
				);
			})}
		</svg>
		{ecarts &&
			PARTIS.map((parti) => {
				const m = moyenne(valeurs[parti]);
				const v = RESULTAT[parti];
				const milieu = (pt(0.5, m)[1] + pt(0.5, v)[1]) / 2;
				return (
					<Boite key={parti} x={X(pt(0.5, m)[0] + 0.2 + 0.15)} y={Y(milieu)} ay={0.5} ecart={0.05 * U}>
						<Ligne taille={14} couleur={COULEURS[parti]}>{`moyenne ${fr(m)}${NBSP}%`}</Ligne>
						<Ligne taille={24} police={SERIF} couleur={COULEURS[parti]}>{`${signe(m - v)} points`}</Ligne>
					</Boite>
				);
			})}
	</Calque>
);

export const Exemple: React.FC<{t: number}> = ({t}) => {
	const sortie = 1 - avance(t, REEL.sortie, 0.8);
	const bascule = avance(t, REEL.bascule, 0.6);
	const u = avance(t, REEL.points, 2.5);
	const n = SONDAGES.length;

	const axes = avance(t, HASARD.axes, 0.8);
	const resultats = avance(t, HASARD.resultats, 1);
	const legende = avance(t, HASARD.points, 2);

	return (
		<Calque opacite={sortie}>
			<Calque opacite={avance(t, TETE.source, 0.6)}>
				<Txt x={0.55 * U} y={Y(2.07)} ay={0.5} taille={14} couleur={C['text-muted']}>
					{`royaume-uni, législatives ${D.exemple.annee} · ${n} derniers jours avant le vote, une moyenne des sondages par jour`}
				</Txt>
			</Calque>

			<Calque opacite={axes}>
				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					{GRADUATIONS.map((v) => (
						<line key={v} {...ligneX(pt(J_MAX + 0.5, v), pt(0.5, v))} stroke={G.grid} strokeWidth={1} />
					))}
					<line {...ligneX(pt(J_MAX + 0.5, Y_MIN), pt(0.5, Y_MIN))} stroke={G.axis} strokeWidth={1.5} />
				</svg>
				{GRADUATIONS.map((v) => (
					<Txt key={v} x={X(pt(J_MAX + 0.5, v)[0] - 0.15)} y={Y(pt(0, v)[1])} ax={1} ay={0.5} taille={14} couleur={C['text-secondary']}>{`${v}${NBSP}%`}</Txt>
				))}
				{JOURS_LIBELLES.map((j) => (
					<Txt key={j} x={X(pt(j, Y_MIN)[0])} y={Y(BAS) + 0.15 * U} ax={0.5} taille={14} couleur={C['text-secondary']}>{fr(j, 0)}</Txt>
				))}
				<Txt x={X((pt(J_MAX, 0)[0] + pt(1, 0)[0]) / 2)} y={Y(BAS) + 0.15 * U + 0.14 * U + 0.12 * U} ax={0.5} taille={14} couleur={C['text-muted']}>
					jours avant l’élection
				</Txt>
			</Calque>

			<Calque opacite={resultats}>
				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					{PARTIS.map((parti) => (
						<line key={parti} {...ligneX(pt(J_MAX + 0.5, RESULTAT[parti]), pt(0.5, RESULTAT[parti]))} stroke={COULEURS[parti]} strokeWidth={2.5} strokeDasharray="13.5 13.5" />
					))}
				</svg>
				{PARTIS.map((parti) => {
					const v = RESULTAT[parti];
					const haut = parti === 'conservateurs';
					return (
						<Boite key={parti} x={X(pt(0.5, v)[0] + 0.2)} y={haut ? Y(pt(0.5, v)[1]) - 0.05 * U : Y(pt(0.5, v)[1]) + 0.05 * U} ay={haut ? 1 : 0} ecart={0.06 * U}>
							<Ligne taille={15} couleur={COULEURS[parti]}>{parti}</Ligne>
							<Ligne taille={14} couleur={COULEURS[parti]}>{`résultat ${fr(v)}${NBSP}%`}</Ligne>
						</Boite>
					);
				})}
			</Calque>

			<Txt x={X(pt(J_MAX + 0.5, Y_MAX)[0])} y={Y(pt(0, Y_MAX)[1]) - 0.05 * U} ay={1} taille={15} couleur={C['text-primary']} opacite={avance(t, HASARD.mode, 0.5) * (1 - bascule)}>
				si seul le hasard jouait · tirages simulés de même taille
			</Txt>
			<Txt x={X(pt(J_MAX + 0.5, Y_MAX)[0])} y={Y(pt(0, Y_MAX)[1]) - 0.05 * U} ay={1} taille={15} couleur={C['text-primary']} opacite={bascule}>
				les vrais sondages
			</Txt>

			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				{PARTIS.map((parti, ip) =>
					SONDAGES.map((s, j) => {
						const o = echelonne(t, HASARD.points, 2, PARTIS.length * n, ip * n + j, 0.05);
						const sim = SIMULES[parti][j];
						const reel = REELS[parti][j];
						const v = lerp(sim, reel, u);
						// Le trait se déforme d'une forme à l'autre : ses deux bouts vont en ligne droite.
						const bas = lerp(sim - marge(sim, s.taille), reel - marge(reel, s.taille), u);
						const haut = lerp(sim + marge(sim, s.taille), reel + marge(reel, s.taille), u);
						const dx = DECALAGE[parti];
						const [x, y] = pt(s.jours_avant, v);
						return (
							<g key={`${parti}${j}`} opacity={o}>
								<line x1={X(x + dx)} x2={X(x + dx)} y1={Y(pt(s.jours_avant, bas)[1])} y2={Y(pt(s.jours_avant, haut)[1])} stroke={COULEURS[parti]} strokeWidth={2} opacity={0.45} />
								<circle cx={X(x + dx)} cy={Y(y)} r={0.055 * U} fill={COULEURS[parti]} />
							</g>
						);
					}),
				)}
			</svg>
			<Txt x={X(pt(0.5, 0)[0])} y={Y(BAS) - 0.12 * U} ax={1} ay={1} taille={13} couleur={C['text-muted']} opacite={legende}>
				trait vertical : marge d’erreur de chaque sondage
			</Txt>

			<Moyennes valeurs={SIMULES} opacite={avance(t, HASARD.moyennes, 0.8) * (1 - bascule)} />
			<Txt x={X(pt(J_MAX + 0.5, 0)[0])} y={Y(BAS) + 0.75 * U} taille={26} police={SERIF} couleur={C['text-primary']} opacite={avance(t, HASARD.compense, 0.6) * (1 - bascule)}>
				certains au-dessus, d’autres en dessous : les erreurs se compensent
			</Txt>

			<Moyennes valeurs={REELS} ecarts opacite={avance(t, REEL.moyennes, 0.8)} />
			<Txt x={X(pt(J_MAX + 0.5, 0)[0])} y={Y(BAS) + 0.75 * U} taille={26} police={SERIF} couleur={C['text-primary']} opacite={avance(t, REEL.memeCote, 0.6)}>
				tous du même côté : la moyenne garde l’erreur
			</Txt>
		</Calque>
	);
};
