// Calque plein écran dont l'opacité et le décalage varient avec l'animation (FadeIn / FadeOut avec `shift` de Manim).
import React from 'react';

/** `dx`, `dy` : décalage courant en px (écran, y vers le bas). Tout enfant se place en absolu dans le repère de l'image. */
export const Fondu: React.FC<{o?: number; dx?: number; dy?: number; children: React.ReactNode}> = ({o = 1, dx = 0, dy = 0, children}) => (
	<div style={{position: 'absolute', left: 0, top: 0, width: 1920, height: 1080, opacity: o, transform: dx || dy ? `translate(${dx}px, ${dy}px)` : undefined}}>{children}</div>
);

/** Calque SVG plein écran (repère en px de l'image). */
export const Calque: React.FC<{children: React.ReactNode}> = ({children}) => (
	<svg width={1920} height={1080} style={{position: 'absolute', left: 0, top: 0, overflow: 'visible'}}>
		{children}
	</svg>
);
