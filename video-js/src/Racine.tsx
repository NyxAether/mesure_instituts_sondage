import React from 'react';
import {AbsoluteFill, Composition, Series} from 'remotion';
import {C, chargerPolices} from './lib/charte';
import {FPS} from './lib/outils';
import {SEQUENCES} from './sequences';

chargerPolices();

const Video: React.FC = () => (
	<AbsoluteFill style={{background: C['bg-primary']}}>
		<Series>
			{SEQUENCES.map(({id, composant: Composant, images}) => (
				<Series.Sequence key={id} durationInFrames={images}>
					<Composant />
				</Series.Sequence>
			))}
		</Series>
	</AbsoluteFill>
);

export const Racine: React.FC = () => (
	<>
		{SEQUENCES.map(({id, composant, images}) => (
			<Composition key={id} id={id} component={composant} durationInFrames={images} fps={FPS} width={1920} height={1080} />
		))}
		<Composition
			id="Video"
			component={Video}
			durationInFrames={SEQUENCES.reduce((somme, s) => somme + s.images, 0)}
			fps={FPS}
			width={1920}
			height={1080}
		/>
	</>
);
