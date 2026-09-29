// Copie dans public/ (ignoré par git) les ressources prêtes du dépôt, et tire la couleur du papier des coupures
// de video/outils/decoupe_portraits.py : rien n'est dupliqué à la main.
// Lancement : node ressources.mjs (fait par npm run render et npm run studio).
import {copyFileSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const ICI = dirname(fileURLToPath(import.meta.url));
const DEPOT = join(ICI, '..');
const EXTERNE = join(DEPOT, 'video', 'externe');

const copies = {
	'meme/quelle_indignite.mp4': join(EXTERNE, 'meme', 'quelle_indignite.mp4'),
	'fonts/Newsreader-400.ttf': join(DEPOT, 'rr-fonts', 'Newsreader-400.ttf'),
	'fonts/Newsreader-400-Italic.ttf': join(DEPOT, 'rr-fonts', 'Newsreader-400-Italic.ttf'),
	'fonts/JetBrainsMono-400.ttf': join(DEPOT, 'rr-fonts', 'JetBrainsMono-400.ttf'),
};
for (const nom of ['fillon', 'juppe', 'sarkozy', 'macron', 'lepen']) {
	copies[`portraits/${nom}.png`] = join(EXTERNE, 'portraits', `${nom}_decoupe.png`);
}
for (const nom of ['fillon', 'juppe', 'sarkozy']) {
	copies[`corps/${nom}.png`] = join(EXTERNE, 'corps', `${nom}_decoupe.png`);
}
for (const [cible, source] of Object.entries(copies)) {
	mkdirSync(dirname(join(ICI, 'public', cible)), {recursive: true});
	copyFileSync(source, join(ICI, 'public', cible));
}

const outil = readFileSync(join(DEPOT, 'video', 'outils', 'decoupe_portraits.py'), 'utf-8');
const m = outil.match(/^PAPIER = np\.array\(\[(\d+), (\d+), (\d+)\]/m);
if (!m) throw new Error('couleur PAPIER introuvable dans decoupe_portraits.py');
mkdirSync(join(ICI, 'src', 'genere'), {recursive: true});
writeFileSync(join(ICI, 'src', 'genere', 'papier.json'), JSON.stringify({papier: `rgb(${m[1]}, ${m[2]}, ${m[3]})`}) + '\n');
console.log(`${Object.keys(copies).length} fichiers copiés dans public/, papier : rgb(${m.slice(1).join(', ')})`);
