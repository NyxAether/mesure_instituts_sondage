# Récupération, analyses et visualisations des rapports d'instituts de sondages

Le but de ce projet est de collecter les données publiées par les instituts de sondages en les extrayant des PDF.

Pour le moment, trois instituts de sondage sont analysés :
* IPSOS
* IFOP
* ELABE

Chacun des instituts de sondages possèdent sont propre dossier contenant :
* Un jupyter d'extraction
* Un jupyter d'analyse
* Un dossier des données pour chaque rapport

Certains instituts peuvent contenir plusieurs dossiers de données en fonctions des études récupérées.

## Visualisations HTML

Les analyses sont progressivement migrées des notebooks vers des pages HTML statiques :

* `analyses/` : modules Python qui calculent les données de chaque page et les exportent dans `docs/data/<page>.js` ;
* `docs/` : pages HTML (Observable Plot, thème clair/sombre, vue tableau pour chaque graphique), ouvrables directement dans un navigateur ou publiables via GitHub Pages.

Pages disponibles :

* `docs/explications.html` : marges d'erreur, loi normale et seuil de significativité des variations.

Régénérer les données d'une page :

```sh
.venv/Scripts/python.exe -m analyses.explications
```

Les PDFs étant la propriété intellectuelle des instituts, ceux-ci ne sont pas disponible sur ce dépot.