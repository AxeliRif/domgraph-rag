# Rapport de projet de master — DOM-Graph RAG

Sources LaTeX du rapport de PDM (EPFL, Section d'Informatique).

## Compilation

```bash
latexmk -pdf main.tex
```

Ou, sans `latexmk` :

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Paquets requis (au-delà d'une TeX Live standard) : `babel-french`, `lmodern`,
`siunitx`, `algorithm2e`, `cleveref`, `titlesec`, `fancyhdr`, `tabularx`,
`subcaption`, `listings`, `emptypage`, `tikz`.

Sur Debian/Ubuntu :

```bash
apt install texlive-latex-extra texlive-lang-french texlive-science \
            texlive-pictures texlive-fonts-recommended lmodern
```

## À compléter avant soumission

Tous les champs à remplir sont en **rouge** dans le PDF et marqués
`\textcolor{red}{[...]}` dans les sources, dans `chapitres/00_page_titre.tex` :

| Champ | Exigé par les directives IC |
|---|---|
| Adresse postale | oui (coordonnées de l'étudiant) |
| Nom du superviseur académique EPFL | oui |
| Nom du laboratoire d'accueil | oui |
| Nom du co-encadrant | si applicable |
| Date de soumission, lieu, signature | oui |
| Remerciements | non (à personnaliser) |

Pour les localiser rapidement :

```bash
grep -rn "textcolor{red}" chapitres/
```

## Structure

```
main.tex                      préambule, ordre des chapitres
references.bib                51 références
chapitres/
  00_page_titre.tex           page de titre, déclaration, remerciements
  00_resume.tex               résumé (fr) et abstract (en)
  01_introduction.tex         introduction
  02_etat_de_lart.tex         chapitre 1 — état de l'art et concepts
  03_methodologie.tex         chapitre 2 — méthodologie
  04_resultats.tex            chapitre 3 — résultats, analyse et limites
  05_conclusion.tex           conclusion
  06_annexes.tex              annexes A à E
figures/                      figures (PDF vectoriels et PNG)
make_figures.py               regénère les figures de résultats depuis data/
```

## Regénérer les figures de résultats

`make_figures.py` lit les fichiers de résultats du dépôt principal
(`data/dom_vs_grid_results.json`, `data/qa_dataset/multihop_*.json`) et
produit les six figures `figures/res_*.pdf`. Ajuster la constante `DATA` en
tête du script pour pointer sur le dépôt, puis :

```bash
python3 make_figures.py
```

## Volumétrie

- Corps principal : 66 pages (introduction 4, ch. 1 : 18, ch. 2 : 20,
  ch. 3 : 20, conclusion 4)
- Annexes : 10 pages · Bibliographie : 5 pages
- Total : 101 pages

## Note sur les chiffres

Tous les chiffres du chapitre 3 ont été vérifiés programmatiquement contre les
fichiers de résultats du dépôt (`data/dom_vs_grid_results.json`,
`data/qa_dataset/multihop_controller_eval.json`,
`data/qa_dataset/multihop_e2e_eval.json`). Les valeurs de l'évaluation B
(lecteur base / LoRA) proviennent de `paper/sections/experiment.tex`, aucun
fichier de résultats n'ayant été trouvé pour cette expérience dans `data/`.
