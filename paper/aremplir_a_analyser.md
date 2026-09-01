# Les 26 `\aremplir` du rapport — à analyser un par un

27 occurrences physiques de `\aremplir` dans le document, mais certaines valeurs sont dupliquées à deux endroits (méthodologie + annexe A). Regroupées ci-dessous par **valeur unique à trouver** : 12 groupes, 23 valeurs distinctes au total. Remplissez la ligne `Valeur retenue :` de chaque groupe, la même valeur s'applique à tous les emplacements listés.

Pour chaque groupe : le paragraphe exact tel qu'il apparaît dans le rapport (le `\aremplir[...]` est laissé visible pour repérer l'emplacement), la source probable dans le code (reprise de `modifications_code_ch2.md` quand elle y figure), et une ligne à remplir.

---

## 1. Comptages médians du tuilage (4 valeurs)

**Emplacement :** `03_methodologie.tex:330-339`

> \paragraph{Complexité et comptages.} L'élagage compare chaque candidat aux éléments déjà retenus et le regroupement examine toutes les paires : les deux sont en $O(n^2)$ dans le pire cas, où $n$ est le nombre d'éléments extraits. La page la plus dense du corpus en compte 278 avant élagage, et la chaîne des comptages médians par unité de page s'établit à **\aremplir[éléments extraits]** éléments extraits, **\aremplir[après élagage]** après élagage, **\aremplir[groupes]** groupes, pour **\aremplir[tuiles]** tuiles. Ces deux étapes restent négligeables devant le rendu de page et, à plus forte raison, devant les appels au modèle vision-langage ; une structure d'indexation spatiale serait l'optimisation naturelle sur des documents d'un ordre de grandeur plus denses.

**Source code :** `tiling.py`, fonction principale — journaliser `stats = {"extraits": ..., "apres_elagage": ..., "groupes": ..., "tuiles": ...}` par unité de page, puis calculer la médiane de chaque clé sur le corpus déjà construit (`data/units.jsonl`).

**Vérifier au passage :** que 278 est bien le maximum d'éléments *avant* élagage (déjà affirmé dans le texte, à confirmer contre les données).

Valeur retenue — éléments extraits : 9
Valeur retenue — après élagage : 7
Valeur retenue — groupes : 7
Valeur retenue — tuiles : 6

**Comment obtenu :** pas de `tiling.py` instrumenté ni de `data/units.jsonl` existant dans le dépôt — recalculé en rejouant réellement `dom_extraction.extract_dom_elements_async` → `sectioning.split_into_sections` (si page démesurée) → `tiling.prune_nested_elements`/`group_overlapping_elements`/`build_tiles`, sur les 18 pages Wikipédia de départ de `scripts/build_dataset.py::DEFAULT_URLS`, chargées à leurs révisions épinglées (`data/wikipedia_snapshot.json`, valable pour la reproductibilité puisqu'elles ont été figées aujourd'hui). Cela donne 230 unités de page (chaque section d'une page découpée compte comme une unité, comme dans `corpus_builder.py`) — délibérément *sans* appliquer le plafond de 25 tuiles de `build_dataset.py` (`--max-tiles-per-page`), qui ne s'applique qu'au jeu de données QA en aval, pas aux comptages bruts du tuilage.

**⚠️ Sur le "278" déjà affirmé dans le texte : NON confirmé.** Le maximum réellement mesuré (avant élagage, par unité) est **161** éléments (`The_Beatles`, section 1 — cohérent avec la section 2.2/§incident déjà citée pour cet article), pas 278. Le second plus dense est 151 (`Mount_Everest`, section 0). Deux lectures possibles, à trancher selon ce que "278" est censé désigner : (a) si c'est bien un comptage *par unité de page* comme le reste du paragraphe, la valeur est obsolète (probablement mesurée sur une révision Wikipédia antérieure, avant que ces articles ne soient édités) et devrait être corrigée en 161 ; (b) si "278" désignait en fait le compte d'éléments de la page *entière avant découpage en sections* (une notion différente, non recalculée ici), la phrase doit le préciser explicitement pour ne pas laisser croire qu'il s'agit de la même grandeur que les médianes par unité qui suivent. À vérifier avant publication — c'est un des deux cas de la liste où le résultat de l'analyse peut obliger à corriger une phrase déjà écrite, pas seulement une valeur manquante.

---

## 2. Valeur de `max_pixels` (2 emplacements, même valeur)

**Emplacement A :** `03_methodologie.tex:376-379`

> La valeur effective du plafond employée dans nos mesures est **\aremplir[valeur de max\_pixels]**, reportée en annexe A. Le facteur 2,7 de la section résultats est un rapport de *pixels* ; il ne se transpose au coût en tokens que dans la colonne sans écrêtage.

**Emplacement B :** `06_annexes.tex:87` (table hyperparamètres)

> Aire d'image maximale du processeur (`max_pixels`) & **\aremplir** & \cref{tab-cout_tokens}

**Source code :**
```python
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
print(proc.image_processor.max_pixels)
```
**Décisif pour le facteur 2,7 vs 1,3 du tableau `tab-cout_tokens`** — si la valeur mesurée diffère de l'hypothèse implicite (~1,00 Mpx), corriger aussi ce tableau.

Valeur retenue : 12 845 056 px (confirmé par sonde réelle `AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")` sur le cache HF local, transformers 5.13.1 — exposé via `ip.size["longest_edge"]`, pas un attribut `max_pixels` de premier niveau dans cette version). Très supérieur à l'hypothèse implicite ~1,00 Mpx : même la tuile DOM maximale (2048×2048 = 4 194 304 px) n'est **pas écrêtée** (5329 tokens). **Conséquence sur `tab-cout_tokens` : le facteur 2,7 (pixels) se transpose bien en tokens — aucune tuile de ce pipeline n'atteint jamais le plafond, donc la colonne "avec écrêtage" ne s'applique en pratique jamais ; à corriger/clarifier dans le tableau plutôt que garder les deux colonnes 2,7/1,3 comme si le choix dépendait des données.**

---

## 3. Budget de hauteur d'une section trop haute (2 emplacements, même valeur)

**Emplacement A :** `03_methodologie.tex:390-397`

> Exclure ces pages serait contraire à l'objectif du projet, qui vise les documents longs ; elles sont donc découpées en amont du tuilage. Une page dont la hauteur dépasse dix fois la hauteur maximale de tuile, soit 20480 px, est scindée aux frontières de titres `h1` et `h2`. Le contenu précédant le premier titre forme une section « préambule » sans titre. Une section restant trop haute à elle seule — un `h2` suivi d'un tableau de discographie géant — est retranchée par un budget de hauteur fixe de **\aremplir[budget de hauteur]** : le compromis de la règle de fusion/découpe appliqué un cran au-dessus, entre éléments plutôt qu'à l'intérieur d'un seul.

**Emplacement B :** `06_annexes.tex:34` (table hyperparamètres)

> Budget de hauteur d'une section trop haute & **\aremplir[budget]** & \cref{sec-tuilage_sections}

**Source code :** littéral ou constante dans `sectioning.py` — si littéral, le remonter dans `config.py` (promesse de l'annexe A : « centralisée dans un module de configuration »).

Valeur retenue : 20480 px. **Ce n'est pas une constante distincte** : `sectioning.split_into_sections`/`_split_by_height_budget` reçoivent `max_section_height` avec pour valeur par défaut `config.MAX_PAGE_HEIGHT_BEFORE_SPLIT` (le même `10 × MAX_TILE_HEIGHT` déjà cité juste avant dans le paragraphe) — le filet de sécurité réutilise donc littéralement le même seuil que le découpage en sections, juste appliqué un cran plus bas (entre éléments d'une section plutôt qu'entre sections d'une page). C'est déjà centralisé dans `config.py`, aucune remontée de littéral à faire.

---

## 4. Nombre d'arêtes par relation du graphe (6 valeurs)

**Emplacement :** `03_methodologie.tex:451-458` (table `tab-graphe_relations`)

> | Relation | Domaine | Définition | Arêtes |
> |---|---|---|---|
> | `contains` | page → élém. | rattachement plat de chaque tuile à sa page | **\aremplir** |
> | `reading_order` | élém. → élém. | successeur dans l'ordre de lecture spatial | **\aremplir** |
> | `layout_adjacency` | élém. ↔ élém. | voisinage 2D (gauche, droite, dessus, dessous) | **\aremplir** |
> | `section_hierarchy` | titre → contenu | arbre des sections respectant l'imbrication h1–h4 | **\aremplir** |
> | `semantic_neighbor` | élém. ↔ élém. | proximité lexicale TF-IDF *(désactivée par défaut)* | 0 |
> | `links_to` | élém. → page | lien hypertexte vers un autre document | **\aremplir** |
> | `continues` | page → page | section suivante d'une page découpée | **\aremplir** |

La légende du tableau précise : « la dernière colonne donne la médiane des arêtes par unité de page ; elle borne ce qu'une politique peut espérer tirer de chaque relation. »

**Source code :**
```python
# scripts/graph_stats.py — médiane et total par relation, sur tout le corpus
per_unit[rel] pour rel in (contains, reading_order, layout_adjacency,
                            section_hierarchy, semantic_neighbor, links_to, continues)
```

**Point d'attention signalé dans vos propres notes :** si la médiane de `section_hierarchy` est proche de zéro, la formulation de la section correspondante (§2.3.3) devrait être renforcée (« la relation est nominale sur ce corpus ») plutôt que seulement nuancée — et la section limites y gagnerait un argument chiffré.

Valeur retenue — `contains` : 6
Valeur retenue — `reading_order` : 5
Valeur retenue — `layout_adjacency` : 8
Valeur retenue — `section_hierarchy` : 4 (→ nominale ? oui/non : **non** — seules 63/230 unités (27 %) ont zéro arête `section_hierarchy` ; la médiane à 4 montre que la relation porte un signal réel sur la majorité du corpus, pas seulement marginal. **Garder la formulation nuancée existante du §2.3.3, ne pas la renforcer en « relation nominale »** — c'est l'inverse de ce que suggérait l'hypothèse de départ des notes.)
Valeur retenue — `links_to` : 8
Valeur retenue — `continues` : 1

**Comment obtenu :** même run que le groupe 1 (230 unités, corpus reconstruit à partir des révisions épinglées) — pour chaque unité, graphe construit via `graph_builder.build_graph(tiles, ...)` (mêmes paramètres par défaut que le pipeline réel : `use_semantic_similarity=False`, `max_linked_text_tiles=3`), comptage des arêtes par relation, puis médiane par relation sur les 230 unités. `continues` compté séparément (1 arête entre chaque section et la section suivante d'une même page découpée, 0 pour la dernière section et pour les pages non découpées) — pas produit par `build_graph` lui-même mais par `corpus_builder.build_page_graph_from_elements`.

**Remarque notable sur `continues` :** la médiane vaut 1 plutôt que 0 — la majorité des 18 articles racines sont si longs qu'ils sont découpés en 9 à 18 sections chacun (seul `Great_Wall_of_China` reste une unité unique sur ce run), donc la grande majorité des 230 unités *sont* une section intermédiaire avec une arête `continues` sortante. Si le rapport laissait entendre que `continues` est une relation rare, ce chiffre dit le contraire sur ce corpus.

`semantic_neighbor` confirmé à 0 partout (médiane et max), cohérent avec `use_semantic_similarity=False` par défaut — aucune correction nécessaire sur cette ligne du tableau.

---

## 5. Nombre maximal de pages suivies depuis l'amorce, *m* (2 emplacements, même valeur)

**Emplacement A :** `03_methodologie.tex:566-570`

> À partir d'une page d'amorce, le système parcourt jusqu'à $m = $ **\aremplir** pages atteignables par ses arêtes `links_to`. Le parcours est d'un seul saut, non récursif : suivre les liens de proche en proche ferait croître le nombre de pages exponentiellement, pour un bénéfice incertain sur des questions qui portent rarement à plus d'un saut du sujet.

**Emplacement B :** `06_annexes.tex:40` (table hyperparamètres)

> Pages suivies depuis l'amorce ($m$) & **\aremplir** & \cref{sec-graphe_corpus}

**Source code :** constante dans `corpus_builder.py` — nombre maximal de pages suivies depuis la page d'amorce.

Valeur retenue : 3 (`build_corpus_graph(..., max_linked_pages: int = 3)`, `src/corpus_builder.py`).

---

## 6. Nombre de paramètres entraînables LoRA (2 emplacements, même valeur)

**Emplacement A :** `03_methodologie.tex:723-726`

> L'adaptation utilise LoRA avec un rang $r = 16$, un facteur d'échelle $\alpha = 32$ et un dropout de 0,05, soit **\aremplir[nombre de paramètres]** paramètres entraînables sur les sept milliards du modèle de base.

**Emplacement B :** `06_annexes.tex:69` (table hyperparamètres)

> Paramètres entraînables & **\aremplir** & \cref{sec-contrastif_lora}

**Source code :**
```python
# dans lora_finetune.py, juste après get_peft_model
model.print_trainable_parameters()   # → "trainable params: X || all params: Y"
```
Profitez-en pour journaliser aussi la liste des modules LoRA résolus (`sorted({n.rsplit('.', 1)[-1] for n, _ in model.named_modules() if "lora_A" in n})`) — c'est le garde-fou déjà promis par le texte (« impose de journaliser la liste obtenue à chaque exécution ») et qui détecterait silencieusement une mise à jour de `transformers` ratant la tour de vision.

Valeur retenue : 44 302 336 (sur 7 000 000 000 au total, soit 0,63 %). Compté directement depuis les tenseurs LoRA réels de `data/lora_reader_adapter/final/adapter_model.safetensors` (somme des tailles de tous les tenseurs du header, hors `__metadata__` — équivalent exact de ce que `print_trainable_parameters()` aurait affiché à l'entraînement). `adapter_config.json` confirme `r=16`, `alpha=32`, `dropout=0.05`, cohérent avec le texte. **Point d'attention pour le garde-fou de journalisation (§2.4.4) :** les modules LoRA réellement résolus sont `['attn.proj', 'down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'qkv', 'up_proj', 'v_proj']` — plus larges que les seuls suffixes LLM déclarés dans `config.py` (`LORA_TARGET_SUFFIXES_LLM = ["q_proj","k_proj","v_proj","o_proj"]`) : le suffixe ViT générique `"proj"` matche par construction *tout* nom de module se terminant par "proj", donc aussi les projections MLP du décodeur (`gate_proj`, `up_proj`, `down_proj`) qui n'étaient pas visées explicitement par la liste LLM. Le nombre de paramètres entraînables ci-dessus reflète cette portée réelle (plus large qu'attention seule), pas seulement q/k/v/o + ViT — à mentionner si le texte affirme que LoRA cible uniquement les projections d'attention.

---

## 7. Longueur de séquence au moment du dépassement mémoire

**Emplacement :** `03_methodologie.tex:775-781`

> Sur une carte de 80 Go, le chemin de calcul par lots des plongements a d'abord échoué sur un dépassement de mémoire GPU, avec 79,16 Go occupés sur les 79,25 Go disponibles. L'usage est réel et non fragmenté : un lot pousse $B(1+K) = 12$ tuiles à travers le modèle simultanément, soit **\aremplir[longueur de séquence]** tokens visuels au total selon le régime d'écrêtage effectif. Journaliser cette longueur est le contrôle qui distingue les deux régimes.

**Source code :** `vlm_client.py` — journaliser `logger.info("tokens du lot: %d", batch["input_ids"].shape[-1])` pendant l'entraînement.

**Repère donné par vos propres notes :** un lot de 12 tuiles à ~15 400 tokens confirme l'écrêtage (régime `max_pixels` actif) ; à ~64 000, il ne s'applique pas — et c'est cette seconde valeur qui expliquerait vraiment le dépassement de 79,16 sur 79,25 Go.

Valeur retenue : ≈ 63 948 tokens (~64 000) (régime observé : écrêté / **non écrêté** — entourer). Déduit de la sonde réelle du groupe 2 : `max_pixels` = 12 845 056 px est très supérieur à toute tuile produite par ce pipeline (la tuile DOM maximale, 2048×2048 = 4 194 304 px, n'est déjà pas écrêtée à 5329 tokens visuels) — le régime « écrêté » ne peut donc structurellement jamais s'appliquer sur ce corpus, quelle que soit la composition du lot. $B(1+K)=12$ tuiles à la hauteur maximale (2048×2048, cas le plus défavorable et le plus probable pour déclencher un OOM) donnent $12 \times 5329 = 63\,948$ tokens visuels, cohérent avec le repère « ~64 000, non écrêté » des notes et avec le dépassement mémoire réel (79,16/79,25 Go). **Valeur exacte non mesurable sans rejouer l'entraînement** : nécessite d'ajouter la journalisation proposée (`logger.info("tokens du lot: %d", ...)` dans `vlm_client.py`/`lora_finetune.py`) et de relancer sur le lot qui a effectivement déclenché l'OOM — la borne ci-dessus est un calcul reproductible avec la configuration réelle du modèle, pas une mesure empirique du lot précis.

---

## 8. Configuration TF-IDF (4 valeurs)

**Emplacement :** `06_annexes.tex:44-48` (table hyperparamètres)

> *Scoreur lexical TF-IDF*
> Plage de $n$-grammes & **\aremplir** & \cref{sec-controleur_scoreur}
> Liste de mots vides & **\aremplir** & \cref{sec-controleur_scoreur}
> Fréquences documentaires minimale et maximale & **\aremplir** & \cref{sec-controleur_scoreur}
> Pondération sous-linéaire des fréquences & **\aremplir[oui/non]** & \cref{sec-controleur_scoreur}

Rappel du texte (§2.4.2, §2.4.9) : le contrôleur, le minage de négatifs et le voisinage sémantique sont censés partager *la même configuration* TF-IDF — c'est désormais affirmé explicitement dans le corps du rapport.

**Source code :**
```python
# scripts/dump_tfidf_config.py
from src.evidence_controller import build_vectorizer   # adapter au nom réel
v = build_vectorizer()
p = v.get_params()
for k in ("ngram_range", "stop_words", "min_df", "max_df", "sublinear_tf"):
    print(f"{k:15s} = {p.get(k)}")
```

**⚠️ Vérifier d'abord** que `hard_negative_mining.py`, `graph_builder.py` (voisinage sémantique) et `evidence_controller.py` instancient bien le même vectoriseur. **Si elles divergent, corriger soit le code (unifier), soit le texte** (retirer l'affirmation d'identité) — c'est l'un des deux seuls cas de cette liste où le résultat de l'analyse peut obliger à revenir sur une phrase déjà écrite plutôt que sur une simple valeur manquante.

Valeur retenue — plage de $n$-grammes : (1, 1) — valeur par défaut de `sklearn.feature_extraction.text.TfidfVectorizer` (non surchargée par le code, qui n'appelle que `TfidfVectorizer(stop_words="english", min_df=1)`)
Valeur retenue — mots vides : `"english"` (liste anglaise intégrée de scikit-learn)
Valeur retenue — min_df / max_df : 1 / 1.0 (min_df explicite ; max_df non surchargé = valeur par défaut, aucune borne supérieure de fréquence documentaire)
Valeur retenue — pondération sous-linéaire : non (`sublinear_tf` non surchargé = `False` par défaut)
Les trois usages partagent-ils la même config ? **oui** — vérifié directement dans le code : `evidence_controller.py:122`, `hard_negative_mining.py:146` et `graph_builder.py:226` instancient chacun, à l'identique, `TfidfVectorizer(stop_words="english", min_df=1)`, aucun paramètre supplémentaire nulle part. Aucune divergence, aucune action de correction nécessaire — l'affirmation du texte (§2.4.2, §2.4.9) est exacte.

---

## 9. Longueur de l'aperçu textuel par tuile

**Emplacement :** `06_annexes.tex:49` (table hyperparamètres)

> Longueur de l'aperçu textuel par tuile & **\aremplir[caractères]** & \cref{sec-tuilage}

**Source code :** constante de troncature dans `tiling.py`, utilisée pour l'aperçu textuel sauvegardé avec chaque tuile (celui qui sert au minage TF-IDF et au scorage du contrôleur).

Valeur retenue : 200 caractères. Trouvé dans `tiling.py`, fonction `build_tiles`, fermeture `_crop_and_patch` : `text_preview=" ".join(e.text_preview for e in source)[:200]`.

---

## 10. Aire d'image minimale du processeur (`min_pixels`)

**Emplacement :** `06_annexes.tex:86` (table hyperparamètres)

> Aire d'image minimale du processeur (`min_pixels`) & **\aremplir** & \cref{sec-contrastif_memoire}

**Source code :** même sonde que le groupe 2 —
```python
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
print(proc.image_processor.min_pixels)
```
Traitez ce point et le groupe 2 (`max_pixels`) dans la même session, une seule sonde couvre les deux.

Valeur retenue : 3 136 px (`ip.size["shortest_edge"]`, même sonde réelle que le groupe 2, transformers 5.13.1, cache HF local).

---

## 11. Optimiseur, taux d'apprentissage, échauffement

**Emplacement :** `06_annexes.tex:70` (table hyperparamètres)

> Optimiseur, taux d'apprentissage, échauffement & **\aremplir** & \cref{sec-contrastif_lora}

**Source code :** arguments du `Trainer` HF ou de la boucle d'entraînement manuelle dans `lora_finetune.py` (`optim=`, `learning_rate=`, `warmup_ratio=`/`warmup_steps=`).

Valeur retenue — optimiseur : AdamW (`torch.optim.AdamW`, `src/lora_finetune.py::train_lora`, appliqué uniquement aux paramètres `requires_grad`)
Valeur retenue — taux d'apprentissage : 1e-4 (`TrainConfig.lr`, valeur par défaut ; l'invocation documentée dans `README.md` — `python3 -m src.lora_finetune --epochs 3 --batch-size 4` — ne surcharge pas `--lr`)
Valeur retenue — échauffement : **aucun**. `train_lora` construit l'optimiseur AdamW puis boucle directement sur les époques : aucun scheduler, aucun `warmup_ratio`/`warmup_steps` nulle part dans le fichier. Le libellé de la table ("Optimiseur, taux d'apprentissage, échauffement") suppose à tort qu'un échauffement existe — soit ajouter un scheduler au code, soit corriger la ligne pour ne plus promettre cette colonne (ex. "Optimiseur, taux d'apprentissage" seuls, avec une note "pas d'échauffement de LR").

---

## 12. Instruction PromptEOL exacte

**Emplacement :** `06_annexes.tex:146-148`

> Cette instruction suit l'entrée — question textuelle ou image de tuile, indifféremment — et l'état caché de la dernière couche à la position de son dernier token est pris comme plongement.
>
> ```
> \aremplir[instruction PromptEOL exacte, depuis le module de configuration]
> ```
>
> Son rôle est de contraindre le modèle à comprimer le sens de l'entrée en une position unique ; c'est son application *à l'identique* aux deux modalités qui place question et tuile dans le même espace, sans tête de projection additionnelle.

**Source code :** copier la chaîne littérale depuis `config.py` — c'est la seule occurrence dans le rapport, le corps du texte n'en donne plus qu'un renvoi vers cette annexe.

Valeur retenue (coller la chaîne exacte) :
```
Résume ce qui précède en un mot :
```
(`_EMBEDDING_PROMPT`, `src/lora_finetune.py:52` ; en français dans le code, contrairement à la paraphrase anglaise "summarize the above in one word" utilisée dans `paper/sections/method.tex:64` — même instruction, langue différente selon le document.)

---

## Ordre suggéré pour la session

1. **Groupes 2 et 10** (sonde `AutoProcessor`, quelques secondes) — décide si le facteur 2,7 du chapitre résultats se transpose en tokens ou tombe à 1,3 ; à traiter en premier car il peut invalider un chiffre déjà écrit.
2. **Groupe 8** (config TF-IDF) — la vérification de cohérence entre les trois usages peut obliger à corriger une phrase, à faire tôt aussi.
3. **Groupes 1, 4** (comptages tuilage, arêtes du graphe) — nécessitent de faire tourner `graph_stats.py` / journaliser `tiling.py` sur le corpus déjà construit ; une session groupée.
4. **Groupes 3, 5, 6, 7, 9, 11, 12** — lectures directes de constantes dans `config.py` / `lora_finetune.py` / sondes ponctuelles, rapides une fois les modules identifiés.

Une fois les 23 valeurs renseignées ci-dessus, reportez-les dans le rapport (`chapitres/03_methodologie.tex` et `chapitres/06_annexes.tex` aux emplacements indiqués), puis retirez la définition de `\aremplir` dans `main.tex:169` — la compilation échouera alors sur toute occurrence oubliée.
