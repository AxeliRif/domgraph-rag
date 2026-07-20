# DOM-Graph RAG — Phase 1 (starter)

Squelette de projet pour démarrer la pipeline décrite dans *Visuel DOM-Graph RAG* :
combiner le rendu pixel de **PixelRAG** avec le graphe multi-granulaire de
**MAGE-RAG**, mais en remplaçant le découpage aveugle en grille par un
découpage guidé par le DOM.

## 0. Les deux articles source sont réels — et open-source

En préparant ce squelette, j'ai vérifié les deux articles cités dans tes slides :
ce sont deux publications récentes (juin 2026), avec code public.

- **PixelRAG** — *PIXELRAG: Web Screenshots Beat Text for Retrieval-Augmented
  Generation* (Yichuan Wang et al., UC Berkeley/BAIR/Berkeley NLP).
  [arXiv:2606.28344](https://arxiv.org/abs/2606.28344) ·
  [github.com/StarTrail-org/PixelRAG](https://github.com/StarTrail-org/PixelRAG)
  (Apache-2.0, `pip install pixelrag`).
  Rendu à 875px de large, tuiles de 1024px de haut → correspond bien aux
  "875x1024 pixels" de ta slide. Le modèle d'embedding utilisé est
  `Qwen/Qwen3-VL-Embedding-2B` (LoRA). Une différence à noter : l'index FAISS
  complet pour Wikipédia fait ~217 Go d'après le README actuel du repo, et non
  <120 Go — soit la slide se référait à une config différente (ex. index
  compressé), soit à une version antérieure du papier ; à vérifier si le
  chiffre exact t'importe pour ton rapport.
- **MAGE-RAG** — *Multigranular Adaptive Graph Evidence for Agentic
  Multimodal RAG in Long-Document QA* (Yilong Zuo et al.).
  [arXiv:2606.15906](https://arxiv.org/abs/2606.15906) ·
  code : `github.com/laonuo2004/MAGE-RAG` (pas de paquet pip identifié — plutôt
  un repo de recherche à cloner et lire).
  Résultats publiés : 52.75 (accuracy) sur LongDocURL, 53.26 acc / 51.19 F1
  sur MMLongBench-Doc — ce sont les chiffres de référence à battre pour ta
  Phase 4.

Point notable : une critique indépendante (VentureBeat) sur PixelRAG confirme
exactement le problème que ta slide 6 identifie — *"it slices pages by fixed
pixel height, meaning a table or paragraph can get cut in half mid-tile with
no awareness of content boundaries"*. Ton angle (tuilage guidé par le DOM) répond
donc à une limite réelle et déjà repérée par la communauté, pas seulement à une
hypothèse — bon signal pour la suite.

## 1. Jouer avec PixelRAG et MAGE-RAG directement

Avant de coder ta propre version, ça vaut le coup de voir tourner les deux
projets sources.

**PixelRAG — quickstart sur un seul document (pas besoin des 217 Go d'index Wikipédia) :**
```bash
pip install 'pixelrag[index]'

curl -L -o paper.pdf https://raw.githubusercontent.com/StarTrail-org/PixelRAG/main/assets/pixelrag-paper.pdf

cat > pixelrag.yaml << 'EOF'
source:
  type: local
  path: ./paper.pdf
embed:
  model: Qwen/Qwen3-VL-Embedding-2B
  device: auto   # cuda sur Linux, mps sur Mac Apple Silicon, cpu sinon
output: ./paper_index
EOF

pixelrag index build      # ~1-3 min
pixelrag serve --index-dir ./paper_index --port 30001

curl -X POST http://localhost:30001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"text": "Overview of PixelRAG and the diagram"}], "n_docs": 1}'
```
Le paquet expose aussi `from pixelrag_render import render_url` — pratique
pour comparer, sur une même page, le découpage en grille aveugle de PixelRAG
et ton découpage guidé par le DOM (bon candidat d'ablation pour ton rapport).

**MAGE-RAG — pas de paquet pip, on clone et on lit le code :**
```bash
git clone https://github.com/laonuo2004/MAGE-RAG.git
```
Regarde en particulier la construction du graphe offline (noeuds page/élément,
relations *containment*, *reading order*, *layout adjacency*, *section
hierarchy*, *semantic-neighbor*) et le contrôleur d'évidence en ligne
(Algorithme 1 du papier) — c'est la partie que la Phase 3 de ton projet
réimplémentera.

## 2. Installer ce squelette

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate sous Windows
pip install -r requirements.txt
playwright install chromium      # télécharge le navigateur headless

# Backend VLM recommandé pour démarrer (léger, tu as déjà l'habitude d'Ollama) :
ollama pull qwen3-vl
```

Dans VS Code : ouvre le dossier `domgraph-rag/`, installe l'extension
**Python** + **Jupyter**, sélectionne l'interpréteur `.venv`, puis ouvre
`notebooks/01_pipeline_playground.ipynb` et choisis ce même environnement
comme kernel.

## 3. Démarrer

```bash
# Vérifie que tout fonctionne (page locale, pas besoin d'internet) :
python3 tests/test_pipeline_smoke.py
```
Puis ouvre `notebooks/01_pipeline_playground.ipynb` dans VS Code et exécute
les cellules une à une. Par défaut il tourne sur une page de test locale ; il
suffit de changer la variable `URL` par une vraie page (ex. un article
Wikipedia) pour tester en conditions réelles — le code ne change pas,
Playwright gère aussi bien `file://` qu'une URL distante.

## 4. Structure du projet

```
domgraph-rag/
├── src/
│   ├── config.py               # constantes (tailles de tuiles, sélecteurs de bruit, modèle VLM, hyperparamètres LoRA...)
│   ├── dom_extraction.py       # Phase 1a/1b — Playwright + extraction des bounding boxes DOM
│   ├── tiling.py                # Phase 1c — tuilage adaptatif + patching (pas une grille fixe)
│   ├── graph_builder.py         # Phase 1d — graphe page/éléments + ordre de lecture + export XML
│   ├── corpus_builder.py        # Phase 1e — graphe multi-pages (liens hypertextes)
│   ├── vlm_client.py            # Phase 1a — wrapper VLM (Ollama ou Transformers)
│   ├── qa_generation.py         # Phase 2a — paires question/réponse synthétiques par tuile
│   ├── hard_negative_mining.py  # Phase 2b — négatifs difficiles (TF-IDF + cosinus)
│   ├── contrastive_dataset.py   # Phase 2c — Dataset PyTorch (question, positif, négatifs)
│   ├── lora_finetune.py         # Phase 2d — fine-tuning LoRA du VLM lecteur (ViT + LLM)
│   └── evidence_controller.py   # Phase 3 — contrôleur d'évidence en ligne (Algorithme 1 MAGE-RAG)
├── webapp/                      # Phase 3 — site de démo (question -> réponse + graphe interactif)
│   ├── backend/
│   │   ├── main.py              # API FastAPI (POST /api/ask) + fichiers statiques du frontend
│   │   └── pipeline.py          # orchestre DOM -> tuiles -> graphe -> contrôleur d'évidence -> VLM
│   └── frontend/                # HTML/CSS/JS statique, sans build step (cytoscape.js via CDN)
├── notebooks/
│   └── 01_pipeline_playground.ipynb   # bout en bout Phase 1 + démo Phase 2/3, avec visualisations
├── tests/
│   ├── fixtures/sample_page.html      # page de test locale (aucun accès réseau requis)
│   ├── test_pipeline_smoke.py         # script de vérification bout-en-bout (Phase 1)
│   └── test_phase2_smoke.py           # génération QA + mining + dataset (sans VLM ni GPU réels)
└── requirements.txt
```

**Ce qui est déjà fonctionnel et testé** (bout en bout, sur la page de test locale) :
extraction DOM avec retrait du bruit d'interface → tuilage adaptatif (fusion
des petits éléments, découpe des éléments trop grands) → construction du
graphe 2 niveaux avec ordre de lecture spatial (pas l'ordre du DOM) → export
XML lisible par un modèle lecteur.

**Simplifications volontaires à connaître** (pistes d'amélioration naturelles) :
- les éléments DOM imbriqués (ex. un `<img>` dans un `<figure>`) peuvent se
  chevaucher — pas de déduplication hiérarchique pour l'instant ;
- le graphe a maintenant 4 des 5 relations de MAGE-RAG (`contains`,
  `reading_order`, `layout_adjacency`, `section_hierarchy`, + `links_to` côté
  domgraph-rag pour l'interconnexion multi-pages) ; il manque encore
  *semantic-neighbor*, qui demande des embeddings et reste donc une tâche de
  Phase 2 ;
- chaque noeud élément a déjà un attribut `state` (`inactive` par défaut) —
  prêt pour que le contrôleur d'évidence de la Phase 3 le fasse évoluer.

## 5. Phase 2 — entraînement contrastif (implémentée)

Génère un jeu de données synthétique (question → tuile) et fine-tune le VLM
lecteur par contraste. Quatre étapes, une par module :

1. **`qa_generation.py`** — le VLM lit chaque tuile et pose une question dont
   la réponse n'est visible que dans cette tuile (prompt `QA_GENERATION_PROMPT`
   dans `config.py`). Toutes les tuiles sont sauvegardées sur disque
   (`data/qa_dataset/images/`), pas seulement celles retenues comme positif,
   pour que le mining (étape 2) puisse y piocher des négatifs.
2. **`hard_negative_mining.py`** — pour chaque question, sélectionne quelques
   tuiles de la même page lexicalement proches (TF-IDF + cosinus,
   scikit-learn) mais qui ne contiennent pas la réponse : des négatifs
   "difficiles", plus utiles à l'entraînement qu'une tuile prise au hasard.
   Fonctionne page par page (les `Tile.id` ne sont uniques qu'à l'échelle
   d'une page) ; miner des négatifs *entre* pages d'un corpus est une
   extension naturelle décrite dans le docstring du module.
3. **`contrastive_dataset.py`** — assemble un `torch.utils.data.Dataset`
   (question, image de la tuile positive, images des tuiles négatives) à
   partir des `.jsonl` produits par les deux étapes précédentes.
4. **`lora_finetune.py`** — fine-tune `Qwen2VLForConditionalGeneration` par
   LoRA, appliqué à la fois sur le LLM et sur la tour de vision (ViT), avec
   une loss InfoNCE (le modèle générateur est transformé en encodeur via un
   prompt court, à la façon des VLM utilisés comme encodeurs universels dans
   la littérature récente — référence exacte à vérifier avant de la citer
   dans le rapport, dans le même esprit que la note de sourcing en tête de ce
   README). Nécessite le backend `transformers` + `peft` (voir
   `requirements.txt`, section commentée) et idéalement un GPU ; s'exécute via :
   ```bash
   python3 -m src.lora_finetune --epochs 3 --batch-size 4
   ```

Voir la section 7-9 de `notebooks/01_pipeline_playground.ipynb` pour une démo
des étapes 1 à 3 sur la page chargée en Phase 1, et
`tests/test_phase2_smoke.py` pour des tests qui n'ont besoin ni de VLM réel ni
de GPU (génération avec un client factice, mining, assemblage du dataset).

## 6. Phase 3 — contrôleur d'évidence + site de démo (implémentée)

`src/evidence_controller.py` fait évoluer l'attribut `state` de chaque noeud
élément (`inactive → active → opened`/`pruned`) selon la boucle
activer/ouvrir/chercher/élaguer sous budget de l'Algorithme 1 de MAGE-RAG,
étant donné une requête. Voir la section 10 du notebook pour une démo pas à
pas, et `src/evidence_controller.py` pour la politique complète (pertinence
TF-IDF + cosinus, best-first sous budget).

`webapp/` expose cette même boucle derrière un petit site : un formulaire
"URL Wikipédia + question" lance le pipeline complet (extraction DOM →
tuilage → graphe → contrôleur d'évidence, qui lit chaque tuile ouverte via le
VLM) puis affiche la réponse synthétisée, le graphe interactif (catégorie de
contenu en couleur, état du contrôleur en bordure, arêtes filtrables par
relation) et les pages Wikipédia liées (`links_to`) — avec un badge qui
signale celles portées par une tuile effectivement retenue comme évidence.

```bash
# Ollama doit tourner avec le modèle configuré (config.OLLAMA_MODEL) :
ollama pull qwen3-vl

pip install -r requirements.txt   # ajoute fastapi + uvicorn
uvicorn webapp.backend.main:app --reload
```
Puis ouvrir http://127.0.0.1:8000/. Chaque requête relance le pipeline en
direct (pas de corpus pré-construit) — compter de quelques dizaines de
secondes à quelques minutes selon la page et le budget du contrôleur (le VLM
est interrogé une fois par tuile ouverte, plus une fois pour synthétiser la
réponse finale).

## 7. Feuille de route (Phase 4)

- **Phase 4 — évaluation** : benchmarks LongDocURL, MMLongBench-Doc, SimpleQA ;
  diagnostic parser/rank/reader loss ; rédaction.
