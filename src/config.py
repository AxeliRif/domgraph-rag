"""
Configuration centrale du pipeline DOM-Graph RAG.

Note sur les constantes de rendu : une version antérieure visait 875px de large
(le format de PixelRAG, arXiv:2606.28344) pour rester dans le même espace de
coordonnées. La valeur a depuis été portée à 2048px (viewport = hauteur max de
tuile, cf. method.tex, §Rendering/Patching) pour lire les tuiles à résolution
native ; le paper documente cette valeur de 2048px, pas 875px.
"""
from pathlib import Path

# --- Rendu de page (Phase 1b) ---
RENDER_WIDTH = 2048        # largeur du viewport Playwright, en pixels
MAX_TILE_HEIGHT = 2048    # hauteur max d'une tuile avant découpe forcée ("patching")
MIN_TILE_HEIGHT = 80        # en dessous, l'élément est fusionné avec ses voisins verticaux

# Hauteur de page (px) au-delà de laquelle une page est découpée en sections
# (cf. sectioning.py) plutôt que tuilée d'un bloc : une page démesurément
# longue (ex. un article Wikipedia à la discographie interminable, capture
# ~46 000px pour "The Beatles") peut sinon produire des dizaines de tuiles
# dont le traitement séquentiel (génération QA VLM) prend des heures d'un
# bloc, sans possibilité de reprise partielle en cas d'échec en cours de
# route. Fixé à 10x MAX_TILE_HEIGHT : une page "normale" (quelques tuiles)
# n'est jamais découpée, seules les pages franchement démesurées le sont.
MAX_PAGE_HEIGHT_BEFORE_SPLIT = 10 * MAX_TILE_HEIGHT

# --- Sélecteurs retirés avant capture (bruit d'interface, cf. slide "Rendu et découpage") ---
NOISE_SELECTORS = [
    "nav", "footer", "header",
    "#mw-navigation", "#mw-panel", ".vector-header", ".vector-menu",
    ".navbox", ".mw-editsection", "#siteNotice", ".ambox", ".noprint",
    "script", "style", "noscript",
]

# --- Balises DOM considérées comme noeuds "élément" du graphe (Phase 1d) ---
ELEMENT_TAGS = [
    "h1", "h2", "h3", "h4",
    "p", "table", "figure", "img",
    "ul", "ol", "blockquote", "pre",
]

# --- Balises qui ne doivent jamais être fusionnées avec leurs voisines, même
# si leur hauteur est inférieure à MIN_TILE_HEIGHT : ce sont des frontières
# structurelles (une section commence à un titre), pas du "bruit" à regrouper. ---
NEVER_MERGE_TAGS = {"h1", "h2", "h3", "h4"}

# --- Interconnexion multi-pages (liens hypertextes) ---
# Les liens ne sont extraits que depuis les <p> (paragraphes de texte) : jamais
# depuis un <table> ni une liste de références en bas de page (typiquement
# rendue en <ol>/<li>, pas en <p>). Voir dom_extraction.py.
# Nombre de tuiles de texte (en ordre de lecture, donc en général l'intro de
# l'article) dont on conserve les liens sortants dans le graphe — au-delà,
# les liens sont ignorés pour éviter l'explosion combinatoire du crawl.
MAX_LINKED_TEXT_TILES = 3

# --- Modèle VLM lecteur (Phase 1a) ---
# "ollama" est le chemin le plus simple pour démarrer (un seul `ollama pull`).
# "transformers" donne plus de contrôle et sera nécessaire pour le fine-tuning (Phase 2).
VLM_BACKEND = "ollama"
OLLAMA_MODEL = "qwen3-vl"                    # ollama pull qwen3-vl
HF_MODEL = "Qwen/Qwen2-VL-7B-Instruct"       # cf. slide Phase 1a ; Qwen2.5-VL/Qwen3-VL sont des alternatives plus récentes

# qwen3-vl raisonne en "thinking" avant de répondre, de façon très variable en
# longueur (constaté : de ~1500 à ~2500+ tokens pour une simple extraction
# factuelle sur une tuile). Avec le num_ctx par défaut d'Ollama (4096), un
# raisonnement un peu long épuise la fenêtre de contexte avant que le modèle
# n'émette sa réponse finale -> `content` vide, `done_reason == "length"`,
# silencieusement traité comme SKIP par qa_generation._parse_qa_response.
# On élargit donc la fenêtre par défaut (cf. vlm_client.VLMClient) plutôt que
# de la laisser tronquer une réponse par ailleurs correcte.
OLLAMA_NUM_CTX = 8192

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
CACHE_DIR = DATA_DIR / "cache"

# --- Phase 2 — entraînement contrastif -----------------------------------
# Jeu de données synthétique (question/réponse par tuile + négatifs difficiles),
# construit par qa_generation.py et hard_negative_mining.py. Voir ces modules
# pour le format exact des fichiers .jsonl.
QA_DATASET_DIR = DATA_DIR / "qa_dataset"
QA_IMAGES_DIR = QA_DATASET_DIR / "images"
TILES_MANIFEST_PATH = QA_DATASET_DIR / "tiles_manifest.jsonl"
QA_PAIRS_PATH = QA_DATASET_DIR / "qa_pairs.jsonl"
CONTRASTIVE_EXAMPLES_PATH = QA_DATASET_DIR / "contrastive_examples.jsonl"

# --- Ablation compression (cf. src/image_compression.py) -----------------
# PixelRAG (arXiv:2606.28344) documente la résolution d'image comme un levier
# d'efficacité ("up to 3x token cost reduction at lower resolutions while
# maintaining accuracy") sans préciser le filtre de ré-échantillonnage ni les
# résolutions testées (ni le papier ni le dépôt public ne le détaillent). On
# retient Lanczos, le filtre de downscale haute qualité standard (défaut
# recommandé par Pillow), et un dossier miroir de QA_DATASET_DIR aux images
# compressées : tiles_manifest.jsonl et contrastive_examples.jsonl sont
# réutilisés tels quels des deux côtés (les chemins relatifs qu'ils stockent
# n'encodent pas la résolution), seul `images_root` change entre les deux
# runs de comparaison.
QA_DATASET_COMPRESSED_DIR = DATA_DIR / "qa_dataset_compressed"
TILE_COMPRESSION_SCALE = 0.5  # facteur d'échelle linéaire ; 0.5 ≈ 4x moins de pixels

# Prompt envoyé au VLM pour générer une paire question/réponse ancrée dans une
# seule tuile (cf. qa_generation.py). On lui demande explicitement une
# question dont la réponse n'est trouvable que dans l'image fournie, pour que
# la paire soit utilisable comme exemple positif d'un entraînement contrastif
# question -> tuile (à la façon d'un jeu d'entraînement pour un retriever).
QA_GENERATION_PROMPT = """Tu vois un unique fragment recadré (une "tuile") d'une capture d'écran de page web, pas la page complète.

Génère UNE question précise et autonome dont la réponse est visible dans cette image, et UNIQUEMENT dans cette image (quelqu'un qui n'a pas vu cette tuile ne pourrait pas deviner la réponse par hasard). Évite les questions vagues ("de quoi parle cette image ?") : vise une question factuelle et spécifique (un nombre, un nom, une date, une définition, une légende...).

Si l'image ne contient aucune information exploitable (décorative, vide, illisible), réponds exactement: SKIP

Sinon, réponds STRICTEMENT en JSON, sans texte autour, au format :
{"question": "...", "answer": "..."}"""

# Nombre de négatifs difficiles minés par paire question/tuile positive.
N_HARD_NEGATIVES = 2

# Prompt du "juge" VLM optionnel du hard-negative mining (cf.
# hard_negative_mining._is_judged_false_negative) : le filtre par sous-chaîne
# ne détecte que les faux négatifs lexicaux (la réponse apparaît telle quelle
# dans le texte extrait) et laisse passer les faux négatifs visuels ou
# reformulés (un logo, une couleur dans un graphique, un synonyme). On montre
# donc l'image candidate au VLM lui-même et on lui demande s'il peut y
# retrouver la même information.
FALSE_NEGATIVE_JUDGE_PROMPT = """Tu vois un fragment ("tuile") d'une capture d'écran de page web.

Question : {question}
Réponse attendue (trouvée dans une AUTRE tuile de la même page) : {answer}

Cette image permet-elle, à elle seule, de retrouver cette même information -- même reformulée (synonyme), ou exprimée visuellement (couleur, logo, forme) plutôt que textuellement ? Réponds STRICTEMENT par un seul mot : OUI ou NON."""

# --- Phase 2 — fine-tuning LoRA du VLM lecteur (ViT + LLM dégelés) --------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Suffixes de noms de sous-modules `nn.Linear` ciblés par LoRA. On matche par
# suffixe plutôt que par nom complet car l'architecture interne de Qwen2-VL
# (et donc les noms de modules) peut changer d'une version de `transformers`
# à l'autre — lora_finetune.py fait l'introspection réelle du modèle chargé
# et n'applique LoRA qu'aux modules trouvés (voir `discover_lora_target_modules`).
LORA_TARGET_SUFFIXES_LLM = ["q_proj", "k_proj", "v_proj", "o_proj"]  # décodeur (LLM)
LORA_TARGET_SUFFIXES_VIT = ["qkv", "proj"]                            # tour de vision (ViT)

CONTRASTIVE_TEMPERATURE = 0.05  # température de l'InfoNCE (cf. lora_finetune.py)
LORA_OUTPUT_DIR = DATA_DIR / "lora_reader_adapter"

# --- Phase 3 — site de démo (webapp/) -------------------------------------
# Prompt de synthèse envoyé au VLM en texte seul (VLMClient.ask_text), une
# fois que le contrôleur d'évidence a sélectionné et lu les tuiles pertinentes
# (cf. webapp/backend/pipeline.py) : contrairement à QA_GENERATION_PROMPT (une
# tuile -> une question), celui-ci part de l'évidence déjà rassemblée pour
# produire la réponse finale à la question de l'utilisateur.
ANSWER_SYNTHESIS_PROMPT = """Voici des extraits d'une page web, sélectionnés comme évidence pertinente pour répondre à une question.

Évidence :
{evidence}

Question : {question}

Réponds à la question UNIQUEMENT à partir de cette évidence, en français. Si l'évidence ne permet pas de répondre, dis-le explicitement plutôt que d'inventer."""
