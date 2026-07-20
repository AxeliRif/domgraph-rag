"""
Configuration centrale du pipeline DOM-Graph RAG.

Les constantes de rendu (largeur 875px, hauteur max de tuile 1024px) reprennent
volontairement le format de PixelRAG (arXiv:2606.28344) pour rester dans le même
espace de coordonnées — utile si tu veux comparer tes tuiles "guidées par le DOM"
à celles, en grille aveugle, de PixelRAG.
"""
from pathlib import Path

# --- Rendu de page (Phase 1b) ---
RENDER_WIDTH = 2048        # largeur du viewport Playwright, en pixels
MAX_TILE_HEIGHT = 2048    # hauteur max d'une tuile avant découpe forcée ("patching")
MIN_TILE_HEIGHT = 80        # en dessous, l'élément est fusionné avec ses voisins verticaux

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
