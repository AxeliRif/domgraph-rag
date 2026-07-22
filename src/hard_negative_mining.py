"""
Phase 2b — Hard-negative mining.

Pour chaque paire (question, tuile positive) générée en Phase 2a, on sélectionne
quelques tuiles "négatives difficiles" : des tuiles qui NE contiennent PAS la
réponse mais dont le texte est lexicalement proche de la question — donc des
tuiles qu'un retriever encore peu entraîné pourrait confondre avec la bonne
réponse. C'est plus utile pour l'entraînement contrastif que des négatifs
aléatoires (n'importe quelle autre tuile du corpus), qui sont en général déjà
trivialement faciles à écarter et n'apprennent pas grand-chose au modèle.

On mesure cette proximité avec un TF-IDF + similarité cosinus (scikit-learn) :
c'est un choix volontairement simple et rapide (pas de GPU, pas de modèle à
charger) pour construire le jeu d'entraînement. Une amélioration naturelle
serait de miner avec un embedder appris (ex. le VLM lecteur lui-même, une fois
un premier round de fine-tuning fait — "hard negative mining itératif"), ou un
embedder dédié comme celui de PixelRAG (`Qwen/Qwen3-VL-Embedding-2B`, cf.
README) une fois disponible.

Le filtre anti-faux-négatif reste, lui, en deux temps : un rejet par
sous-chaîne (gratuit, mais purement lexical — ne détecte pas un synonyme, une
couleur de graphique ou un logo qui donnerait la même réponse) suivi,
optionnellement, d'un second filtre où le VLM lui-même "regarde" l'image
candidate et juge s'il peut y retrouver la réponse (cf.
`_is_judged_false_negative`) — passer un `vlm_client` à `mine_hard_negatives`
active ce second filtre, plus coûteux (un appel VLM par candidat non écarté
par le filtre lexical) mais capable d'attraper les faux négatifs non
textuels.

Limite volontaire : le mining se fait tuile par tuile *au sein d'une même
page* (les `Tile.id` du type "tile_0000" ne sont uniques qu'à l'échelle d'une
page, pas d'un corpus multi-pages — cf. corpus_builder.py qui les préfixe par
un slug pour les fusionner en graphe). Miner des négatifs difficiles *entre*
pages d'un même corpus est une extension naturelle mais demanderait de
reprendre ce même préfixage avant d'appeler `mine_hard_negatives`.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - exercised in environments with broken sklearn stack
    TfidfVectorizer = None
    cosine_similarity = None

from .config import CONTRASTIVE_EXAMPLES_PATH, FALSE_NEGATIVE_JUDGE_PROMPT, N_HARD_NEGATIVES
from .qa_generation import QAPair
from .tiling import Tile
from .vlm_client import VLMClient


@dataclass
class ContrastiveExample:
    """Un exemple d'entraînement contrastif : une question, sa tuile positive,
    et quelques tuiles négatives difficiles (toutes tirées de la même page)."""
    qa_id: str
    page_url: str
    page_slug: str
    question: str
    answer: str
    positive_tile_id: str
    hard_negative_tile_ids: list[str]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _fallback_cosine_similarities(question: str, texts: list[str]) -> list[float]:
    question_tokens = Counter(_tokenize(question))
    if not question_tokens:
        return [0.0] * len(texts)

    similarities: list[float] = []
    for text in texts:
        text_tokens = Counter(_tokenize(text))
        if not text_tokens:
            similarities.append(0.0)
            continue

        common = set(question_tokens) & set(text_tokens)
        if not common:
            similarities.append(0.0)
            continue

        dot = sum(question_tokens[token] * text_tokens[token] for token in common)
        q_norm = sum(count * count for count in question_tokens.values()) ** 0.5
        t_norm = sum(count * count for count in text_tokens.values()) ** 0.5
        similarities.append(dot / (q_norm * t_norm) if q_norm and t_norm else 0.0)

    return similarities


def _is_judged_false_negative(vlm_client: VLMClient, tile: Tile, question: str, answer: str) -> bool:
    """Montre l'image de `tile` au VLM et lui demande s'il peut y retrouver
    `answer` -- attrape les faux négatifs que le filtre par sous-chaîne rate
    (réponse reformulée, portée par un visuel plutôt que par du texte : logo,
    couleur, forme). Toute erreur (VLM indisponible, etc.) vaut "pas un faux
    négatif" : le candidat reste éligible plutôt que d'être écarté au premier
    accroc du VLM."""
    try:
        prompt = FALSE_NEGATIVE_JUDGE_PROMPT.format(question=question, answer=answer)
        response = vlm_client.ask(tile.image, prompt, think=False)
    except Exception:
        return False
    return response.strip().upper().startswith("OUI")


def mine_hard_negatives(
    qa_pairs: list[QAPair],
    tiles: list[Tile],
    n_negatives: int = N_HARD_NEGATIVES,
    vlm_client: VLMClient | None = None,
) -> list[ContrastiveExample]:
    """Pour chaque QAPair, sélectionne jusqu'à `n_negatives` tuiles de `tiles`
    (attendues : toutes les tuiles de la même page que la QAPair) qui maximisent
    la similarité cosinus avec la question, tout en excluant :
      - la tuile positive elle-même ;
      - toute tuile dont le texte contient la réponse (pour éviter les "faux
        négatifs" : une autre tuile qui donnerait accidentellement la même
        information, ex. un résumé qui répète un chiffre du corps du texte) ;
      - si `vlm_client` est fourni, toute tuile que le VLM lui-même juge
        capable de répondre à la question (cf. `_is_judged_false_negative`) --
        un second filtre plus coûteux (un appel VLM par candidat) mais qui
        attrape les faux négatifs non lexicaux que le filtre par sous-chaîne
        seul laisse passer.

    Si scikit-learn est indisponible ou cassé dans l'environnement Python du
    notebook, une implémentation de secours basée sur un simple overlap de
    tokens (sans dépendance externe) est utilisée.
    """
    if not qa_pairs or not tiles:
        return []

    tiles_by_id = {t.id: t for t in tiles}
    tile_ids = list(tiles_by_id.keys())
    tile_texts = [tiles_by_id[tid].text_preview or "" for tid in tile_ids]

    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tile_matrix = vectorizer.fit_transform(tile_texts)
    else:
        vectorizer = None
        tile_matrix = None

    examples: list[ContrastiveExample] = []
    for qa in qa_pairs:
        if qa.tile_id not in tiles_by_id:
            continue  # QAPair orpheline (tuile non fournie) -> ignorée

        if vectorizer is not None and tile_matrix is not None:
            question_vector = vectorizer.transform([qa.question])
            similarities = cosine_similarity(question_vector, tile_matrix)[0]
        else:
            similarities = _fallback_cosine_similarities(qa.question, tile_texts)

        answer_lower = qa.answer.lower()
        ranked = sorted(range(len(tile_ids)), key=lambda i: similarities[i], reverse=True)

        negatives: list[str] = []
        for i in ranked:
            tid = tile_ids[i]
            if tid == qa.tile_id:
                continue
            if answer_lower and answer_lower in tile_texts[i].lower():
                continue  # faux négatif lexical : la réponse apparaît aussi ici
            if vlm_client is not None and _is_judged_false_negative(vlm_client, tiles_by_id[tid], qa.question, qa.answer):
                continue  # faux négatif visuel/reformulé : le VLM peut aussi répondre depuis cette tuile
            negatives.append(tid)
            if len(negatives) >= n_negatives:
                break

        examples.append(
            ContrastiveExample(
                qa_id=qa.id,
                page_url=qa.page_url,
                page_slug=qa.page_slug,
                question=qa.question,
                answer=qa.answer,
                positive_tile_id=qa.tile_id,
                hard_negative_tile_ids=negatives,
            )
        )

    return examples


def save_contrastive_examples(examples: list[ContrastiveExample], path: Path = CONTRASTIVE_EXAMPLES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def load_contrastive_examples(path: Path = CONTRASTIVE_EXAMPLES_PATH) -> list[ContrastiveExample]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [ContrastiveExample(**json.loads(line)) for line in f if line.strip()]
