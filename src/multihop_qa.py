"""
Phase 4 (évaluation) — Questions multi-hop à partir de HotpotQA, pour
corriger le biais mono-tuile/mono-page de la génération QA synthétique de
qa_generation.py (cf. README/paper, section Limitations : "Single-hop
training and evaluation data cannot exercise the graph").

HotpotQA (Yang et al. 2018, arXiv:1809.09600) fournit des questions dont la
résolution nécessite de croiser DEUX articles Wikipédia distincts
(`supporting_facts`), avec les phrases exactes de chaque article qui
justifient la réponse -- annotées par des humains, pas par notre propre VLM.
On réutilise ce signal plutôt que de le régénérer :

  1. les (au moins deux) titres d'articles cités par `supporting_facts` sont
     convertis en URLs Wikipédia et rendus par le pipeline existant
     (dom_extraction + tiling, inchangés) ;
  2. pour chaque article, les tuiles dont le texte chevauche une phrase de
     support (cf. `_sentence_overlaps_tile`) sont retenues comme positives --
     une question peut donc avoir des tuiles positives sur DEUX pages
     différentes, contrairement à `hard_negative_mining.ContrastiveExample`
     (scopé à une seule page) ;
  3. les négatifs difficiles sont minés par TF-IDF + cosinus (même mécanisme
     que hard_negative_mining.py) sur le pool des tuiles des DEUX articles
     combinées -- l'extension "mining inter-pages" que ce module laissait
     pour plus tard, ici nécessaire puisque les positifs eux-mêmes viennent
     de deux pages.

Une question dont l'un des articles ne se rend pas (page renommée/supprimée
depuis le dump Wikipédia de 2017 utilisé par HotpotQA, erreur réseau) ou dont
une phrase de support ne se retrouve dans aucune tuile du rendu actuel
(l'article a été réécrit depuis) est abandonnée plutôt que de produire un
exemple incomplet -- cf. `scripts/build_multihop_dataset.py` pour le taux
d'abandon observé en pratique.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .config import N_HARD_NEGATIVES
from .dom_extraction import extract_dom_elements_async
from .tiling import Tile, build_tiles

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - environnements avec un sklearn cassé
    TfidfVectorizer = None
    cosine_similarity = None
    ENGLISH_STOP_WORDS = frozenset(
        "a an the of to in and is are was were be been being for on with as by at from this that".split()
    )


class ArticleUnavailableError(Exception):
    """Un des articles cités par `supporting_facts` ne s'est pas rendu
    (renommé/supprimé depuis le dump Wikipédia 2017 utilisé par HotpotQA,
    erreur réseau) -- distinct d'une phrase de support non rattachée, pour
    que l'appelant puisse rapporter les deux taux d'abandon séparément."""


@dataclass
class MultiHopExample:
    """Un exemple contrastif dont les tuiles positives peuvent venir de DEUX
    pages Wikipédia différentes -- contraste avec `ContrastiveExample`
    (hard_negative_mining.py), scopé à une seule page par construction."""
    id: str
    question: str
    answer: str
    hop_type: str  # "bridge" | "comparison", cf. schéma HotpotQA
    positive_tiles: list[tuple[str, str]] = field(default_factory=list)       # (page_slug, tile_id)
    hard_negative_tiles: list[tuple[str, str]] = field(default_factory=list)  # (page_slug, tile_id)


def slug_for_title(title: str) -> str:
    """Identifiant court dérivé d'un titre d'article Wikipédia (ex. "Retrieval-augmented generation")."""
    return title.replace(" ", "_")


def url_for_title(title: str) -> str:
    """URL Wikipédia (anglais) correspondant à un titre d'article `supporting_facts`."""
    return f"https://en.wikipedia.org/wiki/{slug_for_title(title)}"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _fallback_similarities(query: str, texts: list[str]) -> list[float]:
    """Repli sans scikit-learn (même principe que hard_negative_mining.py)."""
    query_tokens = Counter(_tokenize(query))
    if not query_tokens:
        return [0.0] * len(texts)

    similarities: list[float] = []
    for text in texts:
        text_tokens = Counter(_tokenize(text))
        common = set(query_tokens) & set(text_tokens)
        if not text_tokens or not common:
            similarities.append(0.0)
            continue
        dot = sum(query_tokens[t] * text_tokens[t] for t in common)
        q_norm = sum(c * c for c in query_tokens.values()) ** 0.5
        t_norm = sum(c * c for c in text_tokens.values()) ** 0.5
        similarities.append(dot / (q_norm * t_norm) if q_norm and t_norm else 0.0)
    return similarities


def _sentence_overlaps_tile(sentence: str, tile_text: str, min_token_overlap_ratio: float = 0.5) -> bool:
    """Vrai si au moins `min_token_overlap_ratio` des tokens significatifs
    (hors mots vides) de `sentence` apparaissent dans `tile_text`. Un
    recouvrement de tokens plutôt qu'une correspondance de sous-chaîne exacte
    tolère une reformulation mineure de l'article depuis la version utilisée
    par HotpotQA (le dump Wikipédia de 2017) -- constaté en pratique : une
    correspondance de sous-chaîne exacte n'aboutissait presque jamais sur un
    échantillon de test, alors que l'article existe toujours et porte
    toujours la même information (cf. scripts/build_multihop_dataset.py).

    Nommé `min_token_overlap_ratio` (et non `min_overlap_ratio`, cf.
    tiling.group_overlapping_elements) pour éviter la confusion avec le
    recouvrement *spatial* de bounding boxes utilisé par le tuilage -- même
    mot, notion sans rapport ici (chevauchement lexical, pas géométrique)."""
    sentence_tokens = set(_tokenize(sentence)) - ENGLISH_STOP_WORDS
    if not sentence_tokens:
        return False
    tile_tokens = set(_tokenize(tile_text))
    overlap = sentence_tokens & tile_tokens
    return len(overlap) / len(sentence_tokens) >= min_token_overlap_ratio


def find_positive_tiles(tiles: list[Tile], supporting_sentences: list[str]) -> list[str]:
    """Ids des tuiles dont le texte chevauche au moins une phrase de support."""
    return [
        tile.id for tile in tiles
        if any(_sentence_overlaps_tile(sentence, tile.text_preview) for sentence in supporting_sentences)
    ]


def mine_cross_page_hard_negatives(
    question: str,
    pages: list[tuple[str, list[Tile]]],
    positive_ids: set[tuple[str, str]],
    n_negatives: int = N_HARD_NEGATIVES,
) -> list[tuple[str, str]]:
    """Mine des négatifs difficiles sur le pool COMBINÉ des tuiles de toutes
    les pages fournies (même mécanisme TF-IDF + cosinus que
    hard_negative_mining.mine_hard_negatives), plutôt qu'une page à la fois --
    l'extension inter-pages nécessaire ici puisque les positifs eux-mêmes
    viennent de deux pages différentes."""
    pooled: list[tuple[str, str, str]] = [
        (slug, tile.id, tile.text_preview or "")
        for slug, tiles in pages for tile in tiles
        if (slug, tile.id) not in positive_ids
    ]
    if not pooled:
        return []

    texts = [text for _, _, text in pooled]
    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        try:
            matrix = vectorizer.fit_transform([*texts, question])
            similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
        except ValueError:  # vocabulaire vide (textes tous vides)
            similarities = _fallback_similarities(question, texts)
    else:
        similarities = _fallback_similarities(question, texts)

    ranked = sorted(range(len(pooled)), key=lambda i: similarities[i], reverse=True)
    return [(pooled[i][0], pooled[i][1]) for i in ranked[:n_negatives]]


def _sentences_by_title(hotpot_record: dict) -> dict[str, list[str]]:
    sf_titles = hotpot_record["supporting_facts"]["title"]
    sf_sent_ids = hotpot_record["supporting_facts"]["sent_id"]
    ctx_titles = hotpot_record["context"]["title"]
    ctx_sentences = hotpot_record["context"]["sentences"]

    distinct_titles = list(dict.fromkeys(sf_titles))
    sentences: dict[str, list[str]] = {title: [] for title in distinct_titles}
    for title, sent_id in zip(sf_titles, sf_sent_ids):
        if title not in sentences:
            continue
        try:
            ctx_idx = ctx_titles.index(title)
            sentences[title].append(ctx_sentences[ctx_idx][sent_id])
        except (ValueError, IndexError):
            continue  # incohérence rare entre supporting_facts et context -- phrase ignorée
    return sentences


async def build_multihop_example(hotpot_record: dict) -> MultiHopExample | None:
    """Construit un MultiHopExample à partir d'un enregistrement HotpotQA brut
    (schéma `hotpotqa/hotpot_qa`, config "distractor") : rend les articles
    cités par `supporting_facts`, tuile chacun via le pipeline existant, et
    retient les tuiles qui chevauchent une phrase de support comme positifs.

    Retourne None si moins de deux articles distincts sont cités (rare mais
    possible dans les données brutes) ou si aucune phrase de support ne se
    rattache à une tuile du rendu actuel (l'article a été réécrit depuis) --
    un exemple incomplet n'est jamais préférable à aucun exemple. Lève
    `ArticleUnavailableError` si un article ne se rend pas du tout (renommé/
    supprimé depuis le dump 2017, erreur réseau) : distingué du cas
    précédent pour que l'appelant puisse rapporter les deux taux d'abandon
    séparément (cf. scripts/build_multihop_dataset.py)."""
    sentences_by_title = _sentences_by_title(hotpot_record)
    distinct_titles = list(sentences_by_title)
    if len(distinct_titles) < 2:
        return None

    pages: list[tuple[str, list[Tile]]] = []
    positive_ids: set[tuple[str, str]] = set()
    for title in distinct_titles:
        try:
            elements, screenshot = await extract_dom_elements_async(url_for_title(title), wait_until="load")
        except Exception as exc:
            raise ArticleUnavailableError(f"{title!r}: {exc!r}") from exc

        tiles = build_tiles(elements, screenshot)
        slug = slug_for_title(title)
        pages.append((slug, tiles))

        positives = find_positive_tiles(tiles, sentences_by_title[title])
        if not positives:
            return None  # phrase de support introuvable dans ce rendu (article modifié depuis 2017)
        positive_ids.update((slug, tile_id) for tile_id in positives)

    hard_negatives = mine_cross_page_hard_negatives(hotpot_record["question"], pages, positive_ids)

    return MultiHopExample(
        id=hotpot_record["id"],
        question=hotpot_record["question"],
        answer=hotpot_record["answer"],
        hop_type=hotpot_record.get("type", ""),
        positive_tiles=sorted(positive_ids),
        hard_negative_tiles=hard_negatives,
    )
