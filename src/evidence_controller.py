"""
Phase 3 — Contrôleur d'évidence en ligne (inspiré de l'Algorithme 1 de
MAGE-RAG, arXiv:2606.15906).

Le graphe construit en Phase 1 (`graph_builder.build_graph`) donne déjà à
chaque noeud élément un attribut `state`, initialisé à "inactive". Ce module
le fait évoluer selon une politique d'actions sous budget, étant donné une
requête utilisateur :

    inactive --(activation)--> active --(ouverture)--> opened
                                  \\--(élagage)-------> pruned

- "inactive" : jamais considéré par le contrôleur (hors de la frontière).
- "active"   : dans la frontière courante, candidat à l'ouverture.
- "opened"   : lu (par le VLM, ou via son texte déjà extrait), son contenu
               fait partie de l'évidence finale renvoyée au modèle lecteur.
- "pruned"   : jugé non pertinent (score sous le seuil), ou laissé de côté
               car le budget d'ouvertures a été épuisé avant de l'atteindre.

Boucle best-first sous budget :
  1. "seed" — les `top_k_seed` noeuds les plus pertinents pour la requête
     passent de inactive à active.
  2. Tant qu'il reste des noeuds actifs et du budget :
       - on prend le noeud actif le plus pertinent ;
       - sous `open_threshold`, il est élagué (gratuit, ne consomme pas le
         budget) ;
       - sinon il est ouvert (consomme une unité de budget) et ses voisins
         "reading_order" encore inactifs sont activés si leur propre score
         dépasse `activate_threshold` — cette propagation d'activation fait
         avancer la frontière au fil de l'exploration, comme un best-first
         search local plutôt qu'un score de pertinence figé calculé une fois
         pour toutes sur toute la page.
  3. Une fois le budget épuisé (ou plus aucun noeud actif), les noeuds encore
     "active" sont élagués : aucun noeud élément ne doit rester dans un état
     transitoire à la fin.

La pertinence est mesurée par TF-IDF + cosinus (même approche que
`hard_negative_mining.py` pour le mining de négatifs, avec le même repli sans
scikit-learn) : un choix volontairement simple qui ne demande ni GPU ni modèle
à charger. Remplacer `_relevance_scores` par un embedder appris est
l'extension naturelle une fois le VLM lecteur fine-tuné (Phase 2).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import networkx as nx

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - environnements avec un sklearn cassé
    TfidfVectorizer = None
    cosine_similarity = None

from .tiling import Tile


@dataclass
class EvidenceControllerConfig:
    budget: int = 5                    # nombre max de noeuds ouverts (= appels VLM si use_vlm=True)
    top_k_seed: int = 3                # noeuds activés d'entrée de jeu (score le plus haut)
    open_threshold: float = 0.05       # score minimal pour ouvrir un noeud actif plutôt que l'élaguer
    activate_threshold: float = 0.05   # score minimal pour qu'un voisin inactif devienne actif
    use_vlm: bool = False              # si True, interroge le VLM sur chaque tuile ouverte (sinon réutilise text_preview)


@dataclass
class ControllerAction:
    """Une transition d'état, pour le journal retourné par `run_evidence_controller`."""
    step: int
    node_id: str
    action: str  # "seed" | "open" | "activate" | "prune"
    score: float
    state_before: str
    state_after: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _fallback_similarities(query: str, texts: list[str]) -> list[float]:
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


def _relevance_scores(query: str, node_ids: list[str], texts: list[str]) -> dict[str, float]:
    """TF-IDF + cosinus entre `query` et le texte de chaque noeud élément
    (repli sans scikit-learn : overlap de tokens, cf. hard_negative_mining.py)."""
    if not node_ids:
        return {}

    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        matrix = vectorizer.fit_transform(texts + [query])
        similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
    else:
        similarities = _fallback_similarities(query, texts)

    return dict(zip(node_ids, (float(s) for s in similarities)))


def _reading_order_neighbors(graph: nx.MultiDiGraph, node_id: str) -> list[str]:
    """Noeuds directement avant/après `node_id` dans l'ordre de lecture."""
    neighbors = [u for u, _, d in graph.in_edges(node_id, data=True) if d.get("relation") == "reading_order"]
    neighbors += [v for _, v, d in graph.out_edges(node_id, data=True) if d.get("relation") == "reading_order"]
    return neighbors


def run_evidence_controller(
    graph: nx.MultiDiGraph,
    tiles: list[Tile],
    query: str,
    config: EvidenceControllerConfig | None = None,
) -> list[ControllerAction]:
    """Fait évoluer l'attribut `state` des noeuds élément de `graph` (en place)
    selon la politique décrite en tête de module, pour la requête `query`.

    Retourne le journal des actions prises (une entrée par transition
    d'état), utile pour l'inspection/la visualisation dans le notebook.
    """
    config = config or EvidenceControllerConfig()

    tiles_by_id = {t.id: t for t in tiles}
    element_ids = [n for n, d in graph.nodes(data=True) if d.get("type") == "element"]
    texts = [graph.nodes[n].get("text_preview", "") for n in element_ids]
    scores = _relevance_scores(query, element_ids, texts)

    for node_id, score in scores.items():
        graph.nodes[node_id]["relevance_score"] = score

    log: list[ControllerAction] = []
    step = 0

    # 1. Seed : active les top_k_seed noeuds inactifs les plus pertinents.
    seed_candidates = sorted(element_ids, key=lambda n: scores.get(n, 0.0), reverse=True)
    for node_id in seed_candidates[: config.top_k_seed]:
        before = graph.nodes[node_id]["state"]
        graph.nodes[node_id]["state"] = "active"
        log.append(ControllerAction(step, node_id, "seed", scores.get(node_id, 0.0), before, "active"))
        step += 1

    vlm_client = None
    if config.use_vlm:
        from .vlm_client import VLMClient

        vlm_client = VLMClient()

    # 2. Boucle best-first sous budget.
    budget_used = 0
    while budget_used < config.budget:
        active_ids = [n for n in element_ids if graph.nodes[n]["state"] == "active"]
        if not active_ids:
            break

        node_id = max(active_ids, key=lambda n: scores.get(n, 0.0))
        score = scores.get(node_id, 0.0)

        if score < config.open_threshold:
            graph.nodes[node_id]["state"] = "pruned"
            log.append(ControllerAction(step, node_id, "prune", score, "active", "pruned"))
            step += 1
            continue

        evidence = graph.nodes[node_id].get("text_preview", "")
        if vlm_client is not None and node_id in tiles_by_id:
            try:
                prompt = (
                    f"Question : {query}\n"
                    "Extrait uniquement les informations de cette image utiles pour y répondre."
                )
                evidence = vlm_client.ask(tiles_by_id[node_id].image, prompt)
            except Exception:
                pass  # VLM indisponible -> on garde le text_preview comme évidence

        graph.nodes[node_id]["state"] = "opened"
        graph.nodes[node_id]["evidence"] = evidence
        log.append(ControllerAction(step, node_id, "open", score, "active", "opened"))
        step += 1
        budget_used += 1

        for neighbor_id in _reading_order_neighbors(graph, node_id):
            if graph.nodes[neighbor_id].get("type") != "element":
                continue
            if graph.nodes[neighbor_id]["state"] != "inactive":
                continue
            neighbor_score = scores.get(neighbor_id, 0.0)
            if neighbor_score >= config.activate_threshold:
                graph.nodes[neighbor_id]["state"] = "active"
                log.append(ControllerAction(step, neighbor_id, "activate", neighbor_score, "inactive", "active"))
                step += 1

    # 3. Finalisation : budget épuisé -> plus aucun noeud "active" ne doit
    # subsister, faute d'avoir pu l'ouvrir.
    for node_id in element_ids:
        if graph.nodes[node_id]["state"] == "active":
            graph.nodes[node_id]["state"] = "pruned"
            log.append(ControllerAction(step, node_id, "prune", scores.get(node_id, 0.0), "active", "pruned"))
            step += 1

    return log


def collect_evidence(graph: nx.MultiDiGraph) -> str:
    """Concatène l'évidence des noeuds "opened", dans l'ordre de lecture
    (l'ordre d'insertion des noeuds dans le graphe, cf. `build_graph`) — c'est
    l'évidence finale, élaguée, à transmettre au modèle lecteur à la place de
    la page entière."""
    return "\n\n".join(
        data.get("evidence", data.get("text_preview", ""))
        for _, data in graph.nodes(data=True)
        if data.get("type") == "element" and data.get("state") == "opened"
    )


def state_counts(graph: nx.MultiDiGraph) -> dict[str, int]:
    """Petit résumé du nombre de noeuds élément par état, pour affichage rapide."""
    counts: dict[str, int] = {"inactive": 0, "active": 0, "opened": 0, "pruned": 0}
    for _, data in graph.nodes(data=True):
        if data.get("type") == "element":
            counts[data.get("state", "inactive")] = counts.get(data.get("state", "inactive"), 0) + 1
    return counts
