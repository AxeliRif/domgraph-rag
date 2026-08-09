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
         encore inactifs sont activés si leur propre score dépasse
         `activate_threshold` -- le long de `reading_order` (même page) ET,
         si le noeud ouvert en a, en franchissant une frontière de document
         via `links_to` (élément -> page cible, rebranchée sur son vrai
         point d'entrée par `corpus_builder.build_corpus_graph`) ou
         `continues` (page -> section suivante d'une page découpée, §3.2) --
         cette propagation d'activation fait avancer la frontière au fil de
         l'exploration, comme un best-first search local plutôt qu'un score
         de pertinence figé calculé une fois pour toutes sur toute la page.
  3. Une fois le budget épuisé (ou plus aucun noeud actif), les noeuds encore
     "active" sont élagués : aucun noeud élément ne doit rester dans un état
     transitoire à la fin.
  4. Si `use_vlm`, les tuiles "opened" sont relues par le VLM pour peupler
     leur évidence -- en parallèle (cf. `_fill_vlm_evidence`), puisque quel
     noeud est ouvert ne dépend jamais du contenu lu par le VLM (seulement des
     scores figés en amont), ces lectures sont donc indépendantes les unes
     des autres.

La pertinence est mesurée par TF-IDF + cosinus (même approche que
`hard_negative_mining.py` pour le mining de négatifs, avec le même repli sans
scikit-learn) : un choix volontairement simple qui ne demande ni GPU ni modèle
à charger. Remplacer `_relevance_scores` par un embedder appris est
l'extension naturelle une fois le VLM lecteur fine-tuné (Phase 2).
"""
from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    max_concurrent_vlm_calls: int = 4  # requêtes VLM en vol en parallèle (cf. _fill_vlm_evidence) ; 4 = défaut Ollama (OLLAMA_NUM_PARALLEL)


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


def _page_elements(graph: nx.MultiDiGraph, page_node_id: str) -> list[str]:
    return [v for _, v, d in graph.out_edges(page_node_id, data=True) if d.get("relation") == "contains"]


def _containing_page(graph: nx.MultiDiGraph, node_id: str) -> str | None:
    for u, _, d in graph.in_edges(node_id, data=True):
        if d.get("relation") == "contains":
            return u
    return None


def _cross_document_neighbors(graph: nx.MultiDiGraph, node_id: str) -> list[str]:
    """Éléments atteignables depuis `node_id` en franchissant une frontière
    de document -- ce que `reading_order` (§Limitations : "closer to a chain
    than the structure it is built from") ne fait jamais : soit un lien
    hypertexte (`links_to`, élément -> page cible, rebranché sur le vrai
    point d'entrée de la page liée par `corpus_builder.build_corpus_graph`,
    pas un placeholder `external_page`), soit `continues` (page -> page
    suivante, pour une page démesurée découpée en sections, §3.2). Dans les
    deux cas on ouvre la porte sur TOUS les éléments de la page cible, à
    charge pour leur propre score de décider s'ils passent le seuil
    d'activation -- symétrique à la façon dont `reading_order` active un
    voisin, jamais une activation automatique."""
    candidates: list[str] = []
    for _, v, d in graph.out_edges(node_id, data=True):
        if d.get("relation") == "links_to" and graph.nodes[v].get("type") == "page":
            candidates.extend(_page_elements(graph, v))
    container = _containing_page(graph, node_id)
    if container is not None:
        for _, v, d in graph.out_edges(container, data=True):
            if d.get("relation") == "continues":
                candidates.extend(_page_elements(graph, v))
    return candidates


def run_evidence_controller(
    graph: nx.MultiDiGraph,
    tiles: list[Tile],
    query: str,
    config: EvidenceControllerConfig | None = None,
    precomputed_scores: dict[str, float] | None = None,
) -> list[ControllerAction]:
    """Fait évoluer l'attribut `state` des noeuds élément de `graph` (en place)
    selon la politique décrite en tête de module, pour la requête `query`.

    `precomputed_scores` : si fourni, remplace le scorer TF-IDF par défaut
    (`_relevance_scores`) -- un noeud absent du dict reçoit un score de 0.0.
    Pensé pour brancher le lecteur VLM fine-tuné (Phase 2, `lora_finetune.py`)
    comme scorer de pertinence à la place du placeholder lexical, sans
    toucher à la politique best-first/budget ci-dessous (cf. Limitations du
    papier, "wiring in the reader's own similarity"). Le calcul de ces scores
    (embeddings image/texte du lecteur) reste hors de ce module, délibérément
    agnostique à la façon dont ils ont été obtenus.

    Retourne le journal des actions prises (une entrée par transition
    d'état), utile pour l'inspection/la visualisation dans le notebook.
    """
    config = config or EvidenceControllerConfig()

    tiles_by_id = {t.id: t for t in tiles}
    element_ids = [n for n, d in graph.nodes(data=True) if d.get("type") == "element"]
    if precomputed_scores is not None:
        scores = {n: precomputed_scores.get(n, 0.0) for n in element_ids}
    else:
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

    # 2. Boucle best-first sous budget. Quel noeud est ouvert ne dépend que de
    # `scores`, figé une fois pour toutes ci-dessus -- jamais du contenu lu par
    # le VLM, qui ne sert qu'à peupler `evidence` a posteriori. On peut donc
    # dérouler toute la simulation d'états sans appeler le VLM, et ne lire les
    # tuiles ouvertes qu'une fois cette liste connue (étape 4) : ces lectures
    # sont indépendantes les unes des autres et peuvent donc être lancées en
    # parallèle plutôt qu'une par une.
    opened_node_ids: list[str] = []
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

        graph.nodes[node_id]["state"] = "opened"
        graph.nodes[node_id]["evidence"] = graph.nodes[node_id].get("text_preview", "")
        log.append(ControllerAction(step, node_id, "open", score, "active", "opened"))
        step += 1
        budget_used += 1
        opened_node_ids.append(node_id)

        frontier = _reading_order_neighbors(graph, node_id) + _cross_document_neighbors(graph, node_id)
        for neighbor_id in frontier:
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

    # 4. Lecture VLM des tuiles ouvertes (remplace le text_preview posé en
    # étape 2 par une évidence extraite par le VLM), en parallèle.
    if config.use_vlm:
        _fill_vlm_evidence(graph, tiles_by_id, opened_node_ids, query, config.max_concurrent_vlm_calls)

    return log


def _fill_vlm_evidence(
    graph: nx.MultiDiGraph,
    tiles_by_id: dict[str, Tile],
    node_ids: list[str],
    query: str,
    max_workers: int,
) -> None:
    """Interroge le VLM sur chaque tuile de `node_ids` (déjà "opened"), en
    parallèle (thread pool) plutôt qu'une par une : ces lectures sont
    indépendantes (une image, un prompt fixe chacune), et Ollama sert
    nativement plusieurs requêtes concurrentes (OLLAMA_NUM_PARALLEL). Un seul
    VLMClient est réutilisé entre threads -- chaque appel est une requête HTTP
    indépendante, il n'y a pas d'état mutable partagé à protéger côté backend
    Ollama."""
    candidates = [n for n in node_ids if n in tiles_by_id]
    if not candidates:
        return

    from .vlm_client import VLMClient

    vlm_client = VLMClient()
    prompt = (
        f"Question : {query}\n"
        "Extrait uniquement les informations de cette image utiles pour y répondre."
    )

    def _read_tile(node_id: str) -> tuple[str, str | None]:
        try:
            return node_id, vlm_client.ask(tiles_by_id[node_id].image, prompt)
        except Exception:
            return node_id, None  # VLM indisponible -> on garde le text_preview déjà en place

    with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as executor:
        for node_id, evidence in executor.map(_read_tile, candidates):
            if evidence is not None:
                graph.nodes[node_id]["evidence"] = evidence


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
