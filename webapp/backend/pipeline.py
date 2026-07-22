"""
Site de démo (Phase 3) — orchestre le pipeline Phase 1 (extraction DOM ->
tuilage -> graphe) puis le contrôleur d'évidence de Phase 3 pour répondre à
une question posée sur une page.

Contrairement à `corpus_builder.build_corpus_graph`, ce module ne suit PAS
les liens hypertextes trouvés : ils restent des noeuds "external_page"
(URL + texte d'ancre) exposés tels quels au frontend, qui les affiche comme
des liens Wikipedia cliquables — c'est justement le rôle du site ("montrer
les liens employés, et permettre d'aller chercher plus d'information").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx

from src.config import ANSWER_SYNTHESIS_PROMPT
from src.dom_extraction import extract_dom_elements_async
from src.evidence_controller import EvidenceControllerConfig, collect_evidence, run_evidence_controller
from src.graph_builder import build_graph
from src.tiling import build_tiles
from src.vlm_client import VLMClient


@dataclass
class ExternalLink:
    url: str
    anchor_text: str
    source_tile_id: str
    used_as_evidence: bool  # la tuile source a été "opened" par le contrôleur d'évidence


@dataclass
class AskResult:
    page_url: str
    question: str
    answer: str
    graph: dict
    external_links: list[ExternalLink]
    log: list[dict]


def _graph_to_json(G: nx.MultiDiGraph) -> dict:
    """Sérialise le graphe pour le frontend (cytoscape.js) : tous les
    attributs de noeuds/arêtes sont déjà des types JSON-natifs (str/float/int),
    cf. graph_builder.build_graph — pas de conversion nécessaire."""
    nodes = [{"id": node_id, **data} for node_id, data in G.nodes(data=True)]
    edges = [
        {"id": f"{u}__{v}__{key}", "source": u, "target": v, **data}
        for u, v, key, data in G.edges(keys=True, data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def _extract_external_links(G: nx.MultiDiGraph) -> list[ExternalLink]:
    opened_tile_ids = {
        node_id
        for node_id, data in G.nodes(data=True)
        if data.get("type") == "element" and data.get("state") == "opened"
    }

    links: list[ExternalLink] = []
    seen: set[tuple[str, str]] = set()
    for source, target, data in G.edges(data=True):
        if data.get("relation") != "links_to":
            continue
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        links.append(
            ExternalLink(
                url=G.nodes[target].get("url", ""),
                anchor_text=data.get("anchor_text", ""),
                source_tile_id=source,
                used_as_evidence=source in opened_tile_ids,
            )
        )

    # Les liens portés par une tuile retenue comme évidence d'abord : ce sont
    # ceux que l'utilisateur veut voir en priorité ("les liens employés").
    links.sort(key=lambda link: not link.used_as_evidence)
    return links


async def answer_question(
    url: str,
    question: str,
    controller_config: EvidenceControllerConfig | None = None,
) -> AskResult:
    """Pipeline complet pour une requête du site : DOM -> tuiles -> graphe ->
    contrôleur d'évidence (qui lit chaque tuile ouverte via le VLM) -> synthèse
    finale de la réponse par le VLM à partir de l'évidence rassemblée."""
    controller_config = controller_config or EvidenceControllerConfig(use_vlm=True)

    elements, screenshot = await extract_dom_elements_async(url, wait_until="load")
    if not elements:
        raise ValueError("No content element recognized on this page — check the URL.")

    tiles = build_tiles(elements, screenshot)
    graph = build_graph(tiles, page_url=url, page_title=url)

    log = run_evidence_controller(graph, tiles, question, controller_config)

    evidence = collect_evidence(graph)
    if evidence.strip():
        vlm = VLMClient()
        answer = vlm.ask_text(ANSWER_SYNTHESIS_PROMPT.format(evidence=evidence, question=question))
    else:
        answer = "No sufficiently relevant evidence was found on this page to answer the question."

    return AskResult(
        page_url=url,
        question=question,
        answer=answer,
        graph=_graph_to_json(graph),
        external_links=_extract_external_links(graph),
        log=[asdict(action) for action in log],
    )
