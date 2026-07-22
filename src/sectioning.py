"""
Phase 1e (extension) — Découpage d'une page trop grande en sections.

Certaines pages (ex. un article Wikipedia à la discographie interminable)
produisent une capture pleine page si haute que le budget de tuilage explose
et bloque tout le pipeline en aval (tuilage, génération QA VLM) sur une seule
unité de travail impossible à reprendre partiellement en cas d'échec en cours
de route (cf. incident The_Beatles : la page a bloqué `scripts/build_dataset.py`
plusieurs heures sur une seule tuile, sans qu'aucune progression ne soit
sauvegardée pour cette page).

Plutôt que d'exclure ces pages -- ce qui irait à l'encontre de l'objectif
"documents longs" du projet (cf. LongDocURL, MMLongBench-Doc) -- on les
découpe en sections aux frontières de titres (h1/h2). Chaque section est
ensuite traitée comme une page à part entière par le reste du pipeline
(son propre noeud "page" dans le graphe, cf. corpus_builder.py ; sa propre
unité de travail check-pointée, cf. scripts/build_dataset.py). Une section
qui reste malgré tout trop haute (ex. un unique h2 suivi d'une table de
discographie géante) est re-découpée par un budget de hauteur fixe, en filet
de sécurité -- le même compromis que le "patching" déjà appliqué tuile par
tuile pour un élément DOM individuel trop grand (cf. tiling.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import MAX_PAGE_HEIGHT_BEFORE_SPLIT
from .dom_extraction import DOMElement

_SECTION_BOUNDARY_TAGS = {"h1", "h2"}


@dataclass
class PageSection:
    """Un groupe d'éléments DOM promu au rang de section indépendante."""
    index: int
    title: str  # texte du titre de section ; vide pour le préambule avant le premier h1/h2
    elements: list[DOMElement]


def page_height(elements: list[DOMElement]) -> float:
    if not elements:
        return 0.0
    return max(e.y + e.height for e in elements) - min(e.y for e in elements)


def is_oversized_page(elements: list[DOMElement], max_height: float = MAX_PAGE_HEIGHT_BEFORE_SPLIT) -> bool:
    return page_height(elements) > max_height


def _split_by_height_budget(elements: list[DOMElement], max_height: float) -> list[list[DOMElement]]:
    """Filet de sécurité pour une section encore trop haute à elle seule :
    la retranche en morceaux d'au plus `max_height`, sans tenir compte du
    contenu (les éléments restent chacun intacts, seule la frontière entre
    morceaux est arbitraire -- comme le patching d'un élément DOM trop grand,
    cf. tiling._crop_and_patch, mais un cran au-dessus, entre éléments plutôt
    qu'à l'intérieur d'un seul)."""
    chunks: list[list[DOMElement]] = []
    current: list[DOMElement] = []
    chunk_start_y = elements[0].y
    for el in elements:
        if current and (el.y + el.height - chunk_start_y) > max_height:
            chunks.append(current)
            current = []
            chunk_start_y = el.y
        current.append(el)
    if current:
        chunks.append(current)
    return chunks


def split_into_sections(
    elements: list[DOMElement],
    max_section_height: float = MAX_PAGE_HEIGHT_BEFORE_SPLIT,
) -> list[PageSection]:
    """Découpe `elements` en sections aux frontières de titres h1/h2 (le
    contenu avant le tout premier h1/h2 forme une section "préambule" sans
    titre, ex. le paragraphe d'intro d'un article Wikipedia avant son
    sommaire), puis re-découpe par budget de hauteur toute section qui
    dépasse encore `max_section_height` à elle seule.

    Ne filtre pas selon `is_oversized_page` : appelant à charge de décider
    si le découpage est nécessaire. Sur une page qui ne contient aucun h1/h2,
    renvoie une seule section (le filet de sécurité par hauteur s'applique
    quand même si elle est trop haute)."""
    if not elements:
        return []

    ordered = sorted(elements, key=lambda e: e.y)
    raw_sections: list[tuple[str, list[DOMElement]]] = []
    current: list[DOMElement] = []
    current_title = ""
    for el in ordered:
        if el.tag in _SECTION_BOUNDARY_TAGS:
            if current:
                raw_sections.append((current_title, current))
                current = []
            current_title = el.text_preview
        current.append(el)
    if current:
        raw_sections.append((current_title, current))

    sections: list[PageSection] = []
    for title, els in raw_sections:
        if page_height(els) <= max_section_height:
            sections.append(PageSection(index=len(sections), title=title, elements=els))
            continue
        for i, chunk in enumerate(_split_by_height_budget(els, max_section_height)):
            chunk_title = title if i == 0 else f"{title} (suite {i + 1})"
            sections.append(PageSection(index=len(sections), title=chunk_title, elements=chunk))

    return sections
