"""
Phase 1c — Tuilage adaptatif et patching.

Contrairement à PixelRAG, qui découpe la page en tuiles de hauteur fixe sans
tenir compte du contenu (voir slide "découpage aveugle des tuiles... aucune
connexion sémantique n'existe entre les tuiles"), ce module s'appuie sur les
tailles réelles des éléments DOM :

  - un élément dont la hauteur dépasse MAX_TILE_HEIGHT est découpé en
    sous-tuiles ("patching"), pour rester lisible par le module de vision ;
  - plusieurs petits éléments voisins (hauteur < MIN_TILE_HEIGHT, ex. des
    lignes de liste courtes) sont fusionnés dans une même tuile, pour éviter
    de générer une multitude de micro-tuiles quasi vides.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

from PIL import Image

from .config import MAX_TILE_HEIGHT, MIN_TILE_HEIGHT, NEVER_MERGE_TAGS
from .dom_extraction import DOMElement, Link


@dataclass
class Tile:
    """Une tuile visuelle : un recadrage de la capture d'écran, rattaché à un ou
    plusieurs éléments DOM sources."""
    id: str
    x: int
    y: int
    width: int
    height: int
    image: Image.Image = field(compare=False, repr=False)
    dom_tag: str = ""
    text_preview: str = ""
    source_element_ids: list[str] = field(default_factory=list, compare=False)
    links: list[Link] = field(default_factory=list, compare=False)  # agrégés depuis les <p> sources


def _is_contained(inner: DOMElement, outer: DOMElement, tolerance: float = 1.0) -> bool:
    """Vrai si la bounding box de `inner` est contenue dans celle de `outer`
    (à `tolerance` pixels près, pour absorber les arrondis de rendu)."""
    return (
        inner.x >= outer.x - tolerance
        and inner.y >= outer.y - tolerance
        and inner.x + inner.width <= outer.x + outer.width + tolerance
        and inner.y + inner.height <= outer.y + outer.height + tolerance
    )


def prune_nested_elements(elements: list[DOMElement]) -> list[DOMElement]:
    """Élague les éléments dont la bounding box est entièrement contenue dans
    celle d'un élément moins profond déjà retenu (ex. un <img> dans un
    <figure>, un <p> dans une <table>) : la tuile du parent couvre déjà
    visuellement l'enfant, qui ne ferait que dupliquer des tuiles/tokens
    visuels pour le même contenu.

    Traite les éléments du moins profond au plus profond (`depth` croissant),
    en ne comparant chaque candidat qu'aux éléments déjà retenus — un enfant
    n'est donc jamais comparé à un autre enfant potentiellement lui-même élagué.
    """
    kept: list[DOMElement] = []
    for el in sorted(elements, key=lambda e: e.depth):
        if any(_is_contained(el, other) for other in kept):
            continue
        kept.append(el)
    return kept


def build_tiles(
    elements: list[DOMElement],
    screenshot_bytes: bytes,
    max_tile_height: int = MAX_TILE_HEIGHT,
    min_tile_height: int = MIN_TILE_HEIGHT,
) -> list[Tile]:
    """Construit les tuiles à partir des éléments DOM et de la capture pleine page.

    Les éléments imbriqués dont la bounding box est contenue dans celle d'un
    parent retenu (ex. un <img> dans un <figure>, un <p> dans une <table>) sont
    élagués via `prune_nested_elements` avant tuilage, pour ne pas payer deux
    fois le budget de tuiles/tokens visuels sur le même contenu.
    """
    elements = prune_nested_elements(elements)
    page_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    page_width, page_height = page_image.size

    tiles: list[Tile] = []
    buffer: list[DOMElement] = []
    pending_heading: DOMElement | None = None

    def _crop_and_patch(x: float, y: float, w: float, h: float, source: list[DOMElement], tag: str) -> None:
        """Ajoute une ou plusieurs tuiles pour la zone donnée, en découpant si h > max_tile_height."""
        n_slices = max(1, math.ceil(h / max_tile_height))
        slice_h = h / n_slices
        for s in range(n_slices):
            sy = int(y + s * slice_h)
            sh = int(min(slice_h, page_height - sy))
            if sh <= 0:
                continue
            box = (
                max(0, int(x)),
                max(0, sy),
                min(page_width, int(x + w)),
                min(page_height, sy + sh),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = page_image.crop(box)
            tiles.append(
                Tile(
                    id=f"tile_{len(tiles):04d}",
                    x=box[0],
                    y=box[1],
                    width=box[2] - box[0],
                    height=box[3] - box[1],
                    image=crop,
                    dom_tag=tag,
                    text_preview=" ".join(e.text_preview for e in source)[:200],
                    source_element_ids=[e.id for e in source],
                    links=[link for e in source for link in e.links],
                )
            )

    def _flush_buffer() -> None:
        if not buffer:
            return
        x0 = min(e.x for e in buffer)
        y0 = min(e.y for e in buffer)
        x1 = max(e.x + e.width for e in buffer)
        y1 = max(e.y + e.height for e in buffer)
        _crop_and_patch(x0, y0, x1 - x0, y1 - y0, buffer.copy(), tag="merged")
        buffer.clear()

    def _flush_pending_heading() -> None:
        nonlocal pending_heading
        if pending_heading is not None:
            _crop_and_patch(
                pending_heading.x, pending_heading.y,
                pending_heading.width, pending_heading.height,
                [pending_heading], tag=pending_heading.tag,
            )
            pending_heading = None

    for el in sorted(elements, key=lambda e: e.y):
        if el.tag in NEVER_MERGE_TAGS:
            # Un titre ouvre une nouvelle section : on ne le crop pas tout de
            # suite, on attend l'élément suivant (typiquement un <p>) pour les
            # fusionner dans une même tuile — le titre seul, sans son contenu,
            # n'a pas de valeur sémantique pour le lecteur.
            _flush_buffer()
            _flush_pending_heading()
            pending_heading = el
            continue

        if pending_heading is not None:
            group = [pending_heading, el]
            x0 = min(e.x for e in group)
            y0 = min(e.y for e in group)
            x1 = max(e.x + e.width for e in group)
            y1 = max(e.y + e.height for e in group)
            _crop_and_patch(x0, y0, x1 - x0, y1 - y0, group, tag=f"{pending_heading.tag}+{el.tag}")
            pending_heading = None
            continue

        if el.height < min_tile_height and el.tag not in NEVER_MERGE_TAGS:
            buffer.append(el)
            span = max(e.y + e.height for e in buffer) - min(e.y for e in buffer)
            if span >= min_tile_height:
                _flush_buffer()
        else:
            _flush_buffer()  # d'abord vider les petits éléments en attente de fusion
            _crop_and_patch(el.x, el.y, el.width, el.height, [el], tag=el.tag)

    _flush_buffer()
    _flush_pending_heading()
    return tiles
