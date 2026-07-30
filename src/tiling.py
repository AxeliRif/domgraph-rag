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
    de générer une multitude de micro-tuiles quasi vides ;
  - deux éléments dont les bounding boxes se chevauchent réellement (ex. un
    paragraphe qui occupe toute la largeur du conteneur pendant qu'une image
    ou un infobox flottant partage visuellement le même espace) sont eux
    aussi tuilés ensemble plutôt que séparément, pour ne jamais produire deux
    tuiles dont les crops se recouvriraient (cf. group_overlapping_elements) --
    sans ça, la même région de pixels de la page apparaît dans deux tuiles à
    la fois.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

from PIL import Image

from .config import MAX_TILE_HEIGHT, MIN_OVERLAP_RATIO, MIN_TILE_HEIGHT, NEVER_MERGE_TAGS
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


def _overlap_ratio(a: DOMElement, b: DOMElement) -> float:
    """Aire d'intersection des bounding boxes de `a` et `b`, rapportée à
    l'aire du plus petit des deux (0 si elles ne se chevauchent pas du tout)."""
    ix0, iy0 = max(a.x, b.x), max(a.y, b.y)
    ix1 = min(a.x + a.width, b.x + b.width)
    iy1 = min(a.y + a.height, b.y + b.height)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller_area = min(a.width * a.height, b.width * b.height)
    return (iw * ih) / smaller_area if smaller_area > 0 else 0.0


def group_overlapping_elements(
    elements: list[DOMElement], min_overlap_ratio: float = MIN_OVERLAP_RATIO,
) -> list[list[DOMElement]]:
    """Partitionne `elements` en groupes qui doivent être tuilés ensemble (un
    seul crop) : `prune_nested_elements` ne gère que l'inclusion totale d'un
    élément dans un autre (l'enfant est alors carrément élagué, cf. un <img>
    dans un <figure>) ; ceci couvre le chevauchement *partiel* entre éléments
    par ailleurs indépendants, que `prune_nested_elements` laisse
    volontairement passer (les deux sont gardés, cf.
    test_partially_overlapping_elements_are_both_kept) -- un paragraphe qui
    occupe toute la largeur du conteneur pendant qu'une image ou un infobox
    flottant partage visuellement le même espace en est l'exemple le plus
    courant (mise en page Wikipédia typique). Sans ce regroupement, les deux
    éléments produiraient chacun leur propre tuile, et les deux crops
    contiendraient la même région de pixels.

    Regroupement par composantes connexes (union-find) sur la relation "se
    chevauche à au moins `min_overlap_ratio`" : si A chevauche B et B
    chevauche C, les trois finissent dans le même groupe même si A et C ne se
    chevauchent pas directement.

    Les titres (NEVER_MERGE_TAGS) n'entrent jamais dans un groupe : ils
    gardent leur logique de pairing dédiée avec l'élément suivant (cf.
    `pending_heading` dans `build_tiles`), qui a besoin de rester un titre
    seul en attente, pas un groupe déjà fusionné.
    """
    candidates = [e for e in elements if e.tag not in NEVER_MERGE_TAGS]
    headings = [e for e in elements if e.tag in NEVER_MERGE_TAGS]

    parent = {e.id: e.id for e in candidates}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a_id: str, b_id: str) -> None:
        root_a, root_b = find(a_id), find(b_id)
        if root_a != root_b:
            parent[root_a] = root_b

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if _overlap_ratio(a, b) >= min_overlap_ratio:
                union(a.id, b.id)

    clusters: dict[str, list[DOMElement]] = {}
    for e in candidates:
        clusters.setdefault(find(e.id), []).append(e)

    return [sorted(group, key=lambda e: (e.y, e.x)) for group in clusters.values()] + [[e] for e in headings]


def build_tiles(
    elements: list[DOMElement],
    screenshot_bytes: bytes,
    max_tile_height: int = MAX_TILE_HEIGHT,
    min_tile_height: int = MIN_TILE_HEIGHT,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
) -> list[Tile]:
    """Construit les tuiles à partir des éléments DOM et de la capture pleine page.

    Les éléments imbriqués dont la bounding box est contenue dans celle d'un
    parent retenu (ex. un <img> dans un <figure>, un <p> dans une <table>) sont
    élagués via `prune_nested_elements` avant tuilage, pour ne pas payer deux
    fois le budget de tuiles/tokens visuels sur le même contenu. Les éléments
    restants dont les bounding boxes se chevauchent partiellement (ex. un
    paragraphe et une image/infobox flottante qui partagent visuellement le
    même espace) sont ensuite regroupés par `group_overlapping_elements` et
    tuilés ensemble, pour qu'aucune région de pixels de la page ne se
    retrouve dupliquée dans deux tuiles différentes.
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

    groups = sorted(
        group_overlapping_elements(elements, min_overlap_ratio),
        key=lambda group: (group[0].y, group[0].x),
    )

    for group in groups:
        if len(group) == 1 and group[0].tag in NEVER_MERGE_TAGS:
            # Un titre ouvre une nouvelle section : on ne le crop pas tout de
            # suite, on attend le groupe suivant (typiquement un <p>) pour les
            # fusionner dans une même tuile — le titre seul, sans son contenu,
            # n'a pas de valeur sémantique pour le lecteur.
            _flush_buffer()
            _flush_pending_heading()
            pending_heading = group[0]
            continue

        if pending_heading is not None:
            combined = [pending_heading, *group]
            x0 = min(e.x for e in combined)
            y0 = min(e.y for e in combined)
            x1 = max(e.x + e.width for e in combined)
            y1 = max(e.y + e.height for e in combined)
            tag = pending_heading.tag + "+" + "+".join(dict.fromkeys(e.tag for e in group))
            _crop_and_patch(x0, y0, x1 - x0, y1 - y0, combined, tag=tag)
            pending_heading = None
            continue

        if len(group) > 1:
            # Chevauchement réel entre éléments par ailleurs indépendants
            # (cf. group_overlapping_elements) : toujours tuilés ensemble,
            # quelle que soit leur hauteur individuelle -- éviter de dupliquer
            # une même région de pixels dans deux tuiles prime sur
            # l'heuristique de fusion des petits éléments ci-dessous.
            _flush_buffer()
            x0 = min(e.x for e in group)
            y0 = min(e.y for e in group)
            x1 = max(e.x + e.width for e in group)
            y1 = max(e.y + e.height for e in group)
            tag = "+".join(dict.fromkeys(e.tag for e in group))
            _crop_and_patch(x0, y0, x1 - x0, y1 - y0, group, tag=tag)
            continue

        el = group[0]
        if el.height < min_tile_height:
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
