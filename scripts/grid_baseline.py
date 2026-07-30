"""
Réimplémentation minimale du tuilage en grille fixe (à la PixelRAG) : coupe
la capture d'écran en bandes horizontales de hauteur fixe, sans se soucier
des limites de contenu. Sert uniquement de baseline de comparaison pour
l'ablation tuilage DOM vs grille (cf. scripts/dom_vs_grid_experiment.py) --
n'est pas utilisée par le pipeline principal (src/tiling.py).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

from PIL import Image

from src.dom_extraction import DOMElement

# PixelRAG documente des tuiles de 1024px de haut (à 875px de large, cf.
# README) -- on reprend cette même hauteur mais à notre propre largeur de
# rendu (2048px, cf. config.RENDER_WIDTH) : ne faire varier que la stratégie
# de découpe entre les deux conditions, jamais la résolution de rendu.
GRID_TILE_HEIGHT = 1024

# Même troncature que Tile.text_preview (src/tiling.py, _crop_and_patch) --
# comparaison à budget de texte égal pour le scorer lexical des deux conditions.
_TEXT_PREVIEW_CHARS = 200


@dataclass
class GridTile:
    id: str
    x: int
    y: int
    width: int
    height: int
    text_preview: str = ""


def build_grid_tiles(
    elements: list[DOMElement], screenshot_bytes: bytes, tile_height: int = GRID_TILE_HEIGHT,
) -> list[GridTile]:
    """Découpe la capture en bandes horizontales de hauteur fixe, sans
    connaissance du DOM. Le texte associé à chaque bande (pour un scorer
    lexical comparable à celui du tuilage DOM) est l'agrégat des éléments
    dont la plage verticale chevauche la bande -- un élément qui chevauche
    deux bandes contribue son texte aux deux, comme il apparaîtrait
    visuellement coupé dans les deux crops."""
    page_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    page_width, page_height = page_image.size

    tiles: list[GridTile] = []
    n_bands = max(1, math.ceil(page_height / tile_height))
    for i in range(n_bands):
        y0 = i * tile_height
        y1 = min(y0 + tile_height, page_height)
        if y1 <= y0:
            continue
        band_texts = [
            e.text_preview for e in elements
            if e.text_preview and e.y < y1 and (e.y + e.height) > y0
        ]
        tiles.append(GridTile(
            id=f"grid_{i:04d}", x=0, y=y0, width=page_width, height=y1 - y0,
            text_preview=" ".join(band_texts)[:_TEXT_PREVIEW_CHARS],
        ))
    return tiles
