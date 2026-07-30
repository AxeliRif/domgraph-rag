"""
Test que l'extraction DOM mesure la largeur réellement occupée par le texte
des balises "coulantes" (h1-h4, p, blockquote) plutôt que la largeur de leur
boîte bloc entière (cf. TEXT_FLOW_TAGS, config.py) -- sans ça, un
paragraphe/titre à côté d'un élément flottant (ex. l'infobox d'un article
Wikipédia) mesure toute la largeur du conteneur même si son texte s'arrête
avant, et sa tuile finit par inclure les pixels de l'élément flottant voisin,
dupliqués dans sa propre tuile (bug rapporté : "des parties de la page web
étaient prises alors que ça fait partie d'une autre tuile en soi").

Nécessite Playwright + Chromium (comme test_pipeline_smoke.py).
Usage : python3 -m pytest tests/test_dom_extraction.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dom_extraction import extract_dom_elements_async  # noqa: E402
from src.tiling import build_tiles  # noqa: E402

FIXTURE_URL = (PROJECT_ROOT / "tests" / "fixtures" / "floated_infobox_page.html").as_uri()


def test_text_bbox_stops_before_floated_sibling():
    """Le <h2> et le <p> partagent visuellement la même ligne que l'infobox
    flottant (float: right) -- leur largeur mesurée doit s'arrêter avant sa
    bordure gauche, pas s'étendre sur toute la largeur du conteneur."""
    elements, _ = asyncio.run(extract_dom_elements_async(FIXTURE_URL, wait_until="load"))

    heading = next(e for e in elements if e.tag == "h2")
    paragraph = next(e for e in elements if e.tag == "p")
    table = next(e for e in elements if e.tag == "table")

    assert heading.x + heading.width <= table.x
    assert paragraph.x + paragraph.width <= table.x


def test_list_bbox_stops_before_floated_sibling_despite_li_own_block_box():
    """Régression : la première version du fix mesurait un <ol>/<ul> via un
    Range sur l'élément entier, qui incluait AUSSI le rectangle de boîte
    bloc pleine largeur de chaque <li> (lui-même un élément, donc avec sa
    propre boîte CSS) en plus des rectangles de ses lignes de texte -- ça
    faussait l'union vers la largeur du conteneur, exactement comme avant le
    fix pour <p>. Mesurer uniquement les noeuds texte descendants (jamais
    l'élément <li> lui-même) évite ce piège."""
    elements, _ = asyncio.run(extract_dom_elements_async(FIXTURE_URL, wait_until="load"))

    ol = next(e for e in elements if e.tag == "ol")
    table = next(e for e in elements if e.tag == "table")

    assert ol.x + ol.width <= table.x


def test_build_tiles_keeps_floated_infobox_separate_from_text():
    """Conséquence bout en bout : le tuilage produit une tuile "table" et une
    (ou plusieurs) tuile(s) de texte qui ne se recouvrent pas, plutôt qu'une
    unique tuile fusionnée engloutissant l'infobox (régression constatée
    quand le chevauchement n'était corrigé qu'en fusionnant les éléments au
    lieu de mesurer la largeur réelle du texte)."""
    elements, screenshot = asyncio.run(extract_dom_elements_async(FIXTURE_URL, wait_until="load"))
    tiles = build_tiles(elements, screenshot)

    table_tiles = [t for t in tiles if t.dom_tag == "table"]
    text_tiles = [t for t in tiles if t.dom_tag != "table"]

    assert len(table_tiles) == 1
    assert len(text_tiles) >= 1
    for text_tile in text_tiles:
        assert text_tile.x + text_tile.width <= table_tiles[0].x, (
            "Une tuile de texte empiète sur la colonne de l'infobox -> régression du bug corrigé"
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
