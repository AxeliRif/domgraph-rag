"""
Test de bout en bout du pipeline Phase 1, sur une page locale (pas besoin
d'accès internet). Sert de garde-fou avant de pointer le pipeline vers une
vraie page Wikipedia.

Usage : python3 tests/test_pipeline_smoke.py   (depuis la racine du projet)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dom_extraction import extract_dom_elements_async  # noqa: E402
from src.graph_builder import build_graph, graph_to_xml  # noqa: E402
from src.tiling import build_tiles  # noqa: E402

FIXTURE_URL = (PROJECT_ROOT / "tests" / "fixtures" / "sample_page.html").as_uri()


async def main() -> None:
    print(f"[1/4] Extraction DOM depuis {FIXTURE_URL}")
    elements, screenshot = await extract_dom_elements_async(FIXTURE_URL, wait_until="load")
    assert len(elements) > 0, "Aucun élément extrait -> vérifier ELEMENT_TAGS / le fixture"
    tags_found = sorted({e.tag for e in elements})
    print(f"      {len(elements)} éléments extraits, balises trouvées : {tags_found}")
    # le nav/footer doivent avoir été retirés -> aucun élément ne devrait s'y trouver
    assert len(screenshot) > 0, "Capture d'écran vide"

    print("[2/4] Construction des tuiles (tuilage adaptatif + patching)")
    tiles = build_tiles(elements, screenshot)
    assert len(tiles) > 0
    heights = [t.height for t in tiles]
    print(f"      {len(tiles)} tuiles, hauteur min/max = {min(heights)}/{max(heights)} px")
    assert max(heights) <= 1024 + 1, "Une tuile dépasse MAX_TILE_HEIGHT -> le patching a un bug"
    ul_tiles = [t for t in tiles if t.dom_tag == "ul"]
    print(f"      la longue <ul> a été découpée en {len(ul_tiles)} sous-tuile(s) (patching)")
    assert len(ul_tiles) >= 2, "La liste longue aurait dû être découpée en plusieurs tuiles (patching)"

    print("[3/4] Construction du graphe (page + éléments, ordre de lecture)")
    graph = build_graph(tiles, page_url=FIXTURE_URL, page_title="Page de test")
    n_page_nodes = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "page")
    n_element_nodes = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "element")
    n_contains = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "contains")
    n_reading_order = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "reading_order")
    n_layout_adjacency = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "layout_adjacency")
    n_section_hierarchy = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "section_hierarchy")
    print(f"      noeuds page={n_page_nodes}, éléments={n_element_nodes}")
    print(f"      arêtes contains={n_contains}, reading_order={n_reading_order}, "
          f"layout_adjacency={n_layout_adjacency}, section_hierarchy={n_section_hierarchy}")
    assert n_page_nodes == 1
    assert n_element_nodes == len(tiles)
    assert n_contains == len(tiles)
    assert n_reading_order == len(tiles) - 1
    # layout_adjacency et reading_order peuvent relier la même paire de tuiles
    # (mise en page mono-colonne) : sur un DiGraph simple, la seconde écraserait
    # la première -> le graphe doit être un MultiDiGraph pour les faire coexister.
    assert n_layout_adjacency > 0, "Aucune arête layout_adjacency -> vérifier compute_layout_adjacency"
    assert n_reading_order > 0, "reading_order a disparu -> vérifier que le graphe est bien un MultiDiGraph"
    assert n_section_hierarchy > 0, "Aucune arête section_hierarchy -> le fixture contient pourtant des titres"

    print("[4/4] Export XML du graphe")
    xml_str = graph_to_xml(graph)
    assert "<page" in xml_str and "<element" in xml_str
    print(f"      XML généré ({len(xml_str)} caractères), aperçu :")
    print("      " + xml_str.splitlines()[1][:120])

    print("\nOK — le pipeline Phase 1 fonctionne de bout en bout sur le fixture local.")


if __name__ == "__main__":
    asyncio.run(main())
