"""
Phase 1a/1b — Initialisation de l'environnement + extraction géométrique pure.

Utilise l'API *asynchrone* de Playwright plutôt que l'API synchrone : l'API
synchrone plante dans un notebook Jupyter ("Sync API inside asyncio loop"),
car Jupyter fait déjà tourner sa propre boucle asyncio. Le notebook peut donc
simplement écrire `await extract_dom_elements_async(url)` dans une cellule.
Un wrapper synchrone est fourni pour un usage en script classique (hors notebook).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from playwright.async_api import async_playwright

from .config import ELEMENT_TAGS, NOISE_SELECTORS, RENDER_WIDTH


@dataclass
class Link:
    """Un lien hypertexte trouvé dans un élément de texte (<p> uniquement)."""
    href: str  # URL absolue, résolue via document.baseURI
    anchor_text: str


@dataclass
class DOMElement:
    """Un élément du DOM avec ses coordonnées spatiales absolues (page complète)."""
    id: str
    tag: str
    x: float
    y: float
    width: float
    height: float
    text_preview: str
    depth: int  # profondeur dans l'arbre DOM (utile pour des heuristiques futures)
    links: list[Link] = field(default_factory=list)  # uniquement peuplé pour tag == "p"


async def extract_dom_elements_async(
    url: str,
    render_width: int = RENDER_WIDTH,
    element_tags: list[str] | None = None,
    noise_selectors: list[str] | None = None,
    wait_until: str = "networkidle",
) -> tuple[list[DOMElement], bytes]:
    """
    Charge `url` dans Chromium headless, retire le bruit d'interface, puis extrait
    les coordonnées spatiales absolues (x, y, w, h) de chaque élément pertinent.

    Retourne (liste d'éléments, capture d'écran PNG de la page complète).
    `url` peut être une URL http(s) réelle ou un chemin local `file:///...`.
    """
    element_tags = element_tags or ELEMENT_TAGS
    noise_selectors = noise_selectors or NOISE_SELECTORS

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": render_width, "height": 1000})
            await page.goto(url, wait_until=wait_until)

            # Retrait des éléments d'interface inutiles avant capture (cf. slide PixelRAG)
            await page.evaluate(
                """(selectors) => {
                    selectors.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => el.remove());
                    });
                }""",
                noise_selectors,
            )

            # Extraction géométrique pure : bounding boxes absolues (page complète, pas juste le viewport)
            raw_elements = await page.evaluate(
                """(tags) => {
                    const results = [];
                    const selector = tags.join(',');
                    document.querySelectorAll(selector).forEach((el) => {
                        const r = el.getBoundingClientRect();
                        if (r.width > 2 && r.height > 2) {
                            let depth = 0, node = el;
                            while (node.parentElement) { depth++; node = node.parentElement; }

                            // Liens hypertextes : uniquement pour les <p> (texte de
                            // l'article), jamais pour un <table> ni une liste de
                            // références en bas de page.
                            const links = [];
                            if (el.tagName.toLowerCase() === 'p') {
                                el.querySelectorAll('a[href]').forEach((a) => {
                                    const href = a.getAttribute('href');
                                    if (!href || href.startsWith('#')) return;
                                    let abs;
                                    try {
                                        abs = new URL(href, document.baseURI).href;
                                    } catch (e) {
                                        return;
                                    }
                                    if (!abs.startsWith('http://') && !abs.startsWith('https://')) return;
                                    links.push({
                                        href: abs,
                                        anchor_text: (a.innerText || '').trim().slice(0, 100),
                                    });
                                });
                            }

                            results.push({
                                tag: el.tagName.toLowerCase(),
                                x: r.x + window.scrollX,
                                y: r.y + window.scrollY,
                                width: r.width,
                                height: r.height,
                                text: (el.innerText || '').trim().slice(0, 200),
                                depth,
                                links,
                            });
                        }
                    });
                    return results;
                }""",
                element_tags,
            )

            screenshot = await page.screenshot(full_page=True)
        finally:
            await browser.close()

    elements = [
        DOMElement(
            id=f"el_{i:04d}",
            tag=e["tag"],
            x=e["x"],
            y=e["y"],
            width=e["width"],
            height=e["height"],
            text_preview=e["text"],
            depth=e["depth"],
            links=[Link(href=l["href"], anchor_text=l["anchor_text"]) for l in e["links"]],
        )
        for i, e in enumerate(raw_elements)
    ]
    return elements, screenshot


def extract_dom_elements(url: str, render_width: int = RENDER_WIDTH) -> tuple[list[DOMElement], bytes]:
    """Wrapper synchrone pratique pour un script classique (hors notebook)."""
    return asyncio.run(extract_dom_elements_async(url, render_width))
