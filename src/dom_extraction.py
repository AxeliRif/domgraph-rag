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

from .config import (
    DEVICE_SCALE_FACTOR,
    ELEMENT_TAGS,
    NOISE_SELECTORS,
    RENDER_WIDTH,
    TEXT_FLOW_TAGS,
)


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
    text_flow_tags: set[str] | None = None,
    wait_until: str = "networkidle",
    device_scale_factor: float = DEVICE_SCALE_FACTOR,
) -> tuple[list[DOMElement], bytes]:
    """
    Charge `url` dans Chromium headless, retire le bruit d'interface, puis extrait
    les coordonnées spatiales absolues (x, y, w, h) de chaque élément pertinent.

    Retourne (liste d'éléments, capture d'écran PNG de la page complète).
    `url` peut être une URL http(s) réelle ou un chemin local `file:///...`.
    """
    element_tags = element_tags if element_tags is not None else ELEMENT_TAGS
    noise_selectors = noise_selectors if noise_selectors is not None else NOISE_SELECTORS
    text_flow_tags = text_flow_tags if text_flow_tags is not None else TEXT_FLOW_TAGS

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page(
                viewport={"width": render_width, "height": 1000},
                device_scale_factor=device_scale_factor,
            )
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

            # La capture pleine page fait défiler tout le document -- ce qui peut
            # déclencher du lazy-loading (images/infobox chargées via
            # IntersectionObserver) et donc décaler la mise en page. On capture
            # donc AVANT de mesurer, puis on laisse le réseau se stabiliser une
            # seconde fois : mesurer avant la capture risquerait de renvoyer des
            # coordonnées qui ne correspondent plus à l'image une fois ce
            # chargement déclenché.
            screenshot = await page.screenshot(full_page=True)
            await page.wait_for_load_state("networkidle")

            # Extraction géométrique pure : bounding boxes absolues (page complète, pas juste le viewport)
            raw_elements = await page.evaluate(
                """({tags, textFlowTags}) => {
                    const results = [];
                    const selector = tags.join(',');

                    // Un élément de texte "coulant" (ex. <p>, ou un <ol> de
                    // références) a par défaut une boîte bloc qui s'étend sur toute
                    // la largeur du conteneur, même si le texte s'enroule autour d'un
                    // élément flottant voisin (ex. un infobox Wikipédia) et n'occupe
                    // donc réellement qu'une partie de cette largeur. On mesure donc
                    // uniquement les rectangles de lignes des NOEUDS TEXTE descendants
                    // (jamais d'un Range sur l'élément entier : un <li> à l'intérieur
                    // d'un <ol>/<ul> est lui-même un élément de type bloc, et un Range
                    // qui l'engloberait renverrait AUSSI son propre rectangle de boîte
                    // pleine largeur en plus des rectangles de ses lignes de texte,
                    // ce qui fausserait l'union vers la largeur du conteneur -- un
                    // noeud texte, lui, n'a jamais de boîte propre, donc ses
                    // rectangles ne peuvent être que des lignes réellement rendues).
                    function tightTextBBox(el) {
                        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
                            acceptNode: (node) => (node.nodeValue && node.nodeValue.trim().length > 0)
                                ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP,
                        });
                        const rects = [];
                        let node;
                        while ((node = walker.nextNode())) {
                            const range = document.createRange();
                            range.selectNodeContents(node);
                            for (const r of range.getClientRects()) {
                                if (r.width > 0 && r.height > 0) rects.push(r);
                            }
                        }
                        if (rects.length === 0) return null;
                        const x0 = Math.min(...rects.map(r => r.x));
                        const y0 = Math.min(...rects.map(r => r.y));
                        const x1 = Math.max(...rects.map(r => r.x + r.width));
                        const y1 = Math.max(...rects.map(r => r.y + r.height));
                        return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
                    }

                    document.querySelectorAll(selector).forEach((el) => {
                        const r = el.getBoundingClientRect();
                        if (r.width > 2 && r.height > 2) {
                            const tag = el.tagName.toLowerCase();
                            let box = r;
                            if (textFlowTags.includes(tag)) {
                                const tight = tightTextBBox(el);
                                if (tight && tight.width > 2 && tight.height > 2) {
                                    box = tight;
                                }
                            }

                            let depth = 0, node = el;
                            while (node.parentElement) { depth++; node = node.parentElement; }

                            // Liens hypertextes : uniquement pour les <p> (texte de
                            // l'article), jamais pour un <table> ni une liste de
                            // références en bas de page.
                            const links = [];
                            if (tag === 'p') {
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
                                tag,
                                x: box.x + window.scrollX,
                                y: box.y + window.scrollY,
                                width: box.width,
                                height: box.height,
                                text: (el.innerText || '').trim().slice(0, 200),
                                depth,
                                links,
                            });
                        }
                    });
                    return results;
                }""",
                {"tags": element_tags, "textFlowTags": list(text_flow_tags)},
            )
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


def extract_dom_elements(
    url: str,
    render_width: int = RENDER_WIDTH,
    element_tags: list[str] | None = None,
    noise_selectors: list[str] | None = None,
    text_flow_tags: set[str] | None = None,
    wait_until: str = "networkidle",
    device_scale_factor: float = DEVICE_SCALE_FACTOR,
) -> tuple[list[DOMElement], bytes]:
    """Wrapper synchrone pratique pour un script classique (hors notebook) --
    transmet tels quels tous les paramètres de `extract_dom_elements_async`."""
    return asyncio.run(extract_dom_elements_async(
        url, render_width=render_width, element_tags=element_tags, noise_selectors=noise_selectors,
        text_flow_tags=text_flow_tags, wait_until=wait_until, device_scale_factor=device_scale_factor,
    ))
