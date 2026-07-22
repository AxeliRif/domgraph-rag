"""
Passage à l'échelle du dataset contrastif (Phase 2a/2b) sur plusieurs pages.

Jusqu'ici, `qa_generation.generate_qa_dataset` et `hard_negative_mining.mine_hard_negatives`
n'avaient été exercés que sur une seule page (cf. section 7-9 du notebook). Ce
script répète le même pipeline (DOM -> tuiles -> QA générée par le VLM ->
hard-negative mining) sur une liste de pages Wikipédia, pour obtenir un jeu de
données de taille suffisante pour un fine-tuning contrastif.

Deux réglages qui bornent le temps d'exécution (des articles Wikipédia longs
peuvent produire >150 tuiles ; le VLM est interrogé une fois par tuile) :
  --max-tiles-per-page : ne garde que les `N` premières tuiles (ordre de
                          lecture) de chaque *unité de travail* (une page, ou
                          une section pour une page découpée -- voir plus bas)
                          pour la génération de QA.
  --urls-file           : un fichier texte (une URL par ligne) pour remplacer
                          la liste par défaut ci-dessous.

Une page démesurément longue (cf. `src/sectioning.py` ; observé en pratique
sur "The Beatles", dont la capture pleine page a bloqué un run entier
plusieurs heures sur une seule tuile, sans aucune progression sauvegardée
pour cette page) n'est pas traitée d'un bloc : elle est découpée en sections
aux frontières de titres (h1/h2), chacune devenant sa propre "unité de
travail" avec son propre slug (`<page_slug>__sec<N>`), génération QA, mining,
et surtout son propre check-point indépendant -- un échec sur une section ne
fait perdre que cette section, pas les précédentes déjà traitées pour la même
page.

`qa_pairs.jsonl` et `tiles_manifest.jsonl` sont complétés unité par unité
(mode append, comme le fait déjà `qa_generation.py`). `contrastive_examples.jsonl`
est réécrit en entier à chaque unité (c'est le comportement existant de
`save_contrastive_examples`) mais avec la liste cumulée depuis le début du
run -- donc jamais avec moins que ce qui a déjà été généré, y compris en cas
d'interruption en cours de route.

Usage :
    python3 -m scripts.build_dataset
    python3 -m scripts.build_dataset --max-tiles-per-page 15 --max-pages 5
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass

from src.config import CONTRASTIVE_EXAMPLES_PATH
from src.dom_extraction import extract_dom_elements_async
from src.hard_negative_mining import load_contrastive_examples, mine_hard_negatives, save_contrastive_examples
from src.qa_generation import generate_qa_dataset
from src.sectioning import is_oversized_page, split_into_sections
from src.tiling import Tile, build_tiles
from src.vlm_client import VLMClient

# Articles Wikipédia de longueur moyenne (ni des stubs de quelques tuiles, ni
# des articles vedettes de 150+ tuiles), choisis pour leur diversité
# thématique -- utile au hard-negative mining (TF-IDF), qui a besoin de
# tuiles lexicalement variées au sein d'un même run pour être discriminant.
DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Photosynthesis",
    "https://en.wikipedia.org/wiki/Great_Barrier_Reef",
    "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
    "https://en.wikipedia.org/wiki/The_Beatles",
    "https://en.wikipedia.org/wiki/Mount_Everest",
    "https://en.wikipedia.org/wiki/Ada_Lovelace",
    "https://en.wikipedia.org/wiki/Solar_System",
    "https://en.wikipedia.org/wiki/Renaissance",
    "https://en.wikipedia.org/wiki/Great_Wall_of_China",
    "https://en.wikipedia.org/wiki/Isaac_Newton",
    "https://en.wikipedia.org/wiki/Antarctica",
    "https://en.wikipedia.org/wiki/Amazon_rainforest",
    "https://en.wikipedia.org/wiki/Industrial_Revolution",
    "https://en.wikipedia.org/wiki/DNA",
    "https://en.wikipedia.org/wiki/Eiffel_Tower",
    "https://en.wikipedia.org/wiki/Marie_Curie",
    "https://en.wikipedia.org/wiki/Great_Depression",
]


def slug_for_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


@dataclass
class PageUnit:
    """Une unité de travail pour `main_async` : soit une page entière, soit
    une section d'une page trop grande pour être traitée d'un bloc (cf.
    `src/sectioning.py`) -- chacune a son propre slug, traitée et
    check-pointée indépendamment, pour qu'un échec en cours de route sur une
    grande page ne fasse perdre que la section en cours plutôt que tout le
    travail déjà fait sur cette page (cf. incident The_Beatles, module docstring)."""
    slug: str
    url: str
    tiles: list[Tile]


async def build_page_units(url: str, page_slug: str, max_tiles_per_page: int | None) -> list[PageUnit]:
    elements, screenshot = await extract_dom_elements_async(url, wait_until="load")

    if not is_oversized_page(elements):
        tiles = build_tiles(elements, screenshot)
        if max_tiles_per_page is not None:
            tiles = tiles[:max_tiles_per_page]
        return [PageUnit(slug=page_slug, url=url, tiles=tiles)]

    units: list[PageUnit] = []
    for section in split_into_sections(elements):
        tiles = build_tiles(section.elements, screenshot)
        if max_tiles_per_page is not None:
            tiles = tiles[:max_tiles_per_page]
        section_url = url if section.index == 0 else f"{url}#section-{section.index}"
        units.append(PageUnit(slug=f"{page_slug}__sec{section.index}", url=section_url, tiles=tiles))
    return units


# Un run de plusieurs heures peut traverser une coupure réseau transitoire
# (observé : une panne WiFi a fait échouer 8 pages d'affilée en quelques
# secondes, chacune immédiatement après l'autre, sans jamais laisser la
# connexion le temps de revenir). Réessayer avec un backoff croissant avant
# d'abandonner une page laisse la coupure le temps de se résorber plutôt que
# de brûler le reste de la liste d'URLs quasi instantanément.
RETRY_DELAYS_S = [30, 60, 120, 240]


async def build_page_units_with_retry(url: str, max_tiles_per_page: int | None, page_slug: str) -> list[PageUnit]:
    for attempt, delay in enumerate([0, *RETRY_DELAYS_S], start=1):
        if delay:
            print(f"  {page_slug} — nouvelle tentative dans {delay}s (essai {attempt}/{len(RETRY_DELAYS_S) + 1})...", flush=True)
            await asyncio.sleep(delay)
        try:
            return await build_page_units(url, page_slug, max_tiles_per_page)
        except Exception as exc:  # noqa: BLE001 - on retente tant qu'il reste des essais
            last_exc = exc
            print(f"  {page_slug} — échec essai {attempt}/{len(RETRY_DELAYS_S) + 1} ({exc!r})", flush=True)
    raise last_exc


async def main_async(urls: list[str], max_tiles_per_page: int | None, judge_false_negatives: bool) -> None:
    all_examples = load_contrastive_examples(CONTRASTIVE_EXAMPLES_PATH)  # ne pas perdre les pages déjà traitées
    seen_slugs = {ex.page_slug for ex in all_examples}

    # Un seul VLMClient réutilisé pour la génération QA *et*, si activé, le
    # juge de faux négatifs -- pas un par appel, pour ne pas payer deux fois
    # le coût de démarrage du backend (ex. chargement du modèle en "transformers").
    vlm_client = VLMClient()
    judge_client = vlm_client if judge_false_negatives else None

    total_qa = 0
    for i, url in enumerate(urls, 1):
        page_slug = slug_for_url(url)
        if page_slug in seen_slugs:
            print(f"[{i}/{len(urls)}] {page_slug} — déjà présent dans contrastive_examples.jsonl, skip", flush=True)
            continue

        t0 = time.time()
        print(f"[{i}/{len(urls)}] {page_slug} — extraction + tuilage...", flush=True)
        try:
            units = await build_page_units_with_retry(url, max_tiles_per_page, page_slug)
        except Exception as exc:  # noqa: BLE001 - page individuelle : on continue le run sur les autres
            print(f"[{i}/{len(urls)}] {page_slug} — ÉCHEC extraction après {len(RETRY_DELAYS_S) + 1} essais ({exc!r}), skip", flush=True)
            continue

        if len(units) > 1:
            print(
                f"[{i}/{len(urls)}] {page_slug} — page trop grande, découpée en {len(units)} "
                "sections traitées indépendamment",
                flush=True,
            )

        page_qa = 0
        for unit in units:
            # Une page découpée en sections n'a jamais elle-même de slug dans
            # `seen_slugs` (seuls les slugs de sections y sont ajoutés, cf.
            # plus bas) : ce test au niveau unité rattrape la reprise d'un
            # run interrompu au milieu d'une grande page, en ne refaisant que
            # les sections encore manquantes.
            if unit.slug in seen_slugs:
                print(f"  {unit.slug} — déjà présent dans contrastive_examples.jsonl, skip", flush=True)
                continue

            print(f"  {unit.slug} — {len(unit.tiles)} tuiles, génération QA (VLM)...", flush=True)
            try:
                qa_pairs = generate_qa_dataset(unit.tiles, page_url=unit.url, page_slug=unit.slug, client=vlm_client)
            except Exception as exc:  # noqa: BLE001 - une unité individuelle : on continue sur les suivantes
                print(f"  {unit.slug} — ÉCHEC génération QA ({exc!r}), skip", flush=True)
                continue

            if judge_client is not None:
                print(f"  {unit.slug} — mining négatifs (VLM juge activé, un appel/candidat)...", flush=True)
            examples = mine_hard_negatives(qa_pairs, unit.tiles, vlm_client=judge_client)
            all_examples.extend(examples)
            seen_slugs.add(unit.slug)
            save_contrastive_examples(all_examples)

            page_qa += len(qa_pairs)
            total_qa += len(qa_pairs)
            print(
                f"  {unit.slug} — {len(qa_pairs)} QA / {len(unit.tiles)} tuiles "
                f"(cumul total: {total_qa} QA, {len(all_examples)} exemples contrastifs)",
                flush=True,
            )

        print(
            f"[{i}/{len(urls)}] {page_slug} — terminé en {time.time() - t0:.0f}s "
            f"({page_qa} QA sur cette page, cumul total {total_qa} QA)",
            flush=True,
        )

    print(f"\nTerminé : {len(all_examples)} exemples contrastifs dans {CONTRASTIVE_EXAMPLES_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère le dataset contrastif (Phase 2a/2b) sur plusieurs pages.")
    parser.add_argument("--urls-file", type=str, default=None, help="Fichier texte, une URL Wikipédia par ligne.")
    parser.add_argument("--max-pages", type=int, default=None, help="Limite le nombre de pages traitées.")
    parser.add_argument(
        "--max-tiles-per-page", type=int, default=25,
        help=(
            "Ne garde que les N premières tuiles (ordre de lecture) de chaque unité de "
            "travail (une page, ou une section pour une page démesurément longue -- "
            "cf. module docstring). 0 = pas de limite."
        ),
    )
    parser.add_argument(
        "--judge-false-negatives", action="store_true",
        help=(
            "Ajoute un second filtre anti-faux-négatif au hard-negative mining : le VLM "
            "'regarde' chaque candidat et juge s'il peut y répondre (attrape les faux "
            "négatifs visuels/reformulés que le filtre par sous-chaîne seul laisse passer). "
            "Coûteux : un appel VLM par candidat non déjà écarté par le filtre lexical."
        ),
    )
    args = parser.parse_args()

    if args.urls_file:
        with open(args.urls_file, encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        urls = DEFAULT_URLS

    if args.max_pages:
        urls = urls[: args.max_pages]

    max_tiles = args.max_tiles_per_page if args.max_tiles_per_page > 0 else None

    try:
        asyncio.run(main_async(urls, max_tiles, args.judge_false_negatives))
    except KeyboardInterrupt:
        print("\nInterrompu — les pages déjà traitées sont conservées (qa_pairs.jsonl, contrastive_examples.jsonl).")
        sys.exit(130)


if __name__ == "__main__":
    main()
