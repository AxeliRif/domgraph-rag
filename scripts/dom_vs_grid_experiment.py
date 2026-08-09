"""
Ablation (a) du papier (§5) : tuilage guidé par le DOM vs. deux baselines --
une grille pixel de hauteur fixe (à la PixelRAG) et un chunking de texte plat
(à la RAG standard, cf. §2 "Text-based document RAG") -- au même scorer non
appris (TF-IDF + cosinus, identique à hard_negative_mining.py /
evidence_controller.py) -- pour isoler l'effet des limites de tuiles
alignées sur le contenu, indépendamment de tout modèle lecteur.

La baseline "text" existe parce que le papier argumente contre l'extraction
de texte (§1, §2) sans jamais la mesurer directement : elle concatène le
texte de tous les éléments DOM de la même région de page que les tuiles
DOM/grille, puis le découpe en fenêtres de taille fixe (chunking RAG
standard, sans notion de limite d'élément) -- le strict pendant textuel de
la grille pixel, mêmes questions, même scorer, seule la représentation
(image vs. texte brut) diffère de la tuile DOM.

Réutilise le jeu de QA déjà construit (data/qa_dataset/qa_pairs.jsonl)
plutôt que d'en générer un nouveau, restreint aux pages tuilées comme une
seule unité de travail (pas de découpage en sections, cf. sectioning.py) --
un rendu frais reproduit alors directement toute la page.

Pour chaque page : un seul rendu Playwright frais, tuilé une fois par DOM
(src/tiling.py), une fois par grille fixe (scripts/grid_baseline.py), une
fois par chunks de texte plat. Pour chaque paire (question, réponse)
existante, la réponse attendue est recherchée telle quelle (sous-chaîne,
insensible à la casse) dans le texte de chaque tuile/chunk -- une paire dont
la réponse n'apparaît littéralement dans aucune unité d'une condition donnée
est exclue de cette condition (taux de rattachement rapporté), plutôt que
comptée comme un échec de rang.

Métrique de comparaison : rang percentile, $1 - (\\text{rang}-1)/(n_\\text{items}-1)$
(1.0 = meilleur, 0.0 = pire, 0.5 en espérance sous un classement aléatoire
quel que soit $n_\\text{items}$) -- invariant à la taille du pool par
construction, contrairement à MRR/chance (une correction par $H_n/n$ reste
une comparaison de ratios, sensible à la formulation) ou à Recall@k/MRR bruts
(mécaniquement gonflés par un pool plus petit, cf. la grille ~5x plus petite
que le DOM, ~40x plus petite que le texte plat).

Usage :
    python3 -m scripts.dom_vs_grid_experiment [--max-pages N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.grid_baseline import build_grid_tiles
from src.config import QA_PAIRS_PATH
from src.dom_extraction import extract_dom_elements_async
from src.tiling import build_tiles

RESULTS_PATH = Path("data/dom_vs_grid_results.json")

# qa_pairs.jsonl a été construit avec ce plafond (cf. scripts/build_dataset.py,
# --max-tiles-per-page par défaut) : seules les MAX_DOM_TILES_PER_PAGE
# premières tuiles DOM (ordre natif de build_tiles, pas compute_reading_order)
# de chaque page ont jamais été montrées au VLM générateur de QA. Un
# rattachement de réponse sur une tuile au-delà de ce plafond serait donc une
# fuite -- on restreint aussi la grille et les chunks de texte à la même
# portion verticale de page, pour comparer les trois conditions sur
# exactement le même contenu couvert.
MAX_DOM_TILES_PER_PAGE = 25

# Taille de chunk pour la baseline texte plat, en caractères -- même ordre de
# grandeur que la longueur de texte typique d'une tuile DOM (comparaison
# équitable de granularité), sans notion de limite d'élément : un chunk peut
# couper un paragraphe ou une ligne de tableau en plein milieu, exactement ce
# qu'un pipeline RAG texte standard fait (§2).
TEXT_CHUNK_CHARS = 400


@dataclass
class RankedItem:
    id: str
    text: str
    width: int = 0
    height: int = 0


def _rank(question: str, items: list[RankedItem]) -> list[int]:
    """Indices de `items`, triés par similarité TF-IDF/cosinus décroissante à `question`."""
    texts = [it.text or "" for it in items]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        matrix = vectorizer.fit_transform([*texts, question])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except ValueError:
        return list(range(len(items)))  # vocabulaire vide (textes vides) -> ordre arbitraire
    return sorted(range(len(items)), key=lambda i: sims[i], reverse=True)


def _contains_answer(answer: str, text: str) -> bool:
    needle = answer.strip().lower()
    return bool(needle) and needle in (text or "").lower()


def _percentile_rank(rank: int, n_items: int) -> float:
    """1.0 si `rank`=1 (meilleur), 0.0 si `rank`=n_items (pire), 0.5 en
    espérance sous un classement uniforme au hasard -- quel que soit
    n_items, contrairement à Recall@k/MRR bruts ou à un ratio MRR/hasard."""
    if n_items <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (n_items - 1)


def _build_text_chunks(elements, y_cutoff: float | None, chunk_size: int = TEXT_CHUNK_CHARS) -> list[RankedItem]:
    """Concatène le texte de tous les éléments DOM sous `y_cutoff` (même
    région que les tuiles DOM/grille) en un seul flux, puis le découpe en
    fenêtres de taille fixe -- le chunking RAG texte standard (§2), sans
    connaissance des limites d'éléments."""
    kept = [e for e in elements if y_cutoff is None or e.y < y_cutoff]
    full_text = " ".join(e.text_preview for e in kept if e.text_preview)
    if not full_text.strip():
        return []
    return [
        RankedItem(id=f"text_chunk_{i:04d}", text=full_text[i : i + chunk_size])
        for i in range(0, len(full_text), chunk_size)
    ]


async def evaluate_page(url: str, qa_pairs: list[dict]) -> dict:
    elements, screenshot = await extract_dom_elements_async(url, wait_until="load")

    dom_tiles = build_tiles(elements, screenshot)[:MAX_DOM_TILES_PER_PAGE]
    dom_items = [RankedItem(t.id, t.text_preview, t.width, t.height) for t in dom_tiles]

    y_cutoff = max((t.y + t.height for t in dom_tiles), default=None)

    all_grid_tiles = build_grid_tiles(elements, screenshot)
    grid_tiles = [t for t in all_grid_tiles if y_cutoff is None or t.y < y_cutoff]
    grid_items = [RankedItem(t.id, t.text_preview, t.width, t.height) for t in grid_tiles]

    text_items = _build_text_chunks(elements, y_cutoff)

    conditions = {"dom": dom_items, "grid": grid_items, "text": text_items}
    per_condition: dict[str, list[dict]] = {c: [] for c in conditions}
    n_unattributable: dict[str, int] = {c: 0 for c in conditions}

    for qa in qa_pairs:
        question, answer = qa["question"], qa["answer"]
        for cond, items in conditions.items():
            if not items:
                n_unattributable[cond] += 1
                continue
            order = _rank(question, items)
            correct_ranks = [
                rank for rank, idx in enumerate(order, start=1)
                if _contains_answer(answer, items[idx].text)
            ]
            if not correct_ranks:
                n_unattributable[cond] += 1
                continue
            best_rank = min(correct_ranks)
            per_condition[cond].append({
                "qa_id": qa["id"],
                "rank": best_rank,
                "n_items": len(items),
                "recall@1": best_rank <= 1,
                "recall@3": best_rank <= 3,
                "recall@5": best_rank <= 5,
                "percentile_rank": _percentile_rank(best_rank, len(items)),
            })

    return {
        "url": url,
        "n_qa_pairs": len(qa_pairs),
        "n_dom_tiles": len(dom_items),
        "n_grid_tiles": len(grid_items),
        "n_text_chunks": len(text_items),
        "dom_pixels_per_tile": (sum(t.width * t.height for t in dom_items) / len(dom_items)) if dom_items else 0.0,
        "grid_pixels_per_tile": (sum(t.width * t.height for t in grid_items) / len(grid_items)) if grid_items else 0.0,
        "n_unattributable": n_unattributable,
        "results": per_condition,
    }


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "recall@1": sum(r["recall@1"] for r in rows) / n,
        "recall@3": sum(r["recall@3"] for r in rows) / n,
        "recall@5": sum(r["recall@5"] for r in rows) / n,
        "percentile_rank": sum(r["percentile_rank"] for r in rows) / n,
    }


async def main_async(max_pages: int | None) -> None:
    qa_pairs = [json.loads(line) for line in QA_PAIRS_PATH.open(encoding="utf-8") if line.strip()]

    by_url: dict[str, list[dict]] = {}
    for qa in qa_pairs:
        if "__sec" in qa["page_slug"]:
            continue  # page découpée en sections -- hors périmètre (cf. docstring du module)
        by_url.setdefault(qa["page_url"], []).append(qa)

    urls = list(by_url.items())
    if max_pages is not None:
        urls = urls[:max_pages]

    all_page_results: list[dict] = []
    for i, (url, pairs) in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url} ({len(pairs)} paires QA)...", flush=True)
        try:
            page_result = await evaluate_page(url, pairs)
        except Exception as exc:  # noqa: BLE001 - une page individuelle : on continue sur les suivantes
            print(f"  ÉCHEC ({exc!r}), page ignorée", flush=True)
            continue
        all_page_results.append(page_result)
        print(
            f"  {page_result['n_dom_tiles']} tuiles DOM, {page_result['n_grid_tiles']} tuiles grille, "
            f"{page_result['n_text_chunks']} chunks texte, "
            f"non-rattachables: dom={page_result['n_unattributable']['dom']} "
            f"grid={page_result['n_unattributable']['grid']} text={page_result['n_unattributable']['text']}",
            flush=True,
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_page_results, f, indent=2)

    print(f"\n{'='*60}\nRésultats agrégés ({len(all_page_results)} pages) -- toutes paires rattachables\n{'='*60}")
    print(
        "ATTENTION : les pools ont des tailles très différentes (dom < grid < text,\n"
        "voir avg tiles/page ci-dessous) -- Recall@k brut n'est PAS directement comparable\n"
        "entre conditions sans corriger de cet effet (cf. percentile_rank, invariant à la\n"
        "taille du pool, et la comparaison appariée sur l'intersection ci-dessous)."
    )
    for cond in ("dom", "grid", "text"):
        rows = [r for pr in all_page_results for r in pr["results"][cond]]
        agg = _aggregate(rows)
        if agg["n"] == 0:
            print(f"{cond.upper()}: aucune paire rattachable")
            continue
        print(
            f"{cond.upper():5s} n={agg['n']:4d}  "
            f"Recall@1={agg['recall@1']:.3f}  Recall@3={agg['recall@3']:.3f}  "
            f"Recall@5={agg['recall@5']:.3f}  Percentile rank={agg['percentile_rank']:.3f}"
        )

    print(f"\n{'='*60}\nComparaison appariée -- paires rattachables dans LES TROIS conditions\n{'='*60}")
    by_id = {
        cond: {r["qa_id"]: r for pr in all_page_results for r in pr["results"][cond]}
        for cond in ("dom", "grid", "text")
    }
    common_ids = sorted(set(by_id["dom"]) & set(by_id["grid"]) & set(by_id["text"]))
    print(f"paires rattachables dans les trois conditions : {len(common_ids)} "
          f"(sur {len(by_id['dom'])} dom / {len(by_id['grid'])} grid / {len(by_id['text'])} text rattachables au total)")
    for cond in ("dom", "grid", "text"):
        rows = [by_id[cond][qid] for qid in common_ids]
        agg = _aggregate(rows)
        if agg["n"] == 0:
            continue
        print(
            f"{cond.upper():5s} n={agg['n']:4d}  "
            f"Recall@1={agg['recall@1']:.3f}  Recall@3={agg['recall@3']:.3f}  "
            f"Recall@5={agg['recall@5']:.3f}  Percentile rank={agg['percentile_rank']:.3f}"
        )

    if all_page_results:
        avg_dom_tiles = sum(pr["n_dom_tiles"] for pr in all_page_results) / len(all_page_results)
        avg_grid_tiles = sum(pr["n_grid_tiles"] for pr in all_page_results) / len(all_page_results)
        avg_text_chunks = sum(pr["n_text_chunks"] for pr in all_page_results) / len(all_page_results)
        avg_dom_px = sum(pr["dom_pixels_per_tile"] for pr in all_page_results) / len(all_page_results)
        avg_grid_px = sum(pr["grid_pixels_per_tile"] for pr in all_page_results) / len(all_page_results)
        total_dom_qa = sum(pr["n_qa_pairs"] for pr in all_page_results)
        print(f"\navg units/page   : DOM={avg_dom_tiles:.1f}   grid={avg_grid_tiles:.1f}   text={avg_text_chunks:.1f}")
        print(f"avg pixels/tile  : DOM={avg_dom_px:,.0f}   grid={avg_grid_px:,.0f}")
        print(f"total QA pairs considered: {total_dom_qa}")

    print(f"\nRésultats détaillés écrits dans {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args.max_pages))


if __name__ == "__main__":
    main()
