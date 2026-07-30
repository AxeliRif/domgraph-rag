"""
Ablation (a) du papier (§5, "Planned ablations") : tuilage guidé par le DOM
vs. une grille pixel de hauteur fixe (à la PixelRAG), au même scorer non
appris (TF-IDF + cosinus, identique à hard_negative_mining.py /
evidence_controller.py) -- pour isoler l'effet des limites de tuiles
alignées sur le contenu, indépendamment de tout modèle lecteur.

Réutilise le jeu de QA déjà construit (data/qa_dataset/qa_pairs.jsonl)
plutôt que d'en générer un nouveau, restreint aux pages tuilées comme une
seule unité de travail (pas de découpage en sections, cf. sectioning.py) --
un rendu frais reproduit alors directement toute la page.

Pour chaque page : un seul rendu Playwright frais, tuilé une fois par DOM
(src/tiling.py) et une fois par grille fixe (scripts/grid_baseline.py). Pour
chaque paire (question, réponse) existante, la réponse attendue est
recherchée telle quelle (sous-chaîne, insensible à la casse) dans le texte
de chaque tuile -- une paire dont la réponse n'apparaît littéralement dans
aucune tuile d'une condition donnée est exclue de cette condition (taux de
rattachement rapporté), plutôt que comptée comme un échec de rang.

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
# fuite -- on restreint aussi la grille à la même portion verticale de page,
# pour comparer les deux tuilages sur exactement le même contenu couvert.
MAX_DOM_TILES_PER_PAGE = 25


@dataclass
class RankedItem:
    id: str
    text: str
    width: int
    height: int


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


async def evaluate_page(url: str, qa_pairs: list[dict]) -> dict:
    elements, screenshot = await extract_dom_elements_async(url, wait_until="load")

    dom_tiles = build_tiles(elements, screenshot)[:MAX_DOM_TILES_PER_PAGE]
    dom_items = [RankedItem(t.id, t.text_preview, t.width, t.height) for t in dom_tiles]

    all_grid_tiles = build_grid_tiles(elements, screenshot)
    if dom_tiles:
        y_cutoff = max(t.y + t.height for t in dom_tiles)
        grid_tiles = [t for t in all_grid_tiles if t.y < y_cutoff]
    else:
        grid_tiles = all_grid_tiles
    grid_items = [RankedItem(t.id, t.text_preview, t.width, t.height) for t in grid_tiles]

    per_condition: dict[str, list[dict]] = {"dom": [], "grid": []}
    n_unattributable: dict[str, int] = {"dom": 0, "grid": 0}

    for qa in qa_pairs:
        question, answer = qa["question"], qa["answer"]
        for cond, items in (("dom", dom_items), ("grid", grid_items)):
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
                "mrr": 1.0 / best_rank,
                "chance_mrr": 1.0 / len(items),  # baseline attendu d'un classement aléatoire, même pool
            })

    return {
        "url": url,
        "n_qa_pairs": len(qa_pairs),
        "n_dom_tiles": len(dom_items),
        "n_grid_tiles": len(grid_items),
        "dom_pixels_per_tile": (sum(t.width * t.height for t in dom_items) / len(dom_items)) if dom_items else 0.0,
        "grid_pixels_per_tile": (sum(t.width * t.height for t in grid_items) / len(grid_items)) if grid_items else 0.0,
        "n_unattributable": n_unattributable,
        "results": per_condition,
    }


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    mean_chance_recall1 = sum(r["chance_mrr"] for r in rows) / n  # E[1/n_items] = P(hit@1 au hasard)
    mean_mrr = sum(r["mrr"] for r in rows) / n
    mean_chance_mrr = sum(r["chance_mrr"] for r in rows) / n
    return {
        "n": n,
        "recall@1": sum(r["recall@1"] for r in rows) / n,
        "recall@3": sum(r["recall@3"] for r in rows) / n,
        "recall@5": sum(r["recall@5"] for r in rows) / n,
        "mrr": mean_mrr,
        "chance_recall@1": mean_chance_recall1,
        "chance_mrr": mean_chance_mrr,
        "mrr_lift_over_chance": (mean_mrr / mean_chance_mrr) if mean_chance_mrr else float("nan"),
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
            f"non-rattachables: dom={page_result['n_unattributable']['dom']} "
            f"grid={page_result['n_unattributable']['grid']}",
            flush=True,
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_page_results, f, indent=2)

    print(f"\n{'='*60}\nRésultats agrégés ({len(all_page_results)} pages) -- toutes paires rattachables\n{'='*60}")
    print(
        "ATTENTION : la grille a un pool de candidats ~5x plus petit (voir avg tiles/page\n"
        "ci-dessous) -- Recall@k et MRR bruts ne sont PAS directement comparables entre\n"
        "conditions sans corriger de cet effet de taille de pool (cf. mrr_lift_over_chance,\n"
        "et la comparaison appariée sur l'intersection ci-dessous)."
    )
    for cond in ("dom", "grid"):
        rows = [r for pr in all_page_results for r in pr["results"][cond]]
        agg = _aggregate(rows)
        if agg["n"] == 0:
            print(f"{cond.upper()}: aucune paire rattachable")
            continue
        print(
            f"{cond.upper():5s} n={agg['n']:4d}  "
            f"Recall@1={agg['recall@1']:.3f}  Recall@3={agg['recall@3']:.3f}  "
            f"Recall@5={agg['recall@5']:.3f}  MRR={agg['mrr']:.3f}  "
            f"| hasard: Recall@1={agg['chance_recall@1']:.3f} MRR={agg['chance_mrr']:.3f}  "
            f"-> MRR / hasard = {agg['mrr_lift_over_chance']:.2f}x"
        )

    print(f"\n{'='*60}\nComparaison appariée -- paires rattachables dans LES DEUX conditions\n{'='*60}")
    dom_by_id = {r["qa_id"]: r for pr in all_page_results for r in pr["results"]["dom"]}
    grid_by_id = {r["qa_id"]: r for pr in all_page_results for r in pr["results"]["grid"]}
    common_ids = sorted(set(dom_by_id) & set(grid_by_id))
    print(f"paires rattachables dans les deux conditions : {len(common_ids)} "
          f"(sur {len(dom_by_id)} dom / {len(grid_by_id)} grid rattachables au total)")
    for cond, by_id in (("dom", dom_by_id), ("grid", grid_by_id)):
        rows = [by_id[qid] for qid in common_ids]
        agg = _aggregate(rows)
        if agg["n"] == 0:
            continue
        print(
            f"{cond.upper():5s} n={agg['n']:4d}  "
            f"Recall@1={agg['recall@1']:.3f}  Recall@3={agg['recall@3']:.3f}  "
            f"Recall@5={agg['recall@5']:.3f}  MRR={agg['mrr']:.3f}  "
            f"| MRR / hasard = {agg['mrr_lift_over_chance']:.2f}x"
        )

    if all_page_results:
        avg_dom_tiles = sum(pr["n_dom_tiles"] for pr in all_page_results) / len(all_page_results)
        avg_grid_tiles = sum(pr["n_grid_tiles"] for pr in all_page_results) / len(all_page_results)
        avg_dom_px = sum(pr["dom_pixels_per_tile"] for pr in all_page_results) / len(all_page_results)
        avg_grid_px = sum(pr["grid_pixels_per_tile"] for pr in all_page_results) / len(all_page_results)
        total_dom_qa = sum(pr["n_qa_pairs"] for pr in all_page_results)
        print(f"\navg tiles/page   : DOM={avg_dom_tiles:.1f}   grid={avg_grid_tiles:.1f}")
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
