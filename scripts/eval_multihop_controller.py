"""
Évalue le contrôleur d'évidence (src/evidence_controller.py) sur les questions
multi-hop de data/qa_dataset/multihop_examples.jsonl (construites depuis
HotpotQA par scripts/build_multihop_dataset.py) : les seules données dont les
tuiles positives couvrent VRAIMENT deux pages, contrairement aux QA
synthétiques single-hop de qa_generation.py qui ne peuvent pas exercer le
graphe par construction (cf. paper, Limitations, "Single-hop self-generated
data cannot exercise the graph").

Pour chaque question : reconstruit un graphe combiné des deux pages
(namespacé comme corpus_builder.build_corpus_graph, mais sans lien réel requis
entre elles -- le seeding du contrôleur score déjà tous les noeuds du graphe
combiné, quelle que soit leur page), fait tourner le contrôleur budgété, et
vérifie si son évidence "opened" couvre au moins une tuile positive sur
CHACUNE des deux pages -- un succès "cross-page" qu'une sélection single-page
ne peut pas obtenir par construction. Compare à un baseline top-k statique sur
les MÊMES scores TF-IDF (lues sur `relevance_score`, posé par
run_evidence_controller), pour isoler l'effet de la politique du contrôleur
(best-first sous budget, propagation le long de reading_order) de celui du
scorer lui-même -- la comparaison que Method (§3.4) annonce sans encore la
chiffrer ("what distinguishes it from static top-k retrieval over the same
scores").

Ne nécessite ni VLM ni GPU (use_vlm=False) : uniquement rendu Playwright +
scoring lexical, comme l'ablation (a) (scripts/dom_vs_grid_experiment.py).

Usage :
    python3 -m scripts.eval_multihop_controller
    python3 -m scripts.eval_multihop_controller --budget 5 --seed-k 3 --max-questions 30
"""
from __future__ import annotations

import argparse
import asyncio
import json

import networkx as nx

from src.config import QA_DATASET_DIR
from src.corpus_builder import build_page_graph
from src.evidence_controller import EvidenceControllerConfig, run_evidence_controller

MULTIHOP_EXAMPLES_PATH = QA_DATASET_DIR / "multihop_examples.jsonl"
CONTROLLER_EVAL_PATH = QA_DATASET_DIR / "multihop_controller_eval.json"


def _url_for_slug(slug: str) -> str:
    return f"https://en.wikipedia.org/wiki/{slug}"


def _namespaced(graph: nx.MultiDiGraph, prefix: str) -> nx.MultiDiGraph:
    return nx.relabel_nodes(graph, {n: f"{prefix}::{n}" for n in graph.nodes}, copy=True)


async def _combined_graph(slugs: list[str], cache: dict[str, nx.MultiDiGraph]) -> nx.MultiDiGraph:
    """Graphe union des pages de `slugs`, chacune rendue au plus une fois par
    run grâce à `cache` -- des articles "pont" reviennent d'une question à
    l'autre dans HotpotQA."""
    combined = nx.MultiDiGraph()
    for slug in slugs:
        if slug not in cache:
            page_graph = await build_page_graph(_url_for_slug(slug))
            cache[slug] = _namespaced(page_graph, slug)
        combined = nx.compose(combined, cache[slug])
    return combined


def _static_topk_opened(graph: nx.MultiDiGraph, k: int) -> set[str]:
    """Baseline : les k noeuds au score le plus haut, tous globalement --
    aucune notion de budget dépensé au fil de l'exploration ni de propagation
    le long de reading_order, juste un classement figé sur les mêmes scores
    que le contrôleur (posés en `relevance_score` par run_evidence_controller)."""
    scored = [
        (n, d.get("relevance_score", 0.0))
        for n, d in graph.nodes(data=True) if d.get("type") == "element"
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {n for n, _ in scored[:k]}


async def evaluate_example(
    example: dict, budget: int, seed_k: int, page_graph_cache: dict[str, nx.MultiDiGraph],
) -> dict | None:
    slugs = sorted({slug for slug, _ in example["positive_tiles"]})
    if len(slugs) < 2:
        return None  # positifs concentrés sur une seule page -- pas un test cross-page

    try:
        graph = await _combined_graph(slugs, page_graph_cache)
    except Exception as exc:  # noqa: BLE001 - une question individuelle : on continue sur les suivantes
        return {"id": example["id"], "error": repr(exc)}

    positive_nodes_by_page = {
        slug: {f"{slug}::{tile_id}" for s, tile_id in example["positive_tiles"] if s == slug}
        for slug in slugs
    }
    # Une tuile positive absente du graphe re-rendu (article changé depuis la
    # construction du dataset multi-hop) ne doit jamais compter comme
    # "couverte" -- filtré implicitement puisque son id n'existera simplement
    # pas parmi les noeuds "opened"/top-k, jamais un faux positif.

    run_evidence_controller(
        graph, tiles=[], query=example["question"],
        config=EvidenceControllerConfig(budget=budget, top_k_seed=seed_k),
    )
    controller_opened = {
        n for n, d in graph.nodes(data=True) if d.get("type") == "element" and d.get("state") == "opened"
    }
    static_opened = _static_topk_opened(graph, budget)

    def _pages_covered(opened: set[str]) -> int:
        return sum(1 for slug in slugs if positive_nodes_by_page[slug] & opened)

    return {
        "id": example["id"],
        "n_pages": len(slugs),
        "controller_pages_covered": _pages_covered(controller_opened),
        "static_pages_covered": _pages_covered(static_opened),
        "controller_opened": len(controller_opened),
        "static_opened": len(static_opened),
    }


async def main_async(budget: int, seed_k: int, max_questions: int | None) -> None:
    if not MULTIHOP_EXAMPLES_PATH.exists():
        raise SystemExit(f"{MULTIHOP_EXAMPLES_PATH} introuvable -- lance d'abord scripts/build_multihop_dataset.py")

    examples = [json.loads(line) for line in MULTIHOP_EXAMPLES_PATH.open(encoding="utf-8") if line.strip()]
    if max_questions is not None:
        examples = examples[:max_questions]

    page_graph_cache: dict[str, nx.MultiDiGraph] = {}
    results: list[dict] = []
    for i, ex in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] {ex['id']}...", flush=True)
        r = await evaluate_example(ex, budget, seed_k, page_graph_cache)
        if r is None:
            print("  skip -- positifs sur une seule page", flush=True)
            continue
        if "error" in r:
            print(f"  ÉCHEC ({r['error']}), skip", flush=True)
            continue
        results.append(r)
        print(
            f"  contrôleur : {r['controller_pages_covered']}/{r['n_pages']} pages couvertes "
            f"({r['controller_opened']} tuiles ouvertes) | "
            f"top-k statique : {r['static_pages_covered']}/{r['n_pages']}",
            flush=True,
        )

    n = len(results)
    if n == 0:
        print("Aucun résultat exploitable.")
        return

    controller_full = sum(1 for r in results if r["controller_pages_covered"] == r["n_pages"])
    static_full = sum(1 for r in results if r["static_pages_covered"] == r["n_pages"])
    print(f"\n{'=' * 60}\n{n} questions cross-page évaluées (budget={budget}, seed_k={seed_k})\n{'=' * 60}")
    print(f"Contrôleur      : {controller_full}/{n} ({100 * controller_full / n:.1f}%) -- les DEUX pages couvertes")
    print(f"Top-k statique  : {static_full}/{n} ({100 * static_full / n:.1f}%) -- mêmes scores, pas de politique de contrôleur")

    CONTROLLER_EVAL_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nRésultats détaillés écrits dans {CONTROLLER_EVAL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=5, help="Budget d'ouverture du contrôleur (cf. EvidenceControllerConfig).")
    parser.add_argument("--seed-k", type=int, default=3, help="Nombre de noeuds activés au seed (cf. EvidenceControllerConfig).")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args.budget, args.seed_k, args.max_questions))


if __name__ == "__main__":
    main()
