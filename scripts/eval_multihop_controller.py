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
les MÊMES scores (posées en `relevance_score` par run_evidence_controller),
pour isoler l'effet de la politique du contrôleur (best-first sous budget,
propagation le long de reading_order) de celui du scorer lui-même -- la
comparaison que Method (§3.4) annonce sans encore la chiffrer ("what
distinguishes it from static top-k retrieval over the same scores").

Deux scorers de pertinence, choisis par --scorer :
  - "tfidf" (défaut) : le placeholder lexical de evidence_controller.py, ne
    nécessite ni VLM ni GPU (use_vlm=False), comme l'ablation (a)
    (scripts/dom_vs_grid_experiment.py).
  - "reader" : le lecteur VLM fine-tuné par contrastive LoRA (Phase 2,
    src/lora_finetune.py) -- embeddings (question, tuile) via PromptEOL,
    similarité cosinus, injectés dans run_evidence_controller via
    `precomputed_scores` (aucun réentraînement, seul le scorer change). Ferme
    la boucle entre §3 (le lecteur fine-tuné améliore le retrieval, ablation
    b) et §4/Table 4 (le contrôleur, jusqu'ici évalué avec le TF-IDF
    placeholder que §3 motive justement à remplacer).

`--seed-k` accepte plusieurs valeurs (comme dans le papier, k_seed=3 puis
k_seed=5) : les scores du lecteur, coûteux (un forward VLM par tuile), sont
calculés UNE SEULE fois par question et réutilisés d'une valeur de k_seed à
l'autre -- seule la politique (Algorithme 1) est rejouée sur un graphe frais
(reconstruit depuis le cache par page, pas re-rendu) à chaque valeur.

Usage :
    python3 -m scripts.eval_multihop_controller
    python3 -m scripts.eval_multihop_controller --budget 5 --seed-k 3 --max-questions 30
    python3 -m scripts.eval_multihop_controller --scorer reader --seed-k 3 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from urllib.parse import unquote

import networkx as nx

from src.config import QA_DATASET_DIR
from src.corpus_builder import build_page_graph
from src.dom_extraction import extract_dom_elements_async
from src.evidence_controller import EvidenceControllerConfig, run_evidence_controller
from src.graph_builder import build_graph
from src.tiling import Tile, build_tiles

MULTIHOP_EXAMPLES_PATH = QA_DATASET_DIR / "multihop_examples.jsonl"
CONTROLLER_EVAL_PATH = QA_DATASET_DIR / "multihop_controller_eval.json"


def _url_for_slug(slug: str) -> str:
    return f"https://en.wikipedia.org/wiki/{slug}"


def _namespaced(graph: nx.MultiDiGraph, prefix: str) -> nx.MultiDiGraph:
    return nx.relabel_nodes(graph, {n: f"{prefix}::{n}" for n in graph.nodes}, copy=True)


def _slug_from_wiki_url(url: str) -> str | None:
    marker = "/wiki/"
    idx = url.find(marker)
    if idx == -1:
        return None
    return unquote(url[idx + len(marker):]).split("#")[0]


def _rewire_cross_links(graph: nx.MultiDiGraph, slugs: list[str]) -> None:
    """Rebranche toute arête \\code{links_to} dont la cible est une AUTRE page
    de `slugs` sur son vrai point d'entrée (\\code{{slug}::page}), exactement
    comme `corpus_builder.build_corpus_graph` le fait pour un vrai crawl --
    sans quoi ces arêtes pendent sur un placeholder \\code{external_page}
    jamais résolu, et l'expansion cross-document du contrôleur
    (`evidence_controller._cross_document_neighbors`) n'a jamais rien à
    traverser sur ce jeu de données. Comparaison par slug d'URL Wikipedia
    (tolérante au pourcentage-encodage) ; un lien qui ne matche aucun des
    `slugs` (vers un troisième article) reste un placeholder inerte, jamais
    une erreur."""
    for node_id, data in list(graph.nodes(data=True)):
        if data.get("type") != "external_page":
            continue
        target_slug = _slug_from_wiki_url(data.get("url", ""))
        if target_slug not in slugs:
            continue
        entry_node = f"{target_slug}::page"
        if entry_node not in graph:
            continue  # page démesurée/découpée -- hors périmètre de ce jeu HotpotQA (articles non split)
        for source, _, edge_data in list(graph.in_edges(node_id, data=True)):
            graph.add_edge(source, entry_node, **edge_data)
        graph.remove_node(node_id)


async def _combined_graph(slugs: list[str], cache: dict[str, nx.MultiDiGraph]) -> nx.MultiDiGraph:
    """Graphe union des pages de `slugs`, chacune rendue au plus une fois par
    run grâce à `cache` -- des articles "pont" reviennent d'une question à
    l'autre dans HotpotQA. Chemin TF-IDF uniquement (pas besoin des images de
    tuile : le scorer lit `text_preview` sur le graphe). `nx.compose` retourne
    un nouveau graphe à chaque appel : rappeler cette fonction pour les mêmes
    `slugs` (ex. un autre k_seed) donne un graphe frais, sans re-rendu."""
    combined = nx.MultiDiGraph()
    for slug in slugs:
        if slug not in cache:
            page_graph = await build_page_graph(_url_for_slug(slug))
            cache[slug] = _namespaced(page_graph, slug)
        combined = nx.compose(combined, cache[slug])
    _rewire_cross_links(combined, slugs)
    return combined


async def _combined_tiles_and_graph(
    slugs: list[str], cache: dict[str, tuple[list[Tile], nx.MultiDiGraph]],
) -> tuple[list[Tile], nx.MultiDiGraph]:
    """Comme `_combined_graph`, mais garde aussi les tuiles (images PIL) --
    nécessaires pour le scorer "reader", qui embarque l'image de chaque
    tuile candidate (le graphe seul, via build_page_graph, ne garde que le
    texte). Même remarque sur la fraîcheur du graphe retourné."""
    combined_graph = nx.MultiDiGraph()
    combined_tiles: list[Tile] = []
    for slug in slugs:
        if slug not in cache:
            elements, screenshot = await extract_dom_elements_async(_url_for_slug(slug), wait_until="load")
            tiles = build_tiles(elements, screenshot)
            graph = build_graph(tiles, page_url=_url_for_slug(slug), page_title=slug)
            namespaced_tiles = [replace(t, id=f"{slug}::{t.id}") for t in tiles]
            namespaced_graph = nx.relabel_nodes(graph, {n: f"{slug}::{n}" for n in graph.nodes}, copy=True)
            cache[slug] = (namespaced_tiles, namespaced_graph)
        tiles, graph = cache[slug]
        combined_tiles.extend(tiles)
        combined_graph = nx.compose(combined_graph, graph)
    _rewire_cross_links(combined_graph, slugs)
    return combined_tiles, combined_graph


def _static_topk_opened(graph: nx.MultiDiGraph, k: int) -> set[str]:
    """Baseline : les k noeuds au score le plus haut, tous globalement --
    aucune notion de budget dépensé au fil de l'exploration ni de propagation
    le long de reading_order, juste un classement figé sur les mêmes scores
    que le contrôleur (posés en `relevance_score` par run_evidence_controller,
    qu'ils viennent de TF-IDF ou du lecteur)."""
    scored = [
        (n, d.get("relevance_score", 0.0))
        for n, d in graph.nodes(data=True) if d.get("type") == "element"
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {n for n, _ in scored[:k]}


def _reader_scores_for_graph(reader, query: str, graph: nx.MultiDiGraph, tiles: list[Tile]) -> dict[str, float]:
    """Scores du lecteur VLM fine-tuné pour tous les noeuds élément de
    `graph`, via `reader` (cf. `_load_reader`). Une tuile élément sans image
    connue (ne devrait pas arriver : tous les noeuds élément viennent de
    `build_tiles`) reçoit un score de 0.0 par défaut côté
    `run_evidence_controller` (precomputed_scores.get(n, 0.0))."""
    from src.lora_finetune import reader_relevance_scores

    tiles_by_id = {t.id: t for t in tiles}
    element_ids = [n for n, d in graph.nodes(data=True) if d.get("type") == "element" and n in tiles_by_id]
    images = [tiles_by_id[n].image for n in element_ids]
    model, processor, batched = reader
    return reader_relevance_scores(model, processor, query, element_ids, images, batched=batched)


async def _rebuild_graph(
    example: dict, scorer: str, page_graph_cache: dict,
) -> tuple[list[Tile], nx.MultiDiGraph] | None:
    """(tuiles, graphe) FRAIS (jamais muté par un run précédent du contrôleur)
    pour `example`, ou None si ses positifs ne couvrent qu'une seule page."""
    slugs = sorted({slug for slug, _ in example["positive_tiles"]})
    if len(slugs) < 2:
        return None
    if scorer == "reader":
        return await _combined_tiles_and_graph(slugs, page_graph_cache)
    return [], await _combined_graph(slugs, page_graph_cache)


def evaluate_on_graph(
    example: dict, tiles: list[Tile], graph: nx.MultiDiGraph, budget: int, seed_k: int,
    precomputed_scores: dict[str, float] | None,
) -> dict:
    """Fait tourner le contrôleur (mute `graph` en place) puis le baseline
    top-k statique sur les mêmes scores, et mesure la couverture cross-page.
    `graph` doit être fraîchement construit (jamais passé à un run
    précédent) : `run_evidence_controller` mute son attribut `state`."""
    slugs = sorted({slug for slug, _ in example["positive_tiles"]})
    positive_nodes_by_page = {
        slug: {f"{slug}::{tile_id}" for s, tile_id in example["positive_tiles"] if s == slug}
        for slug in slugs
    }
    # Une tuile positive absente du graphe re-rendu (article changé depuis la
    # construction du dataset multi-hop) ne doit jamais compter comme
    # "couverte" -- filtré implicitement puisque son id n'existera simplement
    # pas parmi les noeuds "opened"/top-k, jamais un faux positif.

    run_evidence_controller(
        graph, tiles=tiles, question=example["question"],
        controller_config=EvidenceControllerConfig(budget=budget, top_k_seed=seed_k),
        precomputed_scores=precomputed_scores,
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


def _load_reader(adapter_path: str, batched: bool):
    """Charge le modèle de base + adaptateur LoRA (cf. scripts/eval_retrieval.py) --
    une seule fois pour tout le run, réutilisé question après question et
    d'une valeur de seed_k à l'autre.

    `device_map="auto"` (utilisé jusqu'ici par scripts/eval_retrieval.py) fait
    passer `peft.PeftModel.from_pretrained` par le chemin d'offload CPU/disque
    d'accelerate, qui échoue ici avec un `KeyError` sur le nom d'un sous-module
    (`_update_offload`) -- une régression de compat peft/accelerate/transformers
    par rapport aux versions utilisées lors du run initial (Appendix B), pas
    un problème avec l'adaptateur lui-même.

    `.from_pretrained(...).to(device)` (une autre option écartée) matérialise
    d'abord le modèle entier sur CPU puis le copie vers MPS -- sur une machine
    à mémoire unifiée de 17 Go pour un modèle bf16 de ~15 Go, ce doublement
    transitoire (CPU + MPS coexistent brièvement) sature la RAM et fait
    "stuck" le process (constaté empiriquement : `top` rapporte l'état STUCK,
    la machine thrashe indéfiniment même après avoir fermé toutes les autres
    apps). `device_map={"": device}` matérialise directement sur le device
    cible, sans jamais dupliquer -- chargé en ~9s dans les mêmes conditions."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from src.config import HF_MODEL

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_multihop_controller] chargement du modèle de base ({HF_MODEL}) sur {device}...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    processor = AutoProcessor.from_pretrained(HF_MODEL)
    print(f"[eval_multihop_controller] chargement de l'adaptateur LoRA depuis {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, processor, batched


async def main_async(
    budget: int, seed_ks: list[int], max_questions: int | None, scorer: str, adapter_path: str, batched: bool,
) -> None:
    if not MULTIHOP_EXAMPLES_PATH.exists():
        raise SystemExit(f"{MULTIHOP_EXAMPLES_PATH} introuvable -- lance d'abord scripts/build_multihop_dataset.py")

    examples = [json.loads(line) for line in MULTIHOP_EXAMPLES_PATH.open(encoding="utf-8") if line.strip()]
    if max_questions is not None:
        examples = examples[:max_questions]

    reader = None
    no_grad_cm = None
    if scorer == "reader":
        import torch

        reader = _load_reader(adapter_path, batched)
        no_grad_cm = torch.no_grad()
        no_grad_cm.__enter__()

    page_graph_cache: dict = {}
    score_cache: dict[str, dict[str, float] | None] = {}  # id exemple -> scores lecteur (calculés une fois)
    results_by_seed_k: dict[int, list[dict]] = {k: [] for k in seed_ks}
    try:
        for i, ex in enumerate(examples, 1):
            built = await _rebuild_graph(ex, scorer, page_graph_cache)
            if built is None:
                print(f"[{i}/{len(examples)}] {ex['id']}... skip -- positifs sur une seule page", flush=True)
                continue
            tiles, graph_template = built

            if scorer == "reader" and ex["id"] not in score_cache:
                try:
                    score_cache[ex["id"]] = _reader_scores_for_graph(reader, ex["question"], graph_template, tiles)
                except Exception as exc:  # noqa: BLE001 - une question individuelle : on continue sur les suivantes
                    print(f"[{i}/{len(examples)}] {ex['id']}... ÉCHEC scoring ({exc!r}), skip", flush=True)
                    score_cache[ex["id"]] = "error"

            precomputed_scores = score_cache.get(ex["id"]) if scorer == "reader" else None
            if precomputed_scores == "error":
                continue

            print(f"[{i}/{len(examples)}] {ex['id']}...", flush=True)
            for seed_k in seed_ks:
                try:
                    # Un graphe frais par (exemple, seed_k) : run_evidence_controller
                    # mute `state`, jamais rejouable tel quel sur un run précédent.
                    _, graph = await _rebuild_graph(ex, scorer, page_graph_cache)
                    r = evaluate_on_graph(ex, tiles, graph, budget, seed_k, precomputed_scores)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ÉCHEC (seed_k={seed_k}, {exc!r}), skip", flush=True)
                    continue
                results_by_seed_k[seed_k].append(r)
                print(
                    f"  [seed_k={seed_k}] contrôleur : {r['controller_pages_covered']}/{r['n_pages']} pages couvertes "
                    f"({r['controller_opened']} tuiles ouvertes) | "
                    f"top-k statique : {r['static_pages_covered']}/{r['n_pages']}",
                    flush=True,
                )
    finally:
        if no_grad_cm is not None:
            no_grad_cm.__exit__(None, None, None)

    all_results: dict[str, dict] = {}
    for seed_k, results in results_by_seed_k.items():
        n = len(results)
        if n == 0:
            print(f"\nseed_k={seed_k} : aucun résultat exploitable.")
            continue
        controller_full = sum(1 for r in results if r["controller_pages_covered"] == r["n_pages"])
        static_full = sum(1 for r in results if r["static_pages_covered"] == r["n_pages"])
        print(f"\n{'=' * 60}\n{n} questions cross-page évaluées (scorer={scorer}, budget={budget}, seed_k={seed_k})\n{'=' * 60}")
        print(f"Contrôleur      : {controller_full}/{n} ({100 * controller_full / n:.1f}%) -- les DEUX pages couvertes")
        print(f"Top-k statique  : {static_full}/{n} ({100 * static_full / n:.1f}%) -- mêmes scores, pas de politique de contrôleur")
        all_results[str(seed_k)] = {"results": results, "controller_full": controller_full, "static_full": static_full, "n": n}

    out_path = CONTROLLER_EVAL_PATH if scorer == "tfidf" else CONTROLLER_EVAL_PATH.with_name(
        f"{CONTROLLER_EVAL_PATH.stem}_{scorer}{CONTROLLER_EVAL_PATH.suffix}"
    )
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nRésultats détaillés écrits dans {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=5, help="Budget d'ouverture du contrôleur (cf. EvidenceControllerConfig).")
    parser.add_argument(
        "--seed-k", type=int, nargs="+", default=[3],
        help="Nombre(s) de noeuds activés au seed (cf. EvidenceControllerConfig) -- accepte plusieurs valeurs, ex. --seed-k 3 5.",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--scorer", choices=["tfidf", "reader"], default="tfidf",
        help="tfidf (défaut, placeholder lexical) ou reader (lecteur VLM fine-tuné, cf. docstring du module).",
    )
    parser.add_argument("--adapter-path", type=str, default="data/lora_reader_adapter/final")
    parser.add_argument(
        "--no-batched-embeddings", dest="batched_embeddings", action="store_false",
        help="cf. src/lora_finetune.py --no-batched-embeddings : un appel modèle par élément plutôt que batché.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(
        args.budget, args.seed_k, args.max_questions, args.scorer, args.adapter_path, args.batched_embeddings,
    ))


if __name__ == "__main__":
    main()
