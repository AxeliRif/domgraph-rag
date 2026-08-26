"""
Évaluation bout-en-bout (accuracy) du pipeline complet -- tuilage, graphe,
contrôleur d'évidence, lecture VLM des tuiles ouvertes, synthèse finale --
sur les questions multi-hop de data/qa_dataset/multihop_examples.jsonl
(construites depuis HotpotQA, cf. scripts/build_multihop_dataset.py).

Contrairement à scripts/eval_multihop_controller.py (qui ne teste que la
couverture des tuiles positives par le contrôleur, sans VLM), ce script
exerce la chaîne complète telle qu'un utilisateur la vivrait via webapp/ :
`run_evidence_controller(..., use_vlm=True)` fait relire chaque tuile ouverte
par le VLM (pas seulement son text_preview), puis une synthèse finale
répond à la question à partir de cette évidence -- le pendant, sur nos
propres données multi-hop, de la métrique "Accuracy" que MAGE-RAG rapporte
sur LongDocURL/MMLongBench-Doc (cf. Table~1 du papier), non directement
comparable (jeu de données différent) mais du même esprit : le système
répond-il correctement à une vraie question multi-hop ?

Scoré en EM (exact match) et F1 (chevauchement de tokens) après la
normalisation standard SQuAD/HotpotQA (minuscules, ponctuation et articles
retirés) -- pas de correspondance de sous-chaîne brute, qui pénaliserait
injustement une réponse correcte mais reformulée.

Coût : contrairement à scripts/eval_multihop_controller.py, chaque tuile
ouverte ET la synthèse finale sont un appel VLM (cf. le coût de ~2 min/tuile
en local documenté dans le papier, Appendix/Status) -- lance d'abord un petit
échantillon (--max-questions 5) pour mesurer le débit réel sur cette machine
avant de lancer sur l'ensemble des questions.

Échantillonnage : --max-questions N tire N exemples au hasard (pas les N
premiers du fichier) parmi ceux disponibles, via random.Random(--seed)
(défaut 0) pour un tirage reproductible d'un run à l'autre. Le fichier
multihop_examples.jsonl lui-même hérite de l'ordre de streaming HotpotQA
(scripts/build_multihop_dataset.py, non mélangé) ; un simple `[:N]` prendrait
donc un préfixe de cet ordre plutôt qu'un échantillon, avec un risque de
sur-représenter les articles-pont qui reviennent sur des questions
consécutives (cf. le cache par slug ci-dessous).

Usage :
    python3 -m scripts.eval_multihop_e2e --max-questions 5
    python3 -m scripts.eval_multihop_e2e --budget 5 --seed-k 3
    python3 -m scripts.eval_multihop_e2e --max-questions 20 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import string
import time
from dataclasses import replace

import networkx as nx

from src.config import ANSWER_SYNTHESIS_PROMPT, QA_DATASET_DIR
from src.dom_extraction import extract_dom_elements_async
from src.evidence_controller import EvidenceControllerConfig, collect_evidence, run_evidence_controller
from src.graph_builder import build_graph
from src.tiling import Tile, build_tiles
from src.vlm_client import VLMClient

MULTIHOP_EXAMPLES_PATH = QA_DATASET_DIR / "multihop_examples.jsonl"
E2E_EVAL_PATH = QA_DATASET_DIR / "multihop_e2e_eval.json"

# Réponse courte demandée explicitement (contrairement à ANSWER_SYNTHESIS_PROMPT,
# pensé pour le site de démo où une phrase complète est plus lisible) : HotpotQA
# attend des réponses courtes (entité, date, "yes"/"no"...), et un score EM/F1
# contre une phrase complète pénaliserait une réponse par ailleurs correcte pour
# une raison de forme, pas de fond.
SHORT_ANSWER_SUFFIX = " Answer in as few words as possible (a short phrase, not a sentence)."


def _url_for_slug(slug: str) -> str:
    return f"https://en.wikipedia.org/wiki/{slug}"


async def _page_tiles_and_graph(slug: str, cache: dict) -> tuple[list[Tile], nx.MultiDiGraph]:
    """Tuiles + graphe namespacés d'une page, mis en cache par slug (des
    articles "pont" reviennent d'une question à l'autre dans HotpotQA)."""
    if slug not in cache:
        elements, screenshot = await extract_dom_elements_async(_url_for_slug(slug), wait_until="load")
        tiles = build_tiles(elements, screenshot)
        graph = build_graph(tiles, page_url=_url_for_slug(slug), page_title=slug)
        namespaced_tiles = [replace(t, id=f"{slug}::{t.id}") for t in tiles]
        namespaced_graph = nx.relabel_nodes(graph, {n: f"{slug}::{n}" for n in graph.nodes}, copy=True)
        cache[slug] = (namespaced_tiles, namespaced_graph)
    return cache[slug]


async def _combined(slugs: list[str], cache: dict) -> tuple[list[Tile], nx.MultiDiGraph]:
    combined_graph = nx.MultiDiGraph()
    combined_tiles: list[Tile] = []
    for slug in slugs:
        tiles, graph = await _page_tiles_and_graph(slug, cache)
        combined_graph = nx.compose(combined_graph, graph)
        combined_tiles.extend(tiles)
    return combined_tiles, combined_graph


# --- Normalisation + EM/F1 standard SQuAD/HotpotQA -------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _exact_match(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


def _f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.get(t, 0) for t in set(pred_tokens))
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


async def evaluate_example(
    example: dict, budget: int, seed_k: int, page_cache: dict, vlm: VLMClient,
) -> dict | None:
    slugs = sorted({slug for slug, _ in example["positive_tiles"]})
    if len(slugs) < 2:
        return None

    try:
        tiles, graph = await _combined(slugs, page_cache)
    except Exception as exc:  # noqa: BLE001 - une question individuelle : on continue sur les suivantes
        return {"id": example["id"], "error": repr(exc)}

    config = EvidenceControllerConfig(budget=budget, top_k_seed=seed_k, use_vlm=True)
    run_evidence_controller(graph, tiles, example["question"], config)
    evidence = collect_evidence(graph)

    if evidence.strip():
        prompt = ANSWER_SYNTHESIS_PROMPT.format(evidence=evidence, question=example["question"]) + SHORT_ANSWER_SUFFIX
        prediction = vlm.ask_text(prompt)
    else:
        prediction = ""

    gold = example["answer"]
    return {
        "id": example["id"],
        "question": example["question"],
        "gold": gold,
        "prediction": prediction,
        "em": _exact_match(prediction, gold),
        "f1": _f1(prediction, gold),
    }


async def main_async(
    budget: int, seed_k: int, max_questions: int | None, seed: int, ids: list[str] | None = None
) -> None:
    if not MULTIHOP_EXAMPLES_PATH.exists():
        raise SystemExit(f"{MULTIHOP_EXAMPLES_PATH} introuvable -- lance d'abord scripts/build_multihop_dataset.py")

    examples = [json.loads(line) for line in MULTIHOP_EXAMPLES_PATH.open(encoding="utf-8") if line.strip()]
    if ids is not None:
        by_id = {ex["id"]: ex for ex in examples}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"id(s) introuvable(s) dans {MULTIHOP_EXAMPLES_PATH} : {missing}")
        examples = [by_id[i] for i in ids]
        print(f"relance ciblée de {len(examples)} question(s) : {ids}\n", flush=True)
    elif max_questions is not None and max_questions < len(examples):
        n_total = len(examples)
        examples = random.Random(seed).sample(examples, k=max_questions)
        print(f"tirage aléatoire de {max_questions}/{n_total} -- seed={seed} : "
              f"{[ex['id'] for ex in examples]}\n", flush=True)

    page_cache: dict = {}
    vlm = VLMClient()
    results: list[dict] = []
    t0 = time.time()
    for i, ex in enumerate(examples, 1):
        q_t0 = time.time()
        print(f"[{i}/{len(examples)}] {ex['id']} -- {ex['question'][:70]}...", flush=True)
        r = await evaluate_example(ex, budget, seed_k, page_cache, vlm)
        if r is None:
            print("  skip -- positifs sur une seule page", flush=True)
            continue
        if "error" in r:
            print(f"  ÉCHEC ({r['error']}), skip", flush=True)
            continue
        results.append(r)
        print(
            f"  gold={r['gold']!r} | pred={r['prediction']!r} | EM={r['em']} F1={r['f1']:.2f} "
            f"({time.time() - q_t0:.0f}s)",
            flush=True,
        )

    n = len(results)
    elapsed = time.time() - t0
    run_log = {
        "budget": budget, "seed_k": seed_k, "max_questions": max_questions, "seed": seed,
        "ids": ids, "n_evaluated": n, "elapsed_s": round(elapsed),
    }

    if n == 0:
        print("Aucun résultat exploitable.")
    else:
        em = sum(r["em"] for r in results) / n
        f1 = sum(r["f1"] for r in results) / n
        print(f"\n{'=' * 60}\n{n} questions évaluées (budget={budget}, seed_k={seed_k})\n{'=' * 60}")
        print(f"EM = {em:.3f}   F1 = {f1:.3f}")
        print(f"Temps total : {elapsed:.0f}s ({elapsed / n:.0f}s/question en moyenne)")

    # Fusionne avec un run précédent plutôt que d'écraser -- une relance ciblée
    # (--ids) sur les questions tombées en échec réseau doit compléter le
    # fichier existant, pas repartir de zéro et perdre les questions déjà
    # évaluées avec succès.
    prior = json.loads(E2E_EVAL_PATH.read_text(encoding="utf-8")) if E2E_EVAL_PATH.exists() else None
    merged_results = {r["id"]: r for r in (prior["results"] if prior else [])}
    merged_results.update({r["id"]: r for r in results})
    sampled_ids = sorted(set((prior["sampled_ids"] if prior else [])) | {ex["id"] for ex in examples})
    runs = (prior["runs"] if prior and "runs" in prior else []) + [run_log]

    output = {"runs": runs, "sampled_ids": sampled_ids, "results": list(merged_results.values())}
    E2E_EVAL_PATH.write_text(json.dumps(output, indent=2))

    n_merged = len(merged_results)
    n_expected = len(sampled_ids)
    print(f"\nRésultats fusionnés écrits dans {E2E_EVAL_PATH} "
          f"({n_merged}/{n_expected} question(s) échantillonnée(s) au total ont un résultat)")
    if n_merged < n_expected:
        still_missing = sorted(set(sampled_ids) - set(merged_results))
        print(f"encore manquant(es) : {still_missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--seed-k", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Seed du tirage aléatoire pour --max-questions.")
    parser.add_argument(
        "--ids", type=str, default=None,
        help="IDs séparés par des virgules à (re)évaluer, en ignorant --max-questions/--seed -- "
             "pour relancer juste les questions tombées en échec (ex. déconnexion réseau) sans "
             "reperdre les résultats déjà obtenus (fusionnés dans multihop_e2e_eval.json).",
    )
    args = parser.parse_args()
    ids = args.ids.split(",") if args.ids else None
    asyncio.run(main_async(args.budget, args.seed_k, args.max_questions, args.seed, ids))


if __name__ == "__main__":
    main()
