"""
Ablation (b), volet HotpotQA : le lecteur VLM fine-tuné (Phase 2) est
entraîné et évalué (scripts/eval_retrieval.py) sur des questions
synthétiques -- générées par le même modèle qui sert aussi de juge de faux
négatifs (paper, Limitations, "same-model-family"). Ce script rejoue la
même mesure (rang de la tuile positive parmi TOUTES les tuiles de sa page,
Recall@k / MRR) mais sur les questions HotpotQA (multihop_qa.py) --
écrites par des humains, jamais vues à l'entraînement -- pour vérifier que le
gain mesuré n'est pas un artefact de la famille de modèle partagée entre
générateur et lecteur.

Chaque exemple HotpotQA a deux tuiles positives (une par article cité,
`positive_tiles`) : on évalue chaque page séparément comme un retrieval
single-page ordinaire (pas de graphe ni de contrôleur ici, cf.
eval_multihop_controller.py pour la mesure cross-page), exactement le
protocole d'eval_retrieval.py.

Usage :
    python3 -m scripts.eval_hotpotqa_reader
    python3 -m scripts.eval_hotpotqa_reader --max-questions 20
"""
from __future__ import annotations

import argparse
import json

from src.config import QA_DATASET_DIR

MULTIHOP_EXAMPLES_PATH = QA_DATASET_DIR / "multihop_examples.jsonl"


def _url_for_slug(slug: str) -> str:
    return f"https://en.wikipedia.org/wiki/{slug}"


async def _tiles_for_page(slug: str, cache: dict):
    if slug not in cache:
        from src.dom_extraction import extract_dom_elements_async
        from src.tiling import build_tiles

        elements, screenshot = await extract_dom_elements_async(_url_for_slug(slug), wait_until="load")
        cache[slug] = build_tiles(elements, screenshot)
    return cache[slug]


async def evaluate(model, processor, examples: list[dict], batched: bool, max_questions: int | None):
    import torch

    from src.lora_finetune import embed_questions, embed_tiles

    subset = examples if max_questions is None else examples[:max_questions]
    page_tile_cache: dict = {}
    tile_embed_cache: dict = {}  # slug -> (tile_ids, embeds)
    ranks: list[int] = []
    skipped = 0

    for i, ex in enumerate(subset):
        for slug, positive_tile_id in ex["positive_tiles"]:
            try:
                tiles = await _tiles_for_page(slug, page_tile_cache)
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            if len(tiles) < 2:
                skipped += 1
                continue
            tile_ids = [t.id for t in tiles]
            if positive_tile_id not in tile_ids:
                skipped += 1  # article changé depuis la construction du dataset multi-hop
                continue

            if slug not in tile_embed_cache:
                images = [t.image for t in tiles]
                tile_embed_cache[slug] = (tile_ids, embed_tiles(model, processor, images, batched=batched))
            cached_ids, tile_embeds = tile_embed_cache[slug]

            positive_idx = cached_ids.index(positive_tile_id)
            question_embed = embed_questions(model, processor, [ex["question"]], batched=batched)
            similarities = torch.nn.functional.cosine_similarity(question_embed, tile_embeds)
            ranked = similarities.argsort(descending=True).tolist()
            ranks.append(ranked.index(positive_idx))

        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(subset)} questions évaluées", flush=True)

    return ranks, skipped


def _report(ranks: list[int], skipped: int, k_values: list[int]) -> None:
    print(f"\n{len(ranks)} (page, question) évaluées, {skipped} ignorées.")
    if not ranks:
        print("Aucun résultat.")
        return
    for k in k_values:
        recall_at_k = sum(1 for r in ranks if r < k) / len(ranks)
        print(f"  Recall@{k} = {recall_at_k:.3f}")
    mrr = sum(1 / (r + 1) for r in ranks) / len(ranks)
    print(f"  MRR = {mrr:.3f}")


async def main_async(adapter_path: str, batched: bool, max_questions: int | None, k_values: list[int]) -> None:
    if not MULTIHOP_EXAMPLES_PATH.exists():
        raise SystemExit(f"{MULTIHOP_EXAMPLES_PATH} introuvable -- lance d'abord scripts/build_multihop_dataset.py")

    examples = [json.loads(line) for line in MULTIHOP_EXAMPLES_PATH.open(encoding="utf-8") if line.strip()]

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from src.config import HF_MODEL

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_hotpotqa_reader] chargement du modèle de base ({HF_MODEL}) sur {device}...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    processor = AutoProcessor.from_pretrained(HF_MODEL)
    print(f"[eval_hotpotqa_reader] chargement de l'adaptateur LoRA depuis {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    with torch.no_grad():
        print("\n[eval_hotpotqa_reader] passe SANS LoRA (modèle de base)...")
        with model.disable_adapter():
            base_ranks, base_skipped = await evaluate(model, processor, examples, batched, max_questions)
        print("\n[eval_hotpotqa_reader] passe AVEC LoRA (adaptateur fine-tuné)...")
        lora_ranks, lora_skipped = await evaluate(model, processor, examples, batched, max_questions)

    print("\n--- base (sans LoRA) ---")
    _report(base_ranks, base_skipped, k_values)
    print("\n--- LoRA fine-tuné ---")
    _report(lora_ranks, lora_skipped, k_values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=str, default="data/lora_reader_adapter/final")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--no-batched-embeddings", dest="batched_embeddings", action="store_false",
        help="cf. src/lora_finetune.py --no-batched-embeddings : un appel modèle par élément plutôt que batché "
             "(recommandé sur machine à mémoire limitée, cf. Appendix C).",
    )
    args = parser.parse_args()
    import asyncio

    asyncio.run(main_async(args.adapter_path, args.batched_embeddings, args.max_questions, args.k))


if __name__ == "__main__":
    main()
