"""
Évaluation retrieval du VLM lecteur : base (sans LoRA) vs adaptateur LoRA
fine-tuné (Phase 2d, src/lora_finetune.py).

La loss d'entraînement affichée par `train_lora` (InfoNCE moyenne) ne mesure
que la classification, à chaque étape, de la tuile positive parmi 1 positif +
`n_negatives` négatifs difficiles + les positifs/négatifs "en batch" des
autres questions -- un pool de candidats petit (quelques unités) et fixé par
`TrainConfig.n_negatives`/`batch_size`. Ce script mesure autre chose : pour
chaque question du split val, le rang de la tuile positive parmi TOUTES les
tuiles de sa page (pool réaliste, comparable à ce que ferait un vrai
retriever), en Recall@k et MRR -- et compare cette mesure avec et sans
l'adaptateur LoRA chargé sur le MÊME modèle de base (via
`peft.PeftModel.disable_adapter()`, qui désactive temporairement l'adaptateur
plutôt que de recharger un second modèle en mémoire).

ATTENTION -- lecture des résultats : `src/contrastive_dataset.py` n'avait pas
de notion de split train/val au moment où `lora_finetune.train_lora` a tourné
(cf. `split_examples_by_article`, ajoutée après coup). Si l'adaptateur évalué
ici a été entraîné sur l'intégralité de `contrastive_examples.jsonl` (le
comportement par défaut jusqu'ici), le split "val" utilisé par CE script n'est
PAS un vrai jeu tenu à l'écart pour CET adaptateur précis -- ses questions ont
été vues à l'entraînement, donc les chiffres obtenus sont optimistes. Ce
script reste utile pour comparer base vs LoRA (le LoRA a-t-il appris quelque
chose du tout ?) et pour fournir une mesure propre sur un PROCHAIN run,
entraîné avec `ContrastiveTileDataset(examples=train_examples)` (train
uniquement, cf. `split_examples_by_article`) plutôt que sur la totalité.

Usage :
    python3 -m scripts.eval_retrieval
    python3 -m scripts.eval_retrieval --max-questions 20   # test rapide
    python3 -m scripts.eval_retrieval --adapter-path data/lora_reader_adapter/checkpoints/epoch-2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.config import CONTRASTIVE_EXAMPLES_PATH, HF_MODEL, LORA_OUTPUT_DIR, QA_DATASET_DIR, TILES_MANIFEST_PATH
from src.contrastive_dataset import _article_root, load_tile_image_paths, split_examples_by_article
from src.hard_negative_mining import ContrastiveExample, load_contrastive_examples
from src.lora_finetune import embed_questions, embed_tiles


def _tiles_for_page(page_slug: str, image_paths: dict[str, str], images_root: Path) -> tuple[list[str], list[Image.Image]]:
    """Toutes les tuiles connues du manifeste pour `page_slug` -- le pool de
    candidats du retrieval simulé (pas juste le positif + les négatifs minés
    de tel ou tel exemple)."""
    prefix = f"{page_slug}::"
    tile_ids = [key[len(prefix):] for key in image_paths if key.startswith(prefix)]
    images = [Image.open(images_root / image_paths[f"{prefix}{tid}"]).convert("RGB") for tid in tile_ids]
    return tile_ids, images


def evaluate(
    model, processor, examples: list[ContrastiveExample], image_paths: dict[str, str], images_root: Path,
    batched: bool, max_questions: int | None = None,
) -> tuple[list[int], int]:
    """Retourne (rangs, nb_ignorés). `rangs[i]` = position (0 = premier) de la
    tuile positive de `examples[i]` dans le classement par similarité
    décroissante sur TOUTES les tuiles de sa page. Les tuiles d'une même page
    sont embarquées une seule fois (cache), pas une fois par question --
    plusieurs questions du split partagent souvent la même page."""
    import torch

    subset = examples if max_questions is None else examples[:max_questions]
    tile_cache: dict[str, tuple[list[str], torch.Tensor]] = {}
    ranks: list[int] = []
    skipped = 0

    for i, ex in enumerate(subset):
        if ex.page_slug not in tile_cache:
            tile_ids, images = _tiles_for_page(ex.page_slug, image_paths, images_root)
            if len(tile_ids) < 2:
                skipped += 1
                continue
            tile_embeds = embed_tiles(model, processor, images, batched=batched)
            tile_cache[ex.page_slug] = (tile_ids, tile_embeds)
        tile_ids, tile_embeds = tile_cache.get(ex.page_slug, ([], None))
        if ex.positive_tile_id not in tile_ids:
            skipped += 1  # tuile positive absente du manifeste (run désynchronisé) -- cf. ContrastiveTileDataset._is_resolvable
            continue

        positive_idx = tile_ids.index(ex.positive_tile_id)
        question_embed = embed_questions(model, processor, [ex.question], batched=batched)
        similarities = torch.nn.functional.cosine_similarity(question_embed, tile_embeds)
        ranked = similarities.argsort(descending=True).tolist()
        ranks.append(ranked.index(positive_idx))

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(subset)} questions évaluées")

    return ranks, skipped


def _report(label: str, ranks: list[int], skipped: int, k_values: list[int]) -> None:
    print(f"\n--- {label} ---")
    print(f"  {len(ranks)} questions évaluées, {skipped} ignorées (page < 2 tuiles ou tuile positive introuvable)")
    if not ranks:
        print("  Aucun résultat.")
        return
    for k in k_values:
        recall_at_k = sum(1 for r in ranks if r < k) / len(ranks)
        print(f"  Recall@{k} = {recall_at_k:.3f}")
    mrr = sum(1 / (r + 1) for r in ranks) / len(ranks)
    print(f"  MRR = {mrr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=str, default=str(LORA_OUTPUT_DIR / "final"))
    parser.add_argument("--contrastive-examples-path", type=str, default=str(CONTRASTIVE_EXAMPLES_PATH))
    parser.add_argument("--tiles-manifest-path", type=str, default=str(TILES_MANIFEST_PATH))
    parser.add_argument("--images-root", type=str, default=str(QA_DATASET_DIR))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=None, help="Limite le nombre de questions évaluées (test rapide).")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--no-batched-embeddings", dest="batched_embeddings", action="store_false",
        help="cf. src/lora_finetune.py --no-batched-embeddings : un appel modèle par élément plutôt que batché.",
    )
    args = parser.parse_args()

    examples = load_contrastive_examples(Path(args.contrastive_examples_path))
    if not examples:
        raise RuntimeError(f"Aucun exemple contrastif trouvé dans {args.contrastive_examples_path}")
    _, val_examples = split_examples_by_article(examples, val_fraction=args.val_fraction, seed=args.seed)
    val_articles = {_article_root(ex.page_slug) for ex in val_examples}
    print(
        f"[eval_retrieval] {len(examples)} exemples au total -> {len(val_examples)} dans le split val "
        f"({len(val_articles)} articles : {sorted(val_articles)})"
    )

    image_paths = load_tile_image_paths(Path(args.tiles_manifest_path))
    if not image_paths:
        raise RuntimeError(f"Manifeste de tuiles introuvable ou vide : {args.tiles_manifest_path}")
    images_root = Path(args.images_root)

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    print(f"[eval_retrieval] chargement du modèle de base ({HF_MODEL})...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(HF_MODEL)

    print(f"[eval_retrieval] chargement de l'adaptateur LoRA depuis {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    with torch.no_grad():
        print("\n[eval_retrieval] passe SANS LoRA (modèle de base, adaptateur désactivé)...")
        with model.disable_adapter():
            base_ranks, base_skipped = evaluate(
                model, processor, val_examples, image_paths, images_root, args.batched_embeddings, args.max_questions
            )
        print("\n[eval_retrieval] passe AVEC LoRA (adaptateur fine-tuné)...")
        lora_ranks, lora_skipped = evaluate(
            model, processor, val_examples, image_paths, images_root, args.batched_embeddings, args.max_questions
        )

    _report("base (sans LoRA)", base_ranks, base_skipped, args.k)
    _report("LoRA fine-tuné", lora_ranks, lora_skipped, args.k)
    print(
        "\nRappel : si cet adaptateur a été entraîné sur l'intégralité de contrastive_examples.jsonl "
        "(pas seulement le split train), ces chiffres sont optimistes -- cf. docstring de ce module."
    )


if __name__ == "__main__":
    main()
