r"""
Baseline TF-IDF pour l'ablation (b) (scripts/eval_retrieval.py) : même
protocole exact (split val par article, pool = TOUTES les tuiles de la page,
Recall@k/MRR), mais le "lecteur" est le scorer lexical déjà utilisé pour le
hard-negative mining et le contrôleur (\code{hard_negative_mining.py},
\code{evidence_controller.py}) plutôt qu'un VLM -- CPU uniquement, aucun
GPU ni modèle à charger.

But : situer le gain de l'ablation (b) (base VLM -> LoRA fine-tuné) par
rapport à un plancher purement lexical sur le MÊME split val, exactement
comparable puisque même pool de candidats et mêmes questions.

Usage :
    python3 -m scripts.eval_retrieval_tfidf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CONTRASTIVE_EXAMPLES_PATH, TILES_MANIFEST_PATH
from src.contrastive_dataset import _article_root, split_examples_by_article
from src.hard_negative_mining import ContrastiveExample, load_contrastive_examples

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - repli sans scikit-learn
    TfidfVectorizer = None
    cosine_similarity = None


def load_tile_texts(manifest_path: Path = TILES_MANIFEST_PATH) -> dict[str, str]:
    """Charge le manifeste des tuiles, retourne {page_slug::tile_id: text_preview}
    -- même clé que `contrastive_dataset.load_tile_image_paths`, mais le texte
    plutôt que le chemin d'image (le scorer TF-IDF n'a besoin que du texte)."""
    if not manifest_path.exists():
        return {}
    texts: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            texts[f"{record['page_slug']}::{record['tile_id']}"] = record.get("text_preview", "")
    return texts


def _tiles_for_page(page_slug: str, tile_texts: dict[str, str]) -> tuple[list[str], list[str]]:
    prefix = f"{page_slug}::"
    tile_ids = [key[len(prefix):] for key in tile_texts if key.startswith(prefix)]
    texts = [tile_texts[f"{prefix}{tid}"] for tid in tile_ids]
    return tile_ids, texts


def _rank_by_tfidf(question: str, texts: list[str]) -> list[float]:
    """Même mécanisme que `hard_negative_mining.mine_hard_negatives` :
    TF-IDF + cosinus, avec repli sans scikit-learn."""
    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        try:
            matrix = vectorizer.fit_transform([*texts, question])
            return cosine_similarity(matrix[-1], matrix[:-1])[0].tolist()
        except ValueError:  # vocabulaire vide (textes tous vides)
            pass
    # Repli : chevauchement de tokens, cf. hard_negative_mining._fallback_cosine_similarities
    import re
    from collections import Counter

    def tok(t: str) -> list[str]:
        return re.findall(r"\b\w+\b", (t or "").lower())

    q_tokens = Counter(tok(question))
    sims = []
    for text in texts:
        t_tokens = Counter(tok(text))
        common = set(q_tokens) & set(t_tokens)
        if not t_tokens or not common:
            sims.append(0.0)
            continue
        dot = sum(q_tokens[w] * t_tokens[w] for w in common)
        qn = sum(c * c for c in q_tokens.values()) ** 0.5
        tn = sum(c * c for c in t_tokens.values()) ** 0.5
        sims.append(dot / (qn * tn) if qn and tn else 0.0)
    return sims


def evaluate(examples: list[ContrastiveExample], tile_texts: dict[str, str], max_questions: int | None = None) -> tuple[list[int], int]:
    subset = examples if max_questions is None else examples[:max_questions]
    page_cache: dict[str, tuple[list[str], list[str]]] = {}
    ranks: list[int] = []
    skipped = 0

    for ex in subset:
        if ex.page_slug not in page_cache:
            page_cache[ex.page_slug] = _tiles_for_page(ex.page_slug, tile_texts)
        tile_ids, texts = page_cache[ex.page_slug]
        if len(tile_ids) < 2 or ex.positive_tile_id not in tile_ids:
            skipped += 1
            continue

        positive_idx = tile_ids.index(ex.positive_tile_id)
        sims = _rank_by_tfidf(ex.question, texts)
        ranked = sorted(range(len(texts)), key=lambda i: sims[i], reverse=True)
        ranks.append(ranked.index(positive_idx))

    return ranks, skipped


def _report(ranks: list[int], skipped: int, k_values: list[int]) -> None:
    print(f"\n{len(ranks)} questions évaluées, {skipped} ignorées (page < 2 tuiles ou tuile positive introuvable)")
    if not ranks:
        print("Aucun résultat.")
        return
    for k in k_values:
        recall_at_k = sum(1 for r in ranks if r < k) / len(ranks)
        print(f"Recall@{k} = {recall_at_k:.3f}")
    mrr = sum(1 / (r + 1) for r in ranks) / len(ranks)
    print(f"MRR = {mrr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrastive-examples-path", type=str, default=str(CONTRASTIVE_EXAMPLES_PATH))
    parser.add_argument("--tiles-manifest-path", type=str, default=str(TILES_MANIFEST_PATH))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    examples = load_contrastive_examples(Path(args.contrastive_examples_path))
    if not examples:
        raise RuntimeError(f"Aucun exemple contrastif trouvé dans {args.contrastive_examples_path}")
    _, val_examples = split_examples_by_article(examples, val_fraction=args.val_fraction, seed=args.seed)
    val_articles = {_article_root(ex.page_slug) for ex in val_examples}
    print(
        f"[eval_retrieval_tfidf] {len(examples)} exemples au total -> {len(val_examples)} dans le split val "
        f"({len(val_articles)} articles)"
    )

    tile_texts = load_tile_texts(Path(args.tiles_manifest_path))
    if not tile_texts:
        raise RuntimeError(f"Manifeste de tuiles introuvable ou vide : {args.tiles_manifest_path}")

    ranks, skipped = evaluate(val_examples, tile_texts, args.max_questions)
    _report(ranks, skipped, args.k)


if __name__ == "__main__":
    main()
