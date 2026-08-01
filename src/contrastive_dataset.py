"""
Phase 2c — Dataset PyTorch pour l'entraînement contrastif question -> tuile.

Assemble les trois fichiers produits par les étapes précédentes :
  - tiles_manifest.jsonl        (Phase 2a, qa_generation.py)   tile_id -> image
  - qa_pairs.jsonl / contrastive_examples.jsonl (Phase 2a/2b)  question, positif, négatifs

Chaque exemple retourné contient une question et une liste d'images de tuiles
(1 positive + K négatives difficiles) — le modèle est entraîné à donner à la
question une similarité plus grande avec la tuile positive qu'avec les autres
(cf. lora_finetune.py pour la loss InfoNCE qui exploite ce format).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .config import CONTRASTIVE_EXAMPLES_PATH, QA_DATASET_DIR, TILES_MANIFEST_PATH
from .hard_negative_mining import ContrastiveExample, load_contrastive_examples


def _article_root(page_slug: str) -> str:
    """"Mount_Everest__sec6" -> "Mount_Everest" : les `page_slug` de ce jeu de
    données sont des *sections* d'articles Wikipédia, pas des articles entiers
    (cf. build_dataset.py). Regrouper par article plutôt que par section évite
    la fuite d'un split train/val : deux sections du même article partagent
    souvent du vocabulaire, des entités et parfois des tuiles quasi-identiques
    (infobox répétée, résumé en tête de section)."""
    return page_slug.split("__sec")[0]


def split_examples_by_article(
    examples: list[ContrastiveExample], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[ContrastiveExample], list[ContrastiveExample]]:
    """Split (train, val) par ARTICLE (cf. `_article_root`), pas par exemple ni
    par section : sans ça, le modèle a pu voir d'autres questions sur les
    mêmes tuiles (ou des tuiles très proches) pendant l'entraînement, ce qui
    gonflerait artificiellement le recall/MRR mesurés sur le split "val".

    Les articles sont mélangés (seed fixe = reproductible) puis ajoutés au
    split val un par un jusqu'à couvrir `val_fraction` des exemples -- une
    approximation par article plutôt qu'un pourcentage exact par exemple,
    puisque les articles ont des nombres d'exemples très inégaux (cf. la
    distribution constatée sur ce jeu de données : de 1 à 25 exemples par
    section).
    """
    examples_by_article: dict[str, list[ContrastiveExample]] = {}
    for ex in examples:
        examples_by_article.setdefault(_article_root(ex.page_slug), []).append(ex)

    articles = list(examples_by_article)
    random.Random(seed).shuffle(articles)

    target_val_count = round(len(examples) * val_fraction)
    val: list[ContrastiveExample] = []
    train: list[ContrastiveExample] = []
    for article in articles:
        if len(val) < target_val_count:
            val.extend(examples_by_article[article])
        else:
            train.extend(examples_by_article[article])
    return train, val


def load_tile_image_paths(manifest_path: Path = TILES_MANIFEST_PATH) -> dict[str, str]:
    """Charge le manifeste des tuiles, retourne {tile_id: chemin_image_relatif}.

    Attention : les `tile_id` (ex. "tile_0000") ne sont uniques qu'au sein d'une
    page — si le manifeste couvre plusieurs pages, la clé effective utilisée
    ici est `page_slug + "::" + tile_id` pour éviter les collisions.
    """
    if not manifest_path.exists():
        return {}
    paths: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = f"{record['page_slug']}::{record['tile_id']}"
            paths[key] = record["image_path"]
    return paths


@dataclass
class ContrastiveBatchItem:
    question: str
    answer: str
    positive_image: Image.Image
    negative_images: list[Image.Image]


class ContrastiveTileDataset(Dataset):
    """Un exemple = 1 question + 1 image positive + K images négatives (PIL, RGB).

    Les exemples dont la tuile positive ou une tuile négative n'a pas
    d'image correspondante dans le manifeste sont ignorés silencieusement à la
    construction (ex. manifeste et exemples générés à des runs différents).
    """

    def __init__(
        self,
        examples: list[ContrastiveExample] | None = None,
        contrastive_examples_path: Path = CONTRASTIVE_EXAMPLES_PATH,
        tiles_manifest_path: Path = TILES_MANIFEST_PATH,
        images_root: Path = QA_DATASET_DIR,
        n_negatives: int | None = None,
    ):
        self.images_root = images_root
        self._image_paths = load_tile_image_paths(tiles_manifest_path)

        examples = examples if examples is not None else load_contrastive_examples(contrastive_examples_path)
        self.examples = [ex for ex in examples if self._is_resolvable(ex)]

        if n_negatives is not None:
            # Uniformise le nombre de négatifs par exemple (utile pour un collate_fn
            # simple qui empile des tenseurs de forme fixe) en tronquant ceux qui en
            # ont davantage et en écartant ceux qui n'en ont pas assez.
            self.examples = [
                ContrastiveExample(**{**ex.__dict__, "hard_negative_tile_ids": ex.hard_negative_tile_ids[:n_negatives]})
                for ex in self.examples
                if len(ex.hard_negative_tile_ids) >= n_negatives
            ]

    def _is_resolvable(self, ex: ContrastiveExample) -> bool:
        keys = [f"{ex.page_slug}::{ex.positive_tile_id}"] + [
            f"{ex.page_slug}::{tid}" for tid in ex.hard_negative_tile_ids
        ]
        return all(key in self._image_paths for key in keys)

    def _load(self, page_slug: str, tile_id: str) -> Image.Image:
        rel_path = self._image_paths[f"{page_slug}::{tile_id}"]
        return Image.open(self.images_root / rel_path).convert("RGB")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ContrastiveBatchItem:
        ex = self.examples[idx]
        return ContrastiveBatchItem(
            question=ex.question,
            answer=ex.answer,
            positive_image=self._load(ex.page_slug, ex.positive_tile_id),
            negative_images=[self._load(ex.page_slug, tid) for tid in ex.hard_negative_tile_ids],
        )


def contrastive_collate_fn(batch: list[ContrastiveBatchItem]) -> dict:
    """Aplatit un batch de B exemples (1 positif + K négatifs chacun, K fixe au
    sein du batch) en une liste de B questions et une liste de B*(1+K) images,
    ordonnée [pos_0, neg_0_0, ..., neg_0_{K-1}, pos_1, neg_1_0, ...].

    `positive_indices[i]` donne l'indice, dans `tile_images`, de la tuile
    positive de la question i — c'est la cible de la loss InfoNCE
    (cf. lora_finetune.py::info_nce_loss).
    """
    questions = [item.question for item in batch]
    tile_images: list[Image.Image] = []
    positive_indices: list[int] = []

    for item in batch:
        positive_indices.append(len(tile_images))
        tile_images.append(item.positive_image)
        tile_images.extend(item.negative_images)

    return {
        "questions": questions,
        "tile_images": tile_images,
        "positive_indices": positive_indices,
    }
