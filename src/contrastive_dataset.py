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
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .config import CONTRASTIVE_EXAMPLES_PATH, QA_DATASET_DIR, TILES_MANIFEST_PATH
from .hard_negative_mining import ContrastiveExample, load_contrastive_examples


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
