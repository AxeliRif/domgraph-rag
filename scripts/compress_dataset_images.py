"""
Compresse les tuiles de `data/qa_dataset/images/` (sous-échantillonnage
Lanczos, cf. `src/image_compression.py`) vers `data/qa_dataset_compressed/images/`,
pour comparer un entraînement/benchmark sur tuiles pleine résolution vs
compressées. `tiles_manifest.jsonl` et `contrastive_examples.jsonl` ne sont
PAS dupliqués : ils sont réutilisés tels quels des deux côtés (cf. docstring
du module `image_compression`), seul `images_root` change.

Usage :
    python3 -m scripts.compress_dataset_images
    python3 -m scripts.compress_dataset_images --scale 0.6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import QA_DATASET_COMPRESSED_DIR, QA_DATASET_DIR, TILE_COMPRESSION_SCALE
from src.image_compression import compress_images_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compresse les tuiles du dataset contrastif par sous-échantillonnage Lanczos."
    )
    parser.add_argument(
        "--scale", type=float, default=TILE_COMPRESSION_SCALE,
        help="Facteur d'échelle linéaire, dans (0, 1). Défaut : config.TILE_COMPRESSION_SCALE.",
    )
    parser.add_argument("--source-dir", type=str, default=str(QA_DATASET_DIR / "images"))
    parser.add_argument("--dest-dir", type=str, default=str(QA_DATASET_COMPRESSED_DIR / "images"))
    args = parser.parse_args()

    source_dir, dest_dir = Path(args.source_dir), Path(args.dest_dir)
    if not source_dir.exists():
        raise SystemExit(f"Dossier source introuvable : {source_dir} -- lance d'abord scripts/build_dataset.py")

    t0 = time.time()
    n = compress_images_dir(source_dir, dest_dir, scale=args.scale)
    print(f"{n} images compressées (scale={args.scale}) de {source_dir} vers {dest_dir} en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
