"""
Phase 4 (ablation) — Compression des tuiles par sous-échantillonnage Lanczos.

PixelRAG (arXiv:2606.28344) documente la résolution d'image comme un levier
d'efficacité pour réduire le coût en tokens du VLM ("up to 3x token cost
reduction at lower resolutions while maintaining accuracy"), sans préciser le
filtre de ré-échantillonnage ni les résolutions testées -- ni le papier ni le
dépôt public (github.com/StarTrail-org/PixelRAG) ne le détaillent. On retient
Lanczos, le filtre de downscale haute qualité standard (celui recommandé par
Pillow pour réduire une image sans aliasing perceptible), pour produire une
version compressée du dataset de tuiles déjà généré, comparable côte à côte
avec la version pleine résolution.

`compress_images_dir` écrit les images compressées sous les mêmes noms de
fichier, dans un dossier séparé (miroir de `data/qa_dataset/images/`) :
`tiles_manifest.jsonl` et `contrastive_examples.jsonl` restent inchangés et
réutilisables tels quels des deux côtés (leurs chemins relatifs stockés
n'encodent pas la résolution), seul `images_root` change entre un
entraînement/benchmark sur tuiles pleine résolution et un sur tuiles
compressées (cf. `ContrastiveTileDataset`, `lora_finetune.TrainConfig.images_root`).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import TILE_COMPRESSION_SCALE


def compress_image(image: Image.Image, scale: float = TILE_COMPRESSION_SCALE) -> Image.Image:
    """Réduit `image` par le facteur d'échelle linéaire `scale` (ex. 0.5 =
    moitié moins large et moins haute, donc environ 4x moins de pixels) via un
    ré-échantillonnage Lanczos."""
    if not 0 < scale < 1:
        raise ValueError(f"scale doit être dans l'intervalle (0, 1), reçu {scale}")
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def compress_images_dir(source_dir: Path, dest_dir: Path, scale: float = TILE_COMPRESSION_SCALE) -> int:
    """Compresse tous les `.png` de `source_dir` (à plat, comme
    `data/qa_dataset/images/`) vers `dest_dir`, sous le même nom de fichier.
    Retourne le nombre d'images traitées."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source_dir.glob("*.png")):
        with Image.open(path) as img:
            compressed = compress_image(img.convert("RGB"), scale=scale)
            compressed.save(dest_dir / path.name, format="PNG")
        count += 1
    return count
