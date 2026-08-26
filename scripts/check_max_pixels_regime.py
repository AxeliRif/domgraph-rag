"""
Diagnostic ponctuel (§3.1 de la révision chapitre 2) : le processeur Qwen2-VL
tronque le nombre de tokens visuels d'une tuile au-delà de `max_pixels`. On
vérifie ici la valeur réelle de ce plafond (et de `min_pixels`) et combien de
tokens il produit effectivement pour une tuile DOM moyenne, une bande de
grille, et une tuile DOM maximale -- pour savoir si le ratio de budget pixel
de 2.7x (cf. dom_vs_grid_experiment.py) se transpose en tokens ou tombe à un
ratio plus bas une fois ce plafond appliqué.

Usage : python3 -m scripts.check_max_pixels_regime
"""
from __future__ import annotations

from PIL import Image
from transformers import AutoProcessor

from src.config import HF_MODEL


def main() -> None:
    processor = AutoProcessor.from_pretrained(HF_MODEL)
    ip = processor.image_processor

    # Cette version de `transformers` (5.13.1) expose le plafond via
    # `ip.size` (SizeDict) plutôt que des attributs `min_pixels`/`max_pixels`
    # de premier niveau -- confirmé identique au `preprocessor_config.json`
    # brut du dépôt HF du modèle (pas une valeur par défaut de la classe).
    min_pixels = ip.size.get("shortest_edge")
    max_pixels = ip.size.get("longest_edge")

    print(f"modèle       = {HF_MODEL}")
    print(f"min_pixels   = {min_pixels:,}")
    print(f"max_pixels   = {max_pixels:,}")
    print(f"patch_size   = {ip.patch_size}")
    print(f"merge_size   = {ip.merge_size}")
    print()

    cases = [
        (2048, 2048, "tuile DOM max (patching, MAX_TILE_HEIGHT x RENDER_WIDTH)"),
        (2048, 1024, "bande grille (GRID_TILE_HEIGHT x RENDER_WIDTH)"),
        (620, 254, "tuile DOM moyenne (approx. pixels moyens observés / RENDER_WIDTH)"),
        (875, 1024, "bande grille a la largeur PixelRAG (875x1024)"),
    ]

    for w, h, name in cases:
        out = ip(images=[Image.new("RGB", (w, h))], return_tensors="pt")
        grid_thw = out["image_grid_thw"][0]
        gh, gw = int(grid_thw[1]), int(grid_thw[2])
        n_tokens = (gh * gw) // (ip.merge_size**2)
        raw_pixels = w * h
        capped = raw_pixels > max_pixels
        print(
            f"{name:55s} {w}x{h} ({raw_pixels:>10,} px) -> {n_tokens:5d} tokens "
            f"{'[ECRETE]' if capped else '[non-écrêté]'}"
        )


if __name__ == "__main__":
    main()
