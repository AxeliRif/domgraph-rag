"""
Phase 2a — Génération synthétique de paires question/réponse par tuile.

Pour entraîner le VLM lecteur par contraste (une question doit "retrouver" la
bonne tuile parmi d'autres), il faut un jeu de données (question, tuile
positive). On le génère automatiquement : le VLM lui-même lit chaque tuile et
pose une question dont la réponse n'est visible que là (cf. QA_GENERATION_PROMPT
dans config.py). C'est la même idée que la génération de données synthétiques
utilisée pour entraîner des retrievers denses (ex. InPars, Promptagator) —
adaptée ici à des tuiles image plutôt qu'à des passages de texte.

Toutes les tuiles (pas seulement celles avec une QA générée) sont sauvegardées
sur disque via `save_tile_images` : le hard-negative mining (Phase 2b) et le
dataset d'entraînement (Phase 2c) ont besoin de pouvoir charger l'image de
n'importe quelle tuile, y compris celles qui serviront de négatif plutôt que
de positif.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import QA_DATASET_DIR, QA_GENERATION_PROMPT, QA_IMAGES_DIR, QA_PAIRS_PATH, TILES_MANIFEST_PATH
from .tiling import Tile
from .vlm_client import VLMClient


@dataclass
class QAPair:
    """Un exemple positif pour l'entraînement contrastif : une question dont la
    réponse se trouve dans la tuile `tile_id` (et dans elle seule, en théorie)."""
    id: str
    page_url: str
    page_slug: str
    tile_id: str
    question: str
    answer: str
    image_path: str  # chemin (relatif à QA_DATASET_DIR) vers le crop PNG de la tuile positive


def save_tile_images(tiles: list[Tile], page_slug: str, images_dir: Path = QA_IMAGES_DIR) -> dict[str, Path]:
    """Sauvegarde le crop de chaque tuile sur disque, retourne {tile_id: chemin}.

    Fait pour TOUTES les tuiles, pas seulement celles retenues comme positif —
    le hard-negative mining a besoin de charger l'image de n'importe quelle
    tuile de la page, même celle qui n'a pas généré de question.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for tile in tiles:
        path = images_dir / f"{page_slug}__{tile.id}.png"
        if not path.exists():
            tile.image.save(path, format="PNG")
        paths[tile.id] = path
    return paths


def append_tiles_manifest(
    tiles: list[Tile],
    page_url: str,
    page_slug: str,
    image_paths: dict[str, Path],
    manifest_path: Path = TILES_MANIFEST_PATH,
    images_root: Path = QA_DATASET_DIR,
) -> None:
    """Ajoute une ligne JSONL par tuile : {tile_id, page_url, page_slug, image_path,
    dom_tag, text_preview}. Sert de table de correspondance tile_id -> image pour
    le hard-negative mining et le Dataset d'entraînement (une tuile peut être
    référencée comme négatif sans avoir de QAPair associée).

    `image_path` est stocké relatif à `images_root` (le dossier racine du
    dataset, cf. ContrastiveTileDataset), pas au dossier `images/` lui-même —
    c'est ce qui permet à `image_paths` de venir de n'importe quel dossier
    (utile en test, ou si `generate_qa_dataset` est appelé avec un
    `output_dir` custom)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        for tile in tiles:
            record = {
                "tile_id": tile.id,
                "page_url": page_url,
                "page_slug": page_slug,
                "image_path": str(image_paths[tile.id].relative_to(images_root)),
                "dom_tag": tile.dom_tag,
                "text_preview": tile.text_preview,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_qa_response(raw: str) -> tuple[str, str] | None:
    """Parse la réponse du VLM en (question, answer), ou None si la tuile est à
    ignorer (réponse "SKIP") ou si le JSON est invalide.

    Le VLM enveloppe parfois le JSON dans un bloc markdown ```json ... ``` ou
    ajoute une phrase avant/après malgré la consigne — on essaie donc d'abord
    un parse strict, puis on retombe sur une extraction par regex du premier
    objet JSON `{...}` trouvé dans la réponse.
    """
    text = raw.strip()
    if text.upper().startswith("SKIP"):
        return None

    for candidate in (text, _strip_markdown_fence(text)):
        try:
            data = json.loads(candidate)
            question, answer = data.get("question", "").strip(), data.get("answer", "").strip()
            if question and answer:
                return question, answer
        except (json.JSONDecodeError, AttributeError):
            continue

    match = re.search(r'\{[^{}]*"question"[^{}]*\}', text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            question, answer = data.get("question", "").strip(), data.get("answer", "").strip()
            if question and answer:
                return question, answer
        except json.JSONDecodeError:
            pass

    return None


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else text


def generate_qa_pair_for_tile(
    tile: Tile,
    client: VLMClient,
    page_url: str,
    page_slug: str,
    image_paths: dict[str, Path],
    prompt: str = QA_GENERATION_PROMPT,
    images_root: Path = QA_DATASET_DIR,
) -> QAPair | None:
    """Interroge le VLM sur une tuile, retourne une QAPair ou None si la tuile a
    été jugée inexploitable (pas assez d'information propre à elle seule)."""
    raw_response = client.ask(tile.image, prompt, think=False)
    parsed = _parse_qa_response(raw_response)
    if parsed is None:
        return None
    question, answer = parsed
    return QAPair(
        id=f"{page_slug}__{tile.id}__qa",
        page_url=page_url,
        page_slug=page_slug,
        tile_id=tile.id,
        question=question,
        answer=answer,
        image_path=str(image_paths[tile.id].relative_to(images_root)),
    )


def generate_qa_dataset(
    tiles: list[Tile],
    page_url: str,
    page_slug: str,
    client: VLMClient | None = None,
    output_dir: Path = QA_DATASET_DIR,
) -> list[QAPair]:
    """Pipeline complet Phase 2a pour une page : sauvegarde les crops de toutes
    les tuiles, interroge le VLM tuile par tuile, persiste le résultat dans
    `qa_pairs.jsonl` (append) et retourne la liste des QAPair générées.
    """
    client = client or VLMClient()
    images_dir = output_dir / "images"
    manifest_path = output_dir / "tiles_manifest.jsonl"
    qa_pairs_path = output_dir / "qa_pairs.jsonl"

    image_paths = save_tile_images(tiles, page_slug, images_dir)
    append_tiles_manifest(tiles, page_url, page_slug, image_paths, manifest_path, images_root=output_dir)

    qa_pairs: list[QAPair] = []
    qa_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_pairs_path.open("a", encoding="utf-8") as f:
        for tile in tiles:
            qa = generate_qa_pair_for_tile(tile, client, page_url, page_slug, image_paths, images_root=output_dir)
            if qa is None:
                continue
            qa_pairs.append(qa)
            f.write(json.dumps(asdict(qa), ensure_ascii=False) + "\n")

    return qa_pairs


def load_qa_pairs(path: Path = QA_PAIRS_PATH) -> list[QAPair]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [QAPair(**json.loads(line)) for line in f if line.strip()]
