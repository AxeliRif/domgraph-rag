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
from concurrent.futures import ThreadPoolExecutor
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


def _existing_manifest_keys(manifest_path: Path) -> set[tuple[str, str]]:
    """Lit `manifest_path` et retourne l'ensemble des (page_slug, tile_id) déjà
    présents, pour rendre `append_tiles_manifest` idempotent (cf. son
    docstring)."""
    if not manifest_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            keys.add((d["page_slug"], d["tile_id"]))
    return keys


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
    `images_root` custom).

    Idempotent par (page_slug, tile_id) : si `generate_qa_dataset` est
    rappelé sur une unité déjà (partiellement) écrite -- ex. un run interrompu
    entre cet appel et le check-point de `scripts/build_dataset.py`, qui ne
    marque une unité "faite" qu'après le mining, pas après ce seul appel --,
    les tuiles déjà présentes ne sont pas réécrites. Sans ça, `qa_pairs.jsonl`
    / `tiles_manifest.jsonl` accumulent des doublons à chaque reprise, alors
    que `contrastive_examples.jsonl` (check-pointé, lui) reste propre --
    constaté en pratique sur plusieurs pages du corpus."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_manifest_keys(manifest_path)
    with manifest_path.open("a", encoding="utf-8") as f:
        for tile in tiles:
            if (page_slug, tile.id) in existing:
                continue
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
    images_root: Path = QA_DATASET_DIR,
    max_workers: int = 4,
) -> list[QAPair]:
    """Pipeline complet Phase 2a pour une page : sauvegarde les crops de toutes
    les tuiles, interroge le VLM sur chaque tuile, persiste le résultat dans
    `qa_pairs.jsonl` (append) et retourne la liste des QAPair générées.

    Les tuiles sont interrogées en parallèle (thread pool, `max_workers`
    requêtes en vol) plutôt qu'une par une : chaque appel ne porte que sur sa
    propre tuile, donc aucune coordination n'est nécessaire entre elles, et
    Ollama sert nativement plusieurs requêtes concurrentes. `max_workers=4`
    correspond au nombre de requêtes parallèles par défaut d'Ollama
    (OLLAMA_NUM_PARALLEL) -- au-delà, les requêtes en surplus attendraient de
    toute façon derrière les 4 premières côté serveur.

    `ThreadPoolExecutor.map` renvoie les résultats dans l'ordre des tuiles en
    entrée (même si elles terminent dans un ordre différent), donc
    `qa_pairs.jsonl` reste écrit dans l'ordre de lecture des tuiles, comme en
    séquentiel.

    Idempotent par id de QAPair (`{page_slug}__{tile_id}__qa`) : les tuiles
    dont l'id est déjà présent dans `qa_pairs.jsonl` ne sont ni réinterrogées
    (pas d'appel VLM superflu) ni réécrites -- leur QAPair existante est
    réutilisée telle quelle dans la valeur de retour. Nécessaire pour la même
    raison qu'`append_tiles_manifest` : `scripts/build_dataset.py` ne marque
    une unité "faite" qu'après le mining qui suit cet appel, donc une reprise
    après interruption rappelle `generate_qa_dataset` sur une unité déjà
    (partiellement) écrite."""
    client = client or VLMClient()
    images_dir = images_root / "images"
    manifest_path = images_root / "tiles_manifest.jsonl"
    qa_pairs_path = images_root / "qa_pairs.jsonl"

    image_paths = save_tile_images(tiles, page_slug, images_dir)
    append_tiles_manifest(tiles, page_url, page_slug, image_paths, manifest_path, images_root=images_root)

    existing_by_id = {qa.id: qa for qa in load_qa_pairs(qa_pairs_path) if qa.page_slug == page_slug}
    tiles_to_generate = [t for t in tiles if f"{page_slug}__{t.id}__qa" not in existing_by_id]

    def _generate(tile: Tile) -> QAPair | None:
        return generate_qa_pair_for_tile(tile, client, page_url, page_slug, image_paths, images_root=images_root)

    newly_written: dict[str, QAPair] = {}
    qa_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_pairs_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(tiles_to_generate) or 1))) as executor:
            for qa in executor.map(_generate, tiles_to_generate):
                if qa is None:
                    continue
                newly_written[qa.id] = qa
                f.write(json.dumps(asdict(qa), ensure_ascii=False) + "\n")

    return [
        existing_by_id[qid] if qid in existing_by_id else newly_written[qid]
        for tile in tiles
        for qid in [f"{page_slug}__{tile.id}__qa"]
        if qid in existing_by_id or qid in newly_written
    ]


def load_qa_pairs(path: Path = QA_PAIRS_PATH) -> list[QAPair]:
    """Relit le JSONL écrit par `generate_qa_dataset` ([] si `path` n'existe pas encore)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [QAPair(**json.loads(line)) for line in f if line.strip()]
