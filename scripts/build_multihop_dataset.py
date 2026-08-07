"""
Construit un jeu de données multi-hop (data/qa_dataset/multihop_examples.jsonl)
à partir d'un échantillon de HotpotQA (Yang et al. 2018, arXiv:1809.09600), en
résolvant chaque question contre les DEUX articles Wikipédia qu'elle cite via
le pipeline DOM-Graph RAG existant (src/multihop_qa.py) -- cf. README/paper,
section Limitations ("Single-hop training and evaluation data cannot
exercise the graph").

Nécessite le paquet `datasets` (HuggingFace) pour streamer HotpotQA sans en
télécharger l'intégralité (cf. requirements.txt).

Usage :
    python3 -m scripts.build_multihop_dataset --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from src.config import QA_DATASET_DIR
from src.multihop_qa import ArticleUnavailableError, build_multihop_example

MULTIHOP_EXAMPLES_PATH = QA_DATASET_DIR / "multihop_examples.jsonl"
# Journal de TOUTE question tentée (succès ou échec), distinct de
# `multihop_examples.jsonl` qui ne contient que les succès : sans lui, une
# relance saute les questions déjà réussies mais retente indéfiniment celles
# déjà tentées-et-échouées (article introuvable / phrase non rattachée),
# puisque leur id n'apparaît jamais dans `multihop_examples.jsonl` -- ce sont
# des échecs déterministes (même page, même phrase de support), donc les
# retenter ne les fait jamais réussir, et retarde d'autant l'atteinte de
# questions réellement neuves plus loin dans le flux HotpotQA.
MULTIHOP_ATTEMPTED_IDS_PATH = QA_DATASET_DIR / "multihop_attempted_ids.jsonl"


def load_hotpotqa(split: str, limit: int, exclude_ids: set[str]) -> list[dict]:
    """Streame les `limit` premiers enregistrements NON déjà présents dans
    `exclude_ids` -- sans ça, relancer ce script pour scaler le dataset (cf.
    `main_async`) restreame depuis le tout début du dataset HotpotQA et
    retente (ou, pour les succès, duplique en écriture "a") des questions
    déjà traitées lors d'un run précédent, le même bug de fond que celui
    corrigé dans `qa_generation.append_tiles_manifest`."""
    from datasets import load_dataset  # import différé : dépendance optionnelle (cf. requirements.txt)

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    records: list[dict] = []
    for record in dataset:
        if record["id"] in exclude_ids:
            continue
        records.append(record)
        if len(records) >= limit:
            break
    return records


async def main_async(limit: int, split: str) -> None:
    # Union des deux journaux : un succès n'est écrit que dans
    # MULTIHOP_EXAMPLES_PATH (jamais dans MULTIHOP_ATTEMPTED_IDS_PATH avant
    # cette relecture), donc se limiter à ce dernier laisserait passer à
    # nouveau tout succès antérieur à l'introduction de ce fichier -- bug
    # constaté en pratique : un premier run après ce correctif a reconstruit
    # en double les 24 succès du tout premier batch, faute de cette union.
    attempted_ids: set[str] = set()
    if MULTIHOP_EXAMPLES_PATH.exists():
        with MULTIHOP_EXAMPLES_PATH.open(encoding="utf-8") as f:
            attempted_ids |= {json.loads(line)["id"] for line in f if line.strip()}
    if MULTIHOP_ATTEMPTED_IDS_PATH.exists():
        with MULTIHOP_ATTEMPTED_IDS_PATH.open(encoding="utf-8") as f:
            attempted_ids |= {json.loads(line)["id"] for line in f if line.strip()}

    print(
        f"Chargement de {limit} nouvelle(s) question(s) HotpotQA ({split}, streaming) -- "
        f"{len(attempted_ids)} déjà tentées (succès ou échec) ignorées...",
        flush=True,
    )
    records = load_hotpotqa(split, limit, attempted_ids)

    MULTIHOP_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_built = 0
    n_article_unavailable = 0
    n_support_not_matched = 0
    with MULTIHOP_EXAMPLES_PATH.open("a", encoding="utf-8") as f, \
         MULTIHOP_ATTEMPTED_IDS_PATH.open("a", encoding="utf-8") as attempted_f:
        for i, record in enumerate(records, 1):
            titles = list(dict.fromkeys(record["supporting_facts"]["title"]))
            print(f"[{i}/{len(records)}] {record['id']} ({record['type']}, {' + '.join(titles)})...", flush=True)
            attempted_f.write(json.dumps({"id": record["id"]}) + "\n")
            attempted_f.flush()
            try:
                example = await build_multihop_example(record)
            except ArticleUnavailableError as exc:
                n_article_unavailable += 1
                print(f"  abandonné -- article introuvable ({exc})", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001 - une question individuelle : on continue sur les suivantes
                print(f"  ÉCHEC ({exc!r}), question ignorée", flush=True)
                continue
            if example is None:
                n_support_not_matched += 1
                print("  abandonné -- phrase de support non rattachée à une tuile du rendu actuel", flush=True)
                continue
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            n_built += 1
            n_pages = len({slug for slug, _ in example.positive_tiles})
            print(
                f"  OK -- {len(example.positive_tiles)} tuile(s) positive(s) sur {n_pages} page(s), "
                f"{len(example.hard_negative_tiles)} négatif(s) minés",
                flush=True,
            )

    print(
        f"\nTerminé : {n_built}/{len(records)} exemples multi-hop construits dans {MULTIHOP_EXAMPLES_PATH}\n"
        f"  abandons -- article introuvable: {n_article_unavailable}, "
        f"phrase de support non rattachée: {n_support_not_matched}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit un jeu de données multi-hop à partir de HotpotQA.")
    parser.add_argument("--limit", type=int, default=20, help="Nombre de questions HotpotQA à tenter.")
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()
    asyncio.run(main_async(args.limit, args.split))


if __name__ == "__main__":
    main()
