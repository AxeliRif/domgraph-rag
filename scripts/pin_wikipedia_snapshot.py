"""
Fige la révision Wikipedia de chaque page utilisée par l'évaluation A
(scripts/dom_vs_grid_experiment.py) via l'API MediaWiki (action=query,
prop=revisions), et écrit le mapping url -> oldid dans data/wikipedia_snapshot.json.

Sans ce pin, dom_vs_grid_experiment.py refait un rendu Playwright de l'URL
"live" à chaque exécution ; les articles Wikipedia continuent d'être édités
et grossissent au fil du temps (cf. §5 du papier), ce qui rend le run non
reproductible d'une exécution à l'autre (une page qui tenait dans une seule
unité de travail à une date donnée peut être devenue surdimensionnée plus
tard). Une fois pinné, dom_vs_grid_experiment.py rend systématiquement la
même révision -oldid=N-, indépendamment de la date d'exécution.

Usage :
    python3 -m scripts.pin_wikipedia_snapshot
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.config import QA_PAIRS_PATH

SNAPSHOT_PATH = Path("data/wikipedia_snapshot.json")
API_URL = "https://en.wikipedia.org/w/api.php"


def _candidate_urls() -> list[str]:
    """Les URLs de page utilisées par l'évaluation A : mêmes critères que
    main_async dans dom_vs_grid_experiment.py (pages non découpées en
    sections dans qa_pairs.jsonl)."""
    qa_pairs = [json.loads(line) for line in QA_PAIRS_PATH.open(encoding="utf-8") if line.strip()]
    urls: list[str] = []
    seen = set()
    for qa in qa_pairs:
        if "__sec" in qa["page_slug"]:
            continue
        if qa["page_url"] not in seen:
            seen.add(qa["page_url"])
            urls.append(qa["page_url"])
    return urls


def _title_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    return urllib.parse.unquote(path.removeprefix("/wiki/"))


def _fetch_latest_revid(title: str) -> int:
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvlimit": 1,
        "format": "json",
        "formatversion": 2,
    })
    req = urllib.request.Request(f"{API_URL}?{params}", headers={"User-Agent": "domgraph-rag-eval/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    page = payload["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"page MediaWiki introuvable pour le titre {title!r}")
    return page["revisions"][0]["revid"]


def _save(snapshot: dict[str, dict]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    urls = _candidate_urls()
    snapshot: dict[str, dict] = {}
    if SNAPSHOT_PATH.exists():
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    now = datetime.now(timezone.utc).isoformat()
    failures: list[str] = []
    for i, url in enumerate(urls, 1):
        if url in snapshot:
            print(f"[{i}/{len(urls)}] {_title_from_url(url)} — déjà pinné (oldid={snapshot[url]['revid']}), skip", flush=True)
            continue
        title = _title_from_url(url)
        print(f"[{i}/{len(urls)}] {title}...", flush=True)
        try:
            revid = _fetch_latest_revid(title)
        except urllib.error.HTTPError as exc:
            print(f"  ÉCHEC ({exc}), on garde ce qui est déjà pinné et on continue", flush=True)
            failures.append(url)
            time.sleep(5)
            continue
        pinned_url = f"https://en.wikipedia.org/w/index.php?title={urllib.parse.quote(title)}&oldid={revid}"
        snapshot[url] = {"revid": revid, "pinned_url": pinned_url, "pinned_at": now}
        print(f"  oldid={revid}", flush=True)
        _save(snapshot)  # progrès persisté à chaque page -- un run de plusieurs
        # heures sur un gros corpus peut sinon perdre tout son travail sur un
        # 429 tardif (cf. même raisonnement que RETRY_DELAYS_S dans build_dataset.py)
        time.sleep(1)  # espacer les requêtes -- l'API MediaWiki a rate-limité un burst sans délai

    print(f"\n{len(snapshot)}/{len(urls)} révisions pinnées dans {SNAPSHOT_PATH}")
    if failures:
        print(f"{len(failures)} page(s) non pinnée(s), relancer le script pour réessayer : {failures}")


if __name__ == "__main__":
    main()
