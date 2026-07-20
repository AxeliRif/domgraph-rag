"""
Site de démo (Phase 3) : formulaire "URL + question" -> pipeline complet ->
réponse du VLM + graphe interactif + liens Wikipedia à explorer.

Lancement (depuis la racine du projet, avec le venv du projet activé) :
    uvicorn webapp.backend.main:app --reload
puis ouvrir http://127.0.0.1:8000/.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from webapp.backend.pipeline import answer_question  # noqa: E402

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="DOM-Graph RAG — démo")


class AskRequest(BaseModel):
    url: str
    question: str


@app.post("/api/ask")
async def ask(payload: AskRequest) -> dict:
    if not payload.url.strip() or not payload.question.strip():
        raise HTTPException(status_code=400, detail="L'URL et la question sont obligatoires.")

    try:
        result = await answer_question(payload.url, payload.question)
    except Exception as exc:  # renvoyé tel quel au frontend pour affichage dans le bandeau d'erreur
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "page_url": result.page_url,
        "question": result.question,
        "answer": result.answer,
        "graph": result.graph,
        "external_links": [asdict(link) for link in result.external_links],
        "log": result.log,
    }


# Servi en dernier : StaticFiles(html=True) sert index.html sur "/" et intercepterait
# sinon les routes API déclarées après lui.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
