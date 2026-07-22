"""
Phase 1a — Client pour le modèle de vision (VLM) qui lira les tuiles.

Deux backends :
  - "ollama"       : le plus simple pour démarrer -> `ollama pull qwen3-vl`
                     puis rien d'autre à installer (pas de torch/transformers).
  - "transformers" : plus de contrôle (accès aux logits, fine-tuning LoRA en
                     Phase 2), mais nécessite torch + transformers + un GPU
                     correct pour rester confortable avec un modèle 7B.

Le choix est fait via `config.VLM_BACKEND`, ou explicitement à l'instanciation :
    client = VLMClient(backend="ollama")
"""
from __future__ import annotations

import hashlib
import io

from PIL import Image

from .config import CACHE_DIR, HF_MODEL, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_TEMPERATURE, VLM_BACKEND


def _cache_key(*parts: str) -> str:
    """Hash de contenu : les mêmes entrées (même image, même prompt, même
    config de modèle) retombent sur la même clé, y compris d'un run à
    l'autre. Sûr uniquement parce que le décodage est déterministe côté
    Ollama (OLLAMA_TEMPERATURE = 0, cf. config.py) — à revoir si la
    température redevient > 0, ce qui rendrait deux appels identiques non
    reproductibles et donc le cache incorrect."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _image_digest(image: Image.Image) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{image.size}{image.mode}".encode("utf-8"))
    hasher.update(image.tobytes())
    return hasher.hexdigest()


def _cache_get(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _cache_set(key: str, value: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.txt").write_text(value, encoding="utf-8")


class VLMClient:
    def __init__(self, backend: str = VLM_BACKEND):
        if backend not in {"ollama", "transformers"}:
            raise ValueError(f"Backend inconnu : {backend!r} (attendu 'ollama' ou 'transformers')")
        self.backend = backend
        self._model = None
        self._processor = None

    def ask(self, image: Image.Image, question: str, think: bool | None = None) -> str:
        """Envoie une tuile (image PIL) + une question au VLM, retourne sa réponse texte.

        `think` (backend "ollama" uniquement) : passer False pour une tâche
        d'extraction factuelle simple où le raisonnement interne du modèle
        n'apporte rien (cf. qa_generation.py) — sans garantie qu'il soit
        toujours respecté par le modèle (constaté empiriquement), d'où le
        num_ctx élargi côté `_ask_ollama` en filet de sécurité.

        Mise en cache sur disque (data/cache/), transparente : un retry après
        échec réseau ou un re-run du même dataset ne repaie pas les tuiles
        déjà interrogées (cf. build_run.log — chaque appel coûte 100-150s)."""
        key = _cache_key("ask", *self._model_identity(), question, str(think), _image_digest(image))
        cached = _cache_get(key)
        if cached is not None:
            return cached

        result = self._ask_ollama(image, question, think=think) if self.backend == "ollama" else self._ask_transformers(image, question)
        _cache_set(key, result)
        return result

    def ask_text(self, question: str) -> str:
        """Envoie une question texte seule (sans image) au VLM.

        Sert à synthétiser une réponse finale à partir de l'évidence déjà
        extraite tuile par tuile par `ask` (cf. evidence_controller.py), plutôt
        qu'à lire une image. Même mise en cache que `ask`."""
        key = _cache_key("ask_text", *self._model_identity(), question)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        result = self._ask_text_ollama(question) if self.backend == "ollama" else self._ask_text_transformers(question)
        _cache_set(key, result)
        return result

    def _model_identity(self) -> tuple[str, ...]:
        """Composantes de la clé de cache qui identifient la config du modèle
        (backend + nom +, côté Ollama, les réglages de décodage) : si l'un de
        ces réglages change, les anciennes réponses en cache ne doivent pas
        être réutilisées."""
        if self.backend == "ollama":
            return (self.backend, OLLAMA_MODEL, str(OLLAMA_NUM_CTX), str(OLLAMA_TEMPERATURE))
        return (self.backend, HF_MODEL)

    # -- backend Ollama -----------------------------------------------------
    def _ask_ollama(self, image: Image.Image, question: str, think: bool | None = None) -> str:
        import ollama  # import différé : évite la dépendance si non utilisée

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            think=think,
            options={"num_ctx": OLLAMA_NUM_CTX, "temperature": OLLAMA_TEMPERATURE},
            messages=[{"role": "user", "content": question, "images": [buf.getvalue()]}],
        )
        return response["message"]["content"]

    def _ask_text_ollama(self, question: str) -> str:
        import ollama  # import différé : évite la dépendance si non utilisée

        response = ollama.chat(
            model=OLLAMA_MODEL,
            options={"num_ctx": OLLAMA_NUM_CTX, "temperature": OLLAMA_TEMPERATURE},
            messages=[{"role": "user", "content": question}],
        )
        return response["message"]["content"]

    # -- backend Transformers -------------------------------------------------
    def _ask_transformers(self, image: Image.Image, question: str) -> str:
        if self._model is None:
            self._load_transformers_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[chat_text], images=[image], return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _ask_text_transformers(self, question: str) -> str:
        if self._model is None:
            self._load_transformers_model()

        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[chat_text], return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _load_transformers_model(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            HF_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(HF_MODEL)
