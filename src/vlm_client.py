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

import io

from PIL import Image

from .config import HF_MODEL, OLLAMA_MODEL, VLM_BACKEND


class VLMClient:
    def __init__(self, backend: str = VLM_BACKEND):
        if backend not in {"ollama", "transformers"}:
            raise ValueError(f"Backend inconnu : {backend!r} (attendu 'ollama' ou 'transformers')")
        self.backend = backend
        self._model = None
        self._processor = None

    def ask(self, image: Image.Image, question: str) -> str:
        """Envoie une tuile (image PIL) + une question au VLM, retourne sa réponse texte."""
        if self.backend == "ollama":
            return self._ask_ollama(image, question)
        return self._ask_transformers(image, question)

    def ask_text(self, question: str) -> str:
        """Envoie une question texte seule (sans image) au VLM.

        Sert à synthétiser une réponse finale à partir de l'évidence déjà
        extraite tuile par tuile par `ask` (cf. evidence_controller.py), plutôt
        qu'à lire une image."""
        if self.backend == "ollama":
            return self._ask_text_ollama(question)
        return self._ask_text_transformers(question)

    # -- backend Ollama -----------------------------------------------------
    def _ask_ollama(self, image: Image.Image, question: str) -> str:
        import ollama  # import différé : évite la dépendance si non utilisée

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": question, "images": [buf.getvalue()]}],
        )
        return response["message"]["content"]

    def _ask_text_ollama(self, question: str) -> str:
        import ollama  # import différé : évite la dépendance si non utilisée

        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": question}])
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
