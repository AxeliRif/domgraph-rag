"""
À lancer en tout premier, sur le GPU réel, avant tout entraînement.

Vérifie que embed_questions/embed_tiles batchés (src/lora_finetune.py)
produisent les MÊMES embeddings qu'un calcul de référence non batché, un
exemple à la fois -- le point d'attention documenté dans
lora_finetune._batched_last_token_hidden_state (padding à gauche + M-RoPE de
Qwen2-VL) n'a jamais été testé contre le vrai modèle avant ce script, faute
d'accès GPU. Coût : quelques appels modèle, moins d'une minute -- largement
rentabilisé si ça évite de lancer un entraînement complet sur une loss
silencieusement calculée à partir du mauvais token.

Usage :
    python3 -m scripts.verify_batching
"""
from __future__ import annotations

from PIL import Image

from src.lora_finetune import _EMBEDDING_PROMPT, embed_questions, embed_tiles, load_reader_model

# Seuil volontairement large : le bfloat16 introduit du bruit numérique même
# entre deux calculs également corrects (arrondi, non-associativité) -- on ne
# cherche pas une égalité exacte, mais à distinguer "bruit numérique normal"
# de "mauvais token récupéré" (qui ferait chuter la similarité bien plus bas).
MIN_COSINE_SIMILARITY = 0.99


def _reference_single_embedding(model, processor, text: str = "", image=None):
    """Calcul de référence, un exemple à la fois -- volontairement
    indépendant de _batched_last_token_hidden_state (ne partage aucun code
    avec lui), pour servir de vérité terrain à ce script plutôt que de
    vérifier le batching contre lui-même."""
    if image is not None:
        content = [{"type": "image", "image": image}, {"type": "text", "text": _EMBEDDING_PROMPT}]
    else:
        content = [{"type": "text", "text": f"{text}\n{_EMBEDDING_PROMPT}"}]
    messages = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[chat_text], images=[image] if image is not None else None, return_tensors="pt"
    ).to(model.device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[-1][:, -1, :]


def _check(name: str, batched, reference) -> None:
    import torch

    max_diff = (batched - reference).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(batched, reference).min().item()
    print(f"  écart max = {max_diff:.6f}, similarité cosinus min = {cos_sim:.6f}")
    if cos_sim < MIN_COSINE_SIMILARITY:
        raise AssertionError(
            f"{name} : le batching ne correspond pas à la référence non batchée "
            f"(similarité {cos_sim:.4f} < seuil {MIN_COSINE_SIMILARITY}). "
            "NE PAS lancer l'entraînement en l'état -- relance-le avec "
            "`python3 -m src.lora_finetune --no-batched-embeddings` (plus lent, "
            "mais chaque appel redevient un batch de taille 1, donc trivialement correct)."
        )
    print(f"  OK -- {name} fidèle à la référence non batchée.")


def main() -> None:
    import torch

    print("Chargement du modèle (peut prendre une minute)...")
    model, processor = load_reader_model()
    model.eval()

    questions = [
        "What year was this article published?",
        "How many tiles are extracted on average per page?",
        "A noticeably longer question than the others, to exercise variable-length left-padding across the batch.",
    ]
    print(f"\n--- Texte ({len(questions)} questions) ---")
    with torch.no_grad():
        batched = embed_questions(model, processor, questions)
        reference = torch.cat(
            [_reference_single_embedding(model, processor, text=q) for q in questions], dim=0
        )
    _check("texte", batched, reference)

    images = [
        Image.new("RGB", (400, 300), color=(200, 50, 50)),
        Image.new("RGB", (900, 1400), color=(50, 200, 50)),  # résolution différente -> nombre de tokens image différent
    ]
    print("\n--- Images (résolutions différentes) ---")
    with torch.no_grad():
        batched = embed_tiles(model, processor, images)
        reference = torch.cat(
            [_reference_single_embedding(model, processor, image=img) for img in images], dim=0
        )
    _check("image", batched, reference)

    print("\nTout est vérifié : le batching est fidèle à la référence. Tu peux lancer l'entraînement en confiance.")


if __name__ == "__main__":
    main()
