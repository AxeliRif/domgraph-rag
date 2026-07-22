"""
Phase 2d — Fine-tuning LoRA du VLM lecteur (ViT + LLM dégelés) par contraste.

Nécessite le backend "transformers" (voir requirements.txt, section commentée
Phase 2 : torch, transformers, accelerate, qwen-vl-utils, peft, bitsandbytes).
Ce module n'est pas importé par le reste du pipeline pour ne pas forcer cette
dépendance lourde sur la Phase 1 — il ne charge torch/transformers/peft qu'à
l'exécution (dans `load_reader_model` et `main`).

Idée : `Qwen2VLForConditionalGeneration` est un modèle *génératif* (texte ->
texte), pas un encodeur de similarité. Pour l'entraîner par contraste
(question -> tuile), on le transforme en encodeur en lui faisant produire un
prompt court ("Résume ce qui précède en un mot :") et en prenant l'état caché
du dernier token comme embedding — une technique documentée dans la
littérature récente sur les VLM utilisés comme encodeurs universels (à
vérifier/citer précisément dans le rapport, cf. la note de sourcing du
README). On calcule cet embedding aussi bien pour le texte de la question que
pour chaque image de tuile (même prompt, modalité différente), puis on
applique une loss InfoNCE standard : la tuile positive doit être plus proche
de la question que les négatifs difficiles minés en Phase 2b (et que les
positifs des *autres* questions du batch, utilisés comme négatifs "en
batch", technique usuelle des retrievers denses).

LoRA est appliqué à la fois sur le LLM et sur la tour de vision (ViT) du VLM,
plutôt que sur le LLM seul, car la tâche demande d'affiner à la fois "ce qui
est vu" dans la tuile et "ce qui est demandé" dans la question (cf. README,
qui décrit explicitement "LLM + ViT dégelé").
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .config import (
    CONTRASTIVE_TEMPERATURE,
    HF_MODEL,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_OUTPUT_DIR,
    LORA_R,
    LORA_TARGET_SUFFIXES_LLM,
    LORA_TARGET_SUFFIXES_VIT,
    N_HARD_NEGATIVES,
    QA_DATASET_DIR,
)

# Prompt court partagé par les deux modalités : la même consigne est appliquée
# au texte de la question et à chaque image de tuile, pour que l'embedding
# résultant soit comparable dans le même "espace de résumé en un mot".
_EMBEDDING_PROMPT = "Résume ce qui précède en un mot :"


def discover_lora_target_modules(model, suffixes: list[str]) -> list[str]:
    """Introspecte `model` et retourne les noms complets des sous-modules
    `nn.Linear` dont le nom se termine par l'un de `suffixes`.

    On matche par suffixe (et non par nom complet) car les noms de modules
    internes de Qwen2-VL peuvent varier d'une version de `transformers` à
    l'autre ; matcher dynamiquement évite de casser silencieusement le
    fine-tuning (LoRA appliqué à zéro module) si un nom change.
    """
    import torch.nn as nn

    matches = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in suffixes)
    ]
    if not matches:
        raise RuntimeError(
            f"Aucun module nn.Linear ne correspond aux suffixes {suffixes} — "
            "vérifie les noms réels via `[n for n, _ in model.named_modules()]` "
            "(l'architecture de transformers a peut-être changé)."
        )
    return matches


def load_reader_model(hf_model: str = HF_MODEL, r: int = LORA_R, alpha: int = LORA_ALPHA, dropout: float = LORA_DROPOUT):
    """Charge le VLM lecteur et lui applique LoRA sur le LLM ET la tour de
    vision (cf. docstring du module). Retourne (model, processor)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(hf_model)

    target_modules = discover_lora_target_modules(
        base_model, LORA_TARGET_SUFFIXES_LLM + LORA_TARGET_SUFFIXES_VIT
    )
    print(f"[lora_finetune] LoRA appliqué à {len(target_modules)} modules "
          f"(LLM: {LORA_TARGET_SUFFIXES_LLM}, ViT: {LORA_TARGET_SUFFIXES_VIT})")

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type=None,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def _last_token_hidden_state(model, processor, text: str, image=None):
    """Passe (texte [+ image]) dans le modèle et retourne l'état caché du
    dernier token de la dernière couche — c'est l'embedding utilisé pour la
    loss contrastive (cf. docstring du module)."""
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
    return outputs.hidden_states[-1][:, -1, :]  # (1, hidden_dim)


def embed_questions(model, processor, questions: list[str]):
    """Embedding de chaque question (modalité texte seule)."""
    import torch

    return torch.cat([_last_token_hidden_state(model, processor, q) for q in questions], dim=0)


def embed_tiles(model, processor, images: list) -> "torch.Tensor":  # noqa: F821
    """Embedding de chaque image de tuile (modalité image + prompt fixe)."""
    import torch

    return torch.cat(
        [_last_token_hidden_state(model, processor, "", image=img) for img in images], dim=0
    )


def info_nce_loss(question_embeds, tile_embeds, positive_indices: list[int], temperature: float = CONTRASTIVE_TEMPERATURE):
    """Loss InfoNCE standard : pour chaque question, softmax sur sa similarité
    avec TOUTES les tuiles du batch (sa positive + ses négatifs difficiles +,
    implicitement, les positifs/négatifs des autres questions du batch =
    négatifs "en batch"), cible = l'indice de sa tuile positive.
    """
    import torch
    import torch.nn.functional as F

    question_embeds = F.normalize(question_embeds, dim=-1)
    tile_embeds = F.normalize(tile_embeds, dim=-1)

    similarities = question_embeds @ tile_embeds.T / temperature  # (B, n_tiles)
    targets = torch.tensor(positive_indices, device=similarities.device, dtype=torch.long)
    return F.cross_entropy(similarities, targets)


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-4
    # Doit correspondre au nombre de négatifs réellement minés (cf.
    # config.N_HARD_NEGATIVES, utilisé par hard_negative_mining.py) : un
    # `n_negatives` supérieur au nombre miné par exemple fait filtrer TOUT le
    # dataset par ContrastiveTileDataset (elle n'accepte pas de compléter),
    # laissant train_lora lever un RuntimeError "Dataset contrastif vide" au
    # tout premier lancement -- constaté en préparant un run sur GPU loué.
    n_negatives: int = N_HARD_NEGATIVES
    output_dir: str = str(LORA_OUTPUT_DIR)
    checkpoint_every: int = 1
    # Racine des images de tuiles : pointer vers data/qa_dataset_compressed
    # (cf. scripts/compress_dataset_images.py) plutôt que data/qa_dataset pour
    # entraîner sur des tuiles compressées -- même tiles_manifest.jsonl et
    # contrastive_examples.jsonl des deux côtés (chemins relatifs identiques,
    # cf. image_compression.py), ce qui rend la comparaison pleine résolution
    # vs compressée une simple bascule de ce chemin.
    images_root: str = str(QA_DATASET_DIR)


def train_lora(train_config: TrainConfig = TrainConfig()) -> None:
    """Boucle d'entraînement complète : charge le modèle + LoRA, le dataset
    contrastif (Phase 2a/2b), et optimise la loss InfoNCE. Sauvegarde
    l'adaptateur LoRA (pas le modèle de base) dans `train_config.output_dir`.
    """
    import torch
    from torch.utils.data import DataLoader

    from .contrastive_dataset import ContrastiveTileDataset, contrastive_collate_fn

    model, processor = load_reader_model()
    dataset = ContrastiveTileDataset(n_negatives=train_config.n_negatives, images_root=Path(train_config.images_root))
    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset contrastif vide — lance d'abord qa_generation.generate_qa_dataset "
            "puis hard_negative_mining.mine_hard_negatives."
        )
    loader = DataLoader(
        dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=contrastive_collate_fn
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=train_config.lr
    )

    output_dir = Path(train_config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(train_config.epochs):
        total_loss = 0.0
        for batch in loader:
            question_embeds = embed_questions(model, processor, batch["questions"])
            tile_embeds = embed_tiles(model, processor, batch["tile_images"])
            loss = info_nce_loss(question_embeds, tile_embeds, batch["positive_indices"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[lora_finetune] epoch {epoch + 1}/{train_config.epochs} — loss moyenne = {avg_loss:.4f}")

        if train_config.checkpoint_every > 0 and (epoch + 1) % train_config.checkpoint_every == 0:
            epoch_checkpoint = checkpoint_dir / f"epoch-{epoch + 1}"
            model.save_pretrained(epoch_checkpoint)
            print(f"[lora_finetune] checkpoint sauvegardé dans {epoch_checkpoint}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"[lora_finetune] adaptateur LoRA final sauvegardé dans {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA contrastif du VLM lecteur (Phase 2).")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--n-negatives", type=int, default=TrainConfig.n_negatives)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument(
        "--images-root", type=str, default=TrainConfig.images_root,
        help=(
            "Racine des images de tuiles. Pointer vers data/qa_dataset_compressed "
            "(cf. scripts/compress_dataset_images.py) pour entraîner sur des tuiles "
            "compressées plutôt que pleine résolution."
        ),
    )
    args = parser.parse_args()

    train_lora(TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_negatives=args.n_negatives, output_dir=args.output_dir, images_root=args.images_root,
    ))


if __name__ == "__main__":
    main()
