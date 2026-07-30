"""
Phase 2d — Fine-tuning LoRA du VLM lecteur (ViT + LLM dégelés) par contraste.

Nécessite le backend "transformers" (voir requirements.txt, section Phase 2 :
torch, transformers, accelerate, qwen-vl-utils, peft ; bitsandbytes optionnel,
utile sur GPU CUDA < 16 Go de VRAM).
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


def load_reader_model(
    hf_model: str = HF_MODEL, r: int = LORA_R, alpha: int = LORA_ALPHA, dropout: float = LORA_DROPOUT,
    gradient_checkpointing: bool = False,
):
    """Charge le VLM lecteur et lui applique LoRA sur le LLM ET la tour de
    vision (cf. docstring du module). Retourne (model, processor).

    `gradient_checkpointing` : réduit la mémoire d'activations au prix d'un
    recalcul partiel au backward (~20-30% plus lent) -- filet de sécurité si
    le GPU obtenu a moins de VRAM que prévu (cf. Appendix B du papier, qui
    recommande 80 Go ou ce flag sur une carte à 40 Go)."""
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
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # nécessaire avec LoRA : les embeddings d'entrée sont gelés,
        # sans quoi gradient checkpointing casse la rétropropagation faute de tenseur d'entrée demandant un gradient.
        print("[lora_finetune] gradient checkpointing activé (mémoire réduite, ~20-30% plus lent)")
    model.print_trainable_parameters()
    return model, processor


def _build_messages(texts: list[str] | None, images: list | None) -> list[list[dict]]:
    if images is not None:
        return [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": _EMBEDDING_PROMPT}]}]
            for img in images
        ]
    return [
        [{"role": "user", "content": [{"type": "text", "text": f"{t}\n{_EMBEDDING_PROMPT}"}]}]
        for t in texts
    ]


def _sequential_last_token_hidden_state(model, processor, texts: list[str] | None = None, images: list | None = None):
    """Un appel modèle par élément (texte OU image), jamais batché -- lent
    (cf. Appendix B du papier : 16 appels séquentiels par étape à
    batch_size=4/n_negatives=2) mais sans aucune hypothèse sur la façon dont
    Qwen2-VL gère le padding : chaque séquence est son propre batch de
    taille 1, donc `hidden_states[:, -1, :]` est trivialement son dernier
    token réel. Filet de sécurité si `scripts/verify_batching.py` échoue sur
    _batched_last_token_hidden_state (cf. son docstring) -- passer
    `TrainConfig.batched_embeddings=False` pour l'utiliser."""
    embeds = []
    for m in _build_messages(texts, images):
        chat_text = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        image = m[0]["content"][0]["image"] if images is not None else None
        inputs = processor(
            text=[chat_text], images=[image] if image is not None else None, return_tensors="pt"
        ).to(model.device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        embeds.append(outputs.hidden_states[-1][:, -1, :])
    import torch

    return torch.cat(embeds, dim=0)


def _batched_last_token_hidden_state(model, processor, texts: list[str] | None = None, images: list | None = None):
    """Calcule l'embedding (dernier token, dernière couche) d'un batch
    d'entrées EN UN SEUL forward pass, plutôt qu'un par un -- le changement à
    plus fort impact identifié avant de louer du GPU (cf. Appendix B du
    papier : la version séquentielle fait 16 appels modèle par étape à
    batch_size=4/n_negatives=2, contre 2 ici, un pour les questions, un pour
    les tuiles).

    Exactement une des deux (`texts`, `images`) doit être fournie, jamais les
    deux (comme dans les usages existants `embed_questions`/`embed_tiles`).

    ATTENTION -- vérifié empiriquement avant d'écrire cette fonction, sur un
    petit modèle causal texte-seul (`tiny-random-gpt2`) : un padding à
    gauche SEUL ne suffit PAS à faire pointer `hidden_states[:, -1, :]` sur
    le bon token -- sans position_ids explicites dérivés du masque
    d'attention, la similarité cosinus contre un calcul de référence non
    batché tombe à ~0.51 (cassé), contre ~0.9999997 une fois les
    position_ids correctement calculés. Qwen2-VL utilise un encodage de
    position M-RoPE (3 axes temporel/hauteur/largeur, pas une simple suite
    1D), plus complexe à répliquer à la main sans risquer une nouvelle
    version silencieusement incorrecte de ce même bug -- on laisse donc
    volontairement `Qwen2VLForConditionalGeneration.forward` calculer ses
    position_ids en interne à partir de `input_ids`/`attention_mask`/
    `image_grid_thw` (ce que son architecture est censée savoir faire, y
    compris sous padding, puisque le batching multimodal fait partie de son
    usage prévu) plutôt que de les recalculer nous-mêmes. Ce choix N'EST PAS
    vérifié empiriquement contre le vrai modèle (pas de GPU disponible pour
    l'écrire) -- c'est exactement ce que `scripts/verify_batching.py` vérifie
    avant tout entraînement réel : s'il échoue, appeler avec
    `TrainConfig.batched_embeddings=False` (cf. _sequential_last_token_hidden_state)
    plutôt que de lancer l'entraînement sur une loss potentiellement calculée
    à partir du mauvais token.
    """
    from qwen_vl_utils import process_vision_info

    if (texts is None) == (images is None):
        raise ValueError("Fournir exactement un de `texts` ou `images`, jamais les deux ni aucun.")

    messages_batch = _build_messages(texts, images)
    chat_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)

    tokenizer = getattr(processor, "tokenizer", processor)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = processor(
            text=chat_texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    finally:
        tokenizer.padding_side = original_padding_side

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[-1][:, -1, :]  # (len(texts or images), hidden_dim)


def embed_questions(model, processor, questions: list[str], batched: bool = True):
    """Embedding de chaque question (modalité texte seule)."""
    fn = _batched_last_token_hidden_state if batched else _sequential_last_token_hidden_state
    return fn(model, processor, texts=questions)


def embed_tiles(model, processor, images: list, batched: bool = True):
    """Embedding de chaque image de tuile (modalité image + prompt fixe)."""
    fn = _batched_last_token_hidden_state if batched else _sequential_last_token_hidden_state
    return fn(model, processor, images=images)


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
    # Mettre à False si scripts/verify_batching.py échoue sur cet environnement
    # (cf. _batched_last_token_hidden_state) -- retombe sur un appel modèle par
    # élément, plus lent mais sans hypothèse sur le padding/position_ids de
    # Qwen2-VL en batch.
    batched_embeddings: bool = True
    gradient_checkpointing: bool = False


def train_lora(train_config: TrainConfig = TrainConfig()) -> None:
    """Boucle d'entraînement complète : charge le modèle + LoRA, le dataset
    contrastif (Phase 2a/2b), et optimise la loss InfoNCE. Sauvegarde
    l'adaptateur LoRA (pas le modèle de base) dans `train_config.output_dir`.
    """
    import torch
    from torch.utils.data import DataLoader

    from .contrastive_dataset import ContrastiveTileDataset, contrastive_collate_fn

    model, processor = load_reader_model(gradient_checkpointing=train_config.gradient_checkpointing)
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
            question_embeds = embed_questions(model, processor, batch["questions"], batched=train_config.batched_embeddings)
            tile_embeds = embed_tiles(model, processor, batch["tile_images"], batched=train_config.batched_embeddings)
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
    parser.add_argument(
        "--no-batched-embeddings", dest="batched_embeddings", action="store_false",
        help=(
            "Désactive le batching des appels modèle (embed_questions/embed_tiles), retombant "
            "sur un appel par élément. À utiliser si scripts/verify_batching.py échoue sur cet "
            "environnement -- plus lent mais sans hypothèse sur le padding/position_ids en batch."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Réduit la mémoire d'activations (~20-30%% plus lent) -- filet de sécurité sur un GPU à VRAM limitée.",
    )
    args = parser.parse_args()

    train_lora(TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_negatives=args.n_negatives, output_dir=args.output_dir, images_root=args.images_root,
        batched_embeddings=args.batched_embeddings, gradient_checkpointing=args.gradient_checkpointing,
    ))


if __name__ == "__main__":
    main()
