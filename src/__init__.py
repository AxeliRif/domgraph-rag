"""Pipeline DOM-Graph RAG — Phase 1 et Phase 2.

Le package expose des symboles utiles pour l'ensemble du pipeline, mais les
modules d'entraînement contrastif peuvent nécessiter des dépendances lourdes
comme PyTorch. Pour éviter de casser les imports de la Phase 1/2a dès l'
import du package, les symboles sont chargés à la demande via `__getattr__`."""

from importlib import import_module
from typing import Any

__all__ = [
    "DOMElement",
    "extract_dom_elements",
    "extract_dom_elements_async",
    "Tile",
    "build_tiles",
    "build_graph",
    "compute_reading_order",
    "compute_layout_adjacency",
    "compute_section_hierarchy",
    "graph_to_xml",
    "build_page_graph",
    "build_page_graph_from_elements",
    "build_corpus_graph",
    "slug_for_url",
    "PageSection",
    "is_oversized_page",
    "split_into_sections",
    "plot_graph",
    "plot_page_overlay",
    "VLMClient",
    "QAPair",
    "generate_qa_dataset",
    "load_qa_pairs",
    "ContrastiveExample",
    "mine_hard_negatives",
    "ContrastiveTileDataset",
    "contrastive_collate_fn",
    "EvidenceControllerConfig",
    "ControllerAction",
    "run_evidence_controller",
    "collect_evidence",
    "state_counts",
]

_EXPORTS = {
    "DOMElement": (".dom_extraction", "DOMElement"),
    "extract_dom_elements": (".dom_extraction", "extract_dom_elements"),
    "extract_dom_elements_async": (".dom_extraction", "extract_dom_elements_async"),
    "Tile": (".tiling", "Tile"),
    "build_tiles": (".tiling", "build_tiles"),
    "build_graph": (".graph_builder", "build_graph"),
    "compute_reading_order": (".graph_builder", "compute_reading_order"),
    "compute_layout_adjacency": (".graph_builder", "compute_layout_adjacency"),
    "compute_section_hierarchy": (".graph_builder", "compute_section_hierarchy"),
    "graph_to_xml": (".graph_builder", "graph_to_xml"),
    "build_page_graph": (".corpus_builder", "build_page_graph"),
    "build_page_graph_from_elements": (".corpus_builder", "build_page_graph_from_elements"),
    "build_corpus_graph": (".corpus_builder", "build_corpus_graph"),
    "slug_for_url": (".corpus_builder", "slug_for_url"),
    "PageSection": (".sectioning", "PageSection"),
    "is_oversized_page": (".sectioning", "is_oversized_page"),
    "split_into_sections": (".sectioning", "split_into_sections"),
    "plot_graph": (".visualization", "plot_graph"),
    "plot_page_overlay": (".visualization", "plot_page_overlay"),
    "VLMClient": (".vlm_client", "VLMClient"),
    "QAPair": (".qa_generation", "QAPair"),
    "generate_qa_dataset": (".qa_generation", "generate_qa_dataset"),
    "load_qa_pairs": (".qa_generation", "load_qa_pairs"),
    "ContrastiveExample": (".hard_negative_mining", "ContrastiveExample"),
    "mine_hard_negatives": (".hard_negative_mining", "mine_hard_negatives"),
    "ContrastiveTileDataset": (".contrastive_dataset", "ContrastiveTileDataset"),
    "contrastive_collate_fn": (".contrastive_dataset", "contrastive_collate_fn"),
    "EvidenceControllerConfig": (".evidence_controller", "EvidenceControllerConfig"),
    "ControllerAction": (".evidence_controller", "ControllerAction"),
    "run_evidence_controller": (".evidence_controller", "run_evidence_controller"),
    "collect_evidence": (".evidence_controller", "collect_evidence"),
    "state_counts": (".evidence_controller", "state_counts"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
