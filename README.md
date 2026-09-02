# DOM-Graph RAG — Phase 1 (starter)

Project skeleton to get started on the pipeline described in *Visuel DOM-Graph RAG*:
combining **PixelRAG**'s pixel rendering with **MAGE-RAG**'s multigranular
graph, but replacing the blind grid tiling with DOM-guided tiling.

## 0. The two source papers are real — and open-source

While preparing this skeleton, I checked the two papers cited in your slides:
both are recent publications (June 2026) with public code.

- **PixelRAG** — *PIXELRAG: Web Screenshots Beat Text for Retrieval-Augmented
  Generation* (Yichuan Wang et al., UC Berkeley/BAIR/Berkeley NLP).
  [arXiv:2606.28344](https://arxiv.org/abs/2606.28344) ·
  [github.com/StarTrail-org/PixelRAG](https://github.com/StarTrail-org/PixelRAG)
  (Apache-2.0, `pip install pixelrag`).
  Rendered at 875px wide, 1024px-tall tiles → matches the "875x1024 pixels"
  from your slide. The embedding model used is `Qwen/Qwen3-VL-Embedding-2B`
  (LoRA). One thing worth noting: the full FAISS index for Wikipedia is
  ~217 GB according to the repo's current README, not <120 GB — either the
  slide referred to a different config (e.g. a compressed index) or an
  earlier version of the paper; worth checking if the exact figure matters
  for your report.
- **MAGE-RAG** — *Multigranular Adaptive Graph Evidence for Agentic
  Multimodal RAG in Long-Document QA* (Yilong Zuo et al.).
  [arXiv:2606.15906](https://arxiv.org/abs/2606.15906) ·
  code: `github.com/laonuo2004/MAGE-RAG` (no pip package identified — more of
  a research repo to clone and read).
  Published results: 52.75 (accuracy) on LongDocURL, 53.26 acc / 51.19 F1 on
  MMLongBench-Doc — these are the reference numbers to beat for your Phase 4.

Worth noting: an independent review (VentureBeat) of PixelRAG confirms
exactly the problem your slide 6 identifies — *"it slices pages by fixed
pixel height, meaning a table or paragraph can get cut in half mid-tile with
no awareness of content boundaries"*. Your angle (DOM-guided tiling) therefore
addresses a real limitation already flagged by the community, not just a
hypothesis — a good signal going forward.

## 1. Playing with PixelRAG and MAGE-RAG directly

Before coding your own version, it's worth seeing the two source projects run.

**PixelRAG — single-document quickstart (no need for the 217 GB Wikipedia index):**
```bash
pip install 'pixelrag[index]'

curl -L -o paper.pdf https://raw.githubusercontent.com/StarTrail-org/PixelRAG/main/assets/pixelrag-paper.pdf

cat > pixelrag.yaml << 'EOF'
source:
  type: local
  path: ./paper.pdf
embed:
  model: Qwen/Qwen3-VL-Embedding-2B
  device: auto   # cuda on Linux, mps on Apple Silicon Macs, cpu otherwise
output: ./paper_index
EOF

pixelrag index build      # ~1-3 min
pixelrag serve --index-dir ./paper_index --port 30001

curl -X POST http://localhost:30001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"text": "Overview of PixelRAG and the diagram"}], "n_docs": 1}'
```
The package also exposes `from pixelrag_render import render_url` — handy
for comparing, on the same page, PixelRAG's blind grid tiling against your
DOM-guided tiling (a good ablation candidate for your report).

**MAGE-RAG — no pip package, clone and read the code instead:**
```bash
git clone https://github.com/laonuo2004/MAGE-RAG.git
```
Look in particular at the offline graph construction (page/element nodes,
*containment*, *reading order*, *layout adjacency*, *section hierarchy*,
*semantic-neighbor* relations) and the online evidence controller
(Algorithm 1 in the paper) — that's the part Phase 3 of your project will
reimplement.

## 2. Installing this skeleton

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
playwright install chromium      # downloads the headless browser

# Recommended VLM backend to get started (lightweight, and you already know Ollama):
ollama pull qwen3-vl
```

In VS Code: open the `domgraph-rag/` folder, install the **Python** +
**Jupyter** extensions, select the `.venv` interpreter, then open
`notebooks/01_pipeline_playground.ipynb` and pick that same environment
as the kernel.

## 3. Getting started

```bash
# Check that everything works (local page, no internet needed):
python3 tests/test_pipeline_smoke.py
```
Then open `notebooks/00_pipeline_overview.ipynb` in VS Code and run it top to
bottom (~a few seconds, no network or VLM needed) — it walks through every
phase of the pipeline on the local fixture page, with a plot at each step.
Once you're familiar with the pipeline, `notebooks/01_pipeline_playground.ipynb`
runs the same idea against a real page; by default it runs against a local
test page, just change the `URL` variable to a real page (e.g. a Wikipedia
article) to test under real conditions — the code doesn't change, Playwright
handles `file://` and remote URLs the same way.

## 4. Project structure

```
domgraph-rag/
├── src/
│   ├── config.py               # constants (tile sizes, noise selectors, VLM model, LoRA hyperparameters...)
│   ├── dom_extraction.py       # Phase 1a/1b — Playwright + DOM bounding-box extraction
│   ├── tiling.py                # Phase 1c — adaptive tiling + patching (not a fixed grid)
│   ├── graph_builder.py         # Phase 1d — page/element graph + reading order + XML export
│   ├── corpus_builder.py        # Phase 1e — multi-page graph (hyperlinks)
│   ├── vlm_client.py            # Phase 1a — VLM wrapper (Ollama or Transformers)
│   ├── qa_generation.py         # Phase 2a — synthetic question/answer pairs per tile
│   ├── hard_negative_mining.py  # Phase 2b — hard negatives (TF-IDF + cosine)
│   ├── contrastive_dataset.py   # Phase 2c — PyTorch Dataset (question, positive, negatives)
│   ├── lora_finetune.py         # Phase 2d — LoRA fine-tuning of the reader VLM (ViT + LLM)
│   └── evidence_controller.py   # Phase 3 — online evidence controller (MAGE-RAG Algorithm 1)
├── webapp/                      # Phase 3 — demo site (question -> answer + interactive graph)
│   ├── backend/
│   │   ├── main.py              # FastAPI app (POST /api/ask) + frontend static files
│   │   └── pipeline.py          # orchestrates DOM -> tiles -> graph -> evidence controller -> VLM
│   └── frontend/                # static HTML/CSS/JS, no build step (cytoscape.js via CDN)
├── notebooks/
│   ├── 00_pipeline_overview.ipynb     # offline, self-contained walkthrough of every phase (local fixture, no network/VLM)
│   └── 01_pipeline_playground.ipynb   # end-to-end Phase 1 + Phase 2/3 demo on a real page, with interactive visualizations
├── tests/
│   ├── fixtures/sample_page.html      # local test page (no network access required)
│   ├── test_pipeline_smoke.py         # end-to-end sanity check script (Phase 1)
│   └── test_phase2_smoke.py           # QA generation + mining + dataset (no real VLM or GPU)
└── requirements.txt
```

**What's already working and tested** (end-to-end, on the local test page):
DOM extraction with interface-noise removal → adaptive tiling (merging small
elements, splitting oversized ones) → two-level graph construction with
spatial reading order (not DOM order) → XML export readable by a reader
model.

**Deliberate simplifications worth knowing about** (natural next steps):
- nested DOM elements (e.g. an `<img>` inside a `<figure>`) are pruned by
  full containment (`tiling.prune_nested_elements`); partial overlaps between
  otherwise independent elements (e.g. a paragraph and a floated infobox that
  visually share the same space) are tiled together rather than separately
  (`tiling.group_overlapping_elements`), and the width of "text-flow" tags
  (`h1`-`h4`, `p`, `blockquote`, `ul`, `ol`) is measured on the actually
  rendered text rather than their full block box (`dom_extraction.py`,
  `TEXT_FLOW_TAGS`) — both avoid the same region of pixels ending up
  duplicated across two tiles;
- the graph now instantiates all 5 of MAGE-RAG's relations (`contains`,
  `reading_order`, `layout_adjacency`, `section_hierarchy`, `semantic_neighbor`,
  plus `links_to` on the domgraph-rag side for cross-page connectivity).
  `semantic_neighbor` (TF-IDF + cosine between tiles, see
  `graph_builder.compute_semantic_neighbor_edges`) is **off by default**:
  `build_graph(..., use_semantic_similarity=True)` turns it on, meant for
  comparing the evidence controller with and without this relation rather
  than always forcing it on;
- every element node already has a `state` attribute (`inactive` by
  default) — ready for the Phase 3 evidence controller to evolve it.

## 5. Phase 2 — contrastive training (implemented)

Generates a synthetic (question → tile) dataset and contrastively
fine-tunes the reader VLM. Four steps, one per module:

1. **`qa_generation.py`** — the VLM reads each tile and asks a question whose
   answer is only visible in that tile (prompt `QA_GENERATION_PROMPT` in
   `config.py`). Every tile is saved to disk (`data/qa_dataset/images/`), not
   just the ones kept as positives, so mining (step 2) can draw negatives
   from them.
2. **`hard_negative_mining.py`** — for each question, picks a few tiles from
   the same page that are lexically close (TF-IDF + cosine, scikit-learn)
   but don't contain the answer: "hard" negatives, more useful for training
   than a randomly picked tile. Works page by page (`Tile.id` is only unique
   within a page); mining negatives *across* pages of a corpus is a natural
   extension described in the module's docstring.
3. **`contrastive_dataset.py`** — assembles a `torch.utils.data.Dataset`
   (question, positive tile image, negative tile images) from the `.jsonl`
   files produced by the two previous steps.
4. **`lora_finetune.py`** — fine-tunes `Qwen2VLForConditionalGeneration` via
   LoRA, applied to both the LLM and the vision tower (ViT), with an InfoNCE
   loss (the generative model is turned into an encoder via a short prompt,
   in the spirit of VLMs used as universal encoders in the recent literature
   — exact reference to double-check before citing it in the report, in the
   same spirit as the sourcing note at the top of this README). Requires the
   `transformers` + `peft` backend (see the commented-out section of
   `requirements.txt`) and ideally a GPU; run via:
   ```bash
   python3 -m src.lora_finetune --epochs 3 --batch-size 4
   ```

See sections 7-9 of `notebooks/01_pipeline_playground.ipynb` for a demo of
steps 1 to 3 on the page loaded in Phase 1, and `tests/test_phase2_smoke.py`
for tests that need neither a real VLM nor a GPU (generation with a fake
client, mining, dataset assembly).

## 6. Phase 3 — evidence controller + demo site (implemented)

`src/evidence_controller.py` evolves each element node's `state` attribute
(`inactive → active → opened`/`pruned`) following MAGE-RAG Algorithm 1's
activate/open/search/prune loop under budget, given a query. See section 10
of the notebook for a step-by-step demo, and `src/evidence_controller.py`
for the full policy (TF-IDF + cosine relevance, best-first under budget).

`webapp/` exposes this same loop behind a small site: a "Wikipedia URL +
question" form runs the full pipeline (DOM extraction → tiling → graph →
evidence controller, which reads each opened tile via the VLM) and then
displays the synthesized answer, the interactive graph (content category by
color, controller state by border, edges filterable by relation) and the
linked Wikipedia pages (`links_to`) — with a badge flagging the ones carried
by a tile actually retained as evidence.

```bash
# Ollama must be running with the configured model (config.OLLAMA_MODEL):
ollama pull qwen3-vl

pip install -r requirements.txt   # adds fastapi + uvicorn
uvicorn webapp.backend.main:app --reload
```
Then open http://127.0.0.1:8000/. Each request re-runs the pipeline live (no
pre-built corpus) — expect anywhere from a few dozen seconds to a few
minutes depending on the page and the controller's budget (the VLM is
queried once per opened tile, plus once to synthesize the final answer).

## 7. Roadmap (Phase 4)

- **Phase 4 — evaluation**: LongDocURL, MMLongBench-Doc, SimpleQA benchmarks;
  diagnostic parser/rank/reader loss; write-up.
