"use strict";

// -- Palette : résolue depuis les custom properties CSS (style.css), pour que
// cytoscape (dessiné sur <canvas>, ne comprend pas var(--x)) reste cohérent
// avec le thème clair/sombre actif au chargement de la page. -----------------
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const PALETTE = {
  surface: cssVar("--surface-1"),
  textPrimary: cssVar("--text-primary"),
  textMuted: cssVar("--text-muted"),
  category: {
    heading: cssVar("--cat-heading"),
    text: cssVar("--cat-text"),
    table: cssVar("--cat-table"),
    list: cssVar("--cat-list"),
    media: cssVar("--cat-media"),
    quote_code: cssVar("--cat-quote-code"),
    other: cssVar("--cat-other"),
  },
  state: {
    inactive: cssVar("--text-muted"),
    active: cssVar("--status-warning"),
    opened: cssVar("--status-good"),
    pruned: cssVar("--status-serious"),
  },
  edge: {
    contains: cssVar("--edge-contains"),
    reading_order: cssVar("--edge-reading-order"),
    layout_adjacency: cssVar("--edge-layout-adjacency"),
    section_hierarchy: cssVar("--edge-section-hierarchy"),
    links_to: cssVar("--edge-links-to"),
  },
};

const CATEGORY_LABELS = {
  heading: "Titre (h1-h4)",
  text: "Texte",
  table: "Tableau",
  list: "Liste",
  media: "Média",
  quote_code: "Citation / code",
  other: "Autre",
};

const STATE_LABELS = { inactive: "Inactif", active: "Actif", opened: "Ouvert (évidence)", pruned: "Élagué" };

const RELATIONS = [
  { key: "reading_order", label: "Ordre de lecture", defaultOn: true, dashed: false },
  { key: "section_hierarchy", label: "Hiérarchie de sections", defaultOn: true, dashed: false },
  { key: "links_to", label: "Liens externes", defaultOn: true, dashed: false },
  { key: "contains", label: "Contenance", defaultOn: false, dashed: false },
  { key: "layout_adjacency", label: "Voisinage 2D", defaultOn: false, dashed: true },
];

function tagCategory(tag) {
  const first = (tag || "").split("+")[0];
  if (["h1", "h2", "h3", "h4"].includes(first)) return "heading";
  if (first === "p" || tag === "merged") return "text";
  if (first === "table") return "table";
  if (first === "ul" || first === "ol") return "list";
  if (first === "img" || first === "figure") return "media";
  if (first === "blockquote" || first === "pre") return "quote_code";
  return "other";
}

// -- État applicatif -----------------------------------------------------
let cy = null;

// -- Bandeau de statut -----------------------------------------------------
function setStatus(message, type) {
  const el = document.getElementById("status-banner");
  if (!message) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.className = `status-banner ${type}`;
  el.textContent = message;
}

// -- Positionnement (même heuristique que src/visualization.py::plot_graph :
// x = position horizontale réelle sur la page, y = rang dans l'ordre
// d'insertion (= ordre de lecture), pour que la forme du graphe reste
// lisible par rapport à la mise en page d'origine). ------------------------
function computeLayoutPositions(nodes) {
  const elementNodes = nodes.filter((n) => n.type === "element");
  const pageNode = nodes.find((n) => n.type === "page");
  const externalNodes = nodes.filter((n) => n.type === "external_page");

  const xs = elementNodes.map((n) => n.x + n.width / 2);
  const xMin = xs.length ? Math.min(...xs) : 0;
  const xMax = xs.length ? Math.max(...xs) : 1;
  const xSpan = Math.max(xMax - xMin, 1);

  const positions = {};
  elementNodes.forEach((n, rank) => {
    const cx = n.x + n.width / 2;
    positions[n.id] = { x: ((cx - xMin) / xSpan) * 700, y: rank * 58 };
  });

  if (pageNode) {
    const xs2 = elementNodes.map((n) => positions[n.id].x);
    const meanX = xs2.length ? xs2.reduce((a, b) => a + b, 0) / xs2.length : 350;
    positions[pageNode.id] = { x: meanX, y: -100 };
  }

  externalNodes.forEach((n, i) => {
    positions[n.id] = { x: 860, y: i * 66 };
  });

  return positions;
}

// -- Construction du graphe cytoscape ---------------------------------------
function buildCytoscape(graph) {
  const positions = computeLayoutPositions(graph.nodes);

  const elements = [];
  graph.nodes.forEach((n) => {
    elements.push({
      data: {
        ...n,
        category: n.type === "element" ? tagCategory(n.tag) : null,
        shortLabel: n.type === "external_page" ? shortDomainLabel(n.url) : undefined,
      },
      position: positions[n.id] || { x: 0, y: 0 },
    });
  });
  graph.edges.forEach((e) => elements.push({ data: e }));

  if (cy) cy.destroy();
  cy = cytoscape({
    container: document.getElementById("cy"),
    elements,
    style: cytoscapeStylesheet(),
    layout: { name: "preset" },
    wheelSensitivity: 0.25,
    minZoom: 0.2,
    maxZoom: 3,
  });

  RELATIONS.forEach((rel) => {
    cy.edges(`[relation = "${rel.key}"]`).style("display", rel.defaultOn ? "element" : "none");
  });

  attachGraphInteractions(cy);
  cy.fit(undefined, 40);
}

function cytoscapeStylesheet() {
  const categoryStyles = Object.entries(PALETTE.category).map(([cat, color]) => ({
    selector: `node[type = "element"][category = "${cat}"]`,
    style: { "background-color": color },
  }));

  const stateStyles = Object.entries(PALETTE.state).map(([state, color]) => ({
    selector: `node[type = "element"][state = "${state}"]`,
    style: { "border-color": color, "border-width": state === "opened" ? 4 : state === "active" ? 3 : 2 },
  }));

  const edgeStyles = RELATIONS.map((rel) => ({
    selector: `edge[relation = "${rel.key}"]`,
    style: {
      "line-color": PALETTE.edge[rel.key],
      "target-arrow-color": PALETTE.edge[rel.key],
      "target-arrow-shape": rel.key === "layout_adjacency" || rel.key === "contains" ? "none" : "triangle",
      "line-style": rel.dashed ? "dashed" : "solid",
      width: rel.key === "reading_order" || rel.key === "links_to" || rel.key === "section_hierarchy" ? 2.2 : 1.4,
      opacity: rel.key === "contains" || rel.key === "layout_adjacency" ? 0.55 : 0.95,
      "curve-style": "bezier",
      "control-point-step-size": 12,
      "arrow-scale": 0.9,
    },
  }));

  return [
    {
      selector: "node",
      style: {
        "font-size": 9,
        color: PALETTE.textMuted,
        "text-valign": "center",
        "text-halign": "right",
        "text-margin-x": 6,
        "border-width": 2,
        "border-color": PALETTE.textMuted,
        "background-color": PALETTE.textMuted,
      },
    },
    {
      selector: 'node[type = "element"]',
      style: { shape: "ellipse", width: 22, height: 22, label: "data(tag)" },
    },
    {
      selector: 'node[type = "page"]',
      style: {
        shape: "diamond",
        width: 34,
        height: 34,
        "background-color": PALETTE.textPrimary,
        "border-color": PALETTE.surface,
        "border-width": 2,
        label: "page",
        "font-weight": "bold",
      },
    },
    {
      selector: 'node[type = "external_page"]',
      style: {
        shape: "round-rectangle",
        width: 150,
        height: 26,
        "background-color": PALETTE.surface,
        "border-color": PALETTE.edge.links_to,
        "border-width": 2,
        label: "data(shortLabel)",
        "text-valign": "center",
        "text-halign": "center",
        "text-margin-x": 0,
        "font-size": 8.5,
        color: PALETTE.edge.links_to,
        "text-wrap": "ellipsis",
        "text-max-width": "140px",
      },
    },
    ...categoryStyles,
    ...stateStyles,
    ...edgeStyles,
    // En dernier : ces règles dynamiques (survol/sélection/pulse) doivent
    // l'emporter sur les règles statiques catégorie/état ci-dessus, quel que
    // soit le noeud concerné -- cytoscape applique les règles dans l'ordre du
    // tableau, la dernière correspondance gagne par propriété.
    { selector: "node.pulse", style: { "border-width": 6 } },
    { selector: "node:selected", style: { "border-width": 5, "border-color": PALETTE.textPrimary } },
  ];
}

function shortDomainLabel(url) {
  try {
    const u = new URL(url);
    const parts = decodeURIComponent(u.pathname).split("/");
    return (parts[parts.length - 1] || u.hostname).replace(/_/g, " ");
  } catch {
    return url;
  }
}

// -- Interactions : survol (tooltip) + clic (panneau de détail) ------------
function attachGraphInteractions(cyInstance) {
  const tooltip = document.getElementById("tooltip");

  cyInstance.on("mouseover", "node", (evt) => {
    const d = evt.target.data();
    tooltip.hidden = false;
    tooltip.innerHTML = tooltipContent(d);
  });
  cyInstance.on("mousemove", "node", (evt) => positionTooltip(evt.originalEvent));
  cyInstance.on("mouseout", "node", () => { tooltip.hidden = true; });

  cyInstance.on("mouseover", "edge", (evt) => {
    const d = evt.target.data();
    tooltip.hidden = false;
    let text = `<strong>${d.relation}</strong>`;
    if (d.direction) text += `<br/>${d.direction}`;
    if (d.anchor_text) text += `<br/>« ${d.anchor_text} »`;
    tooltip.innerHTML = text;
  });
  cyInstance.on("mousemove", "edge", (evt) => positionTooltip(evt.originalEvent));
  cyInstance.on("mouseout", "edge", () => { tooltip.hidden = true; });

  cyInstance.on("tap", "node", (evt) => renderNodeDetail(evt.target.data()));
}

function positionTooltip(originalEvent) {
  const tooltip = document.getElementById("tooltip");
  if (!originalEvent) return;
  tooltip.style.left = `${originalEvent.clientX + 14}px`;
  tooltip.style.top = `${originalEvent.clientY + 14}px`;
}

function tooltipContent(d) {
  if (d.type === "page") return `<strong>Page</strong><br/>${d.title || d.url}`;
  if (d.type === "external_page") return `<strong>Page externe</strong><br/>${d.url}`;
  const preview = (d.text_preview || "").slice(0, 140);
  return `<strong>${d.tag}</strong> — ${STATE_LABELS[d.state] || d.state}<br/>${preview}`;
}

function renderNodeDetail(d) {
  const body = document.getElementById("node-detail-body");
  if (d.type === "page") {
    body.innerHTML = `
      <dl>
        <dt>Type</dt><dd>Page</dd>
        <dt>Titre</dt><dd>${escapeHtml(d.title || "")}</dd>
        <dt>URL</dt><dd>${escapeHtml(d.url || "")}</dd>
      </dl>`;
    return;
  }
  if (d.type === "external_page") {
    body.innerHTML = `
      <dl>
        <dt>Type</dt><dd>Page externe</dd>
        <dt>URL</dt><dd>${escapeHtml(d.url || "")}</dd>
      </dl>
      <a class="open-link" href="${escapeAttr(d.url)}" target="_blank" rel="noopener">Ouvrir sur Wikipédia ↗</a>`;
    return;
  }
  const category = tagCategory(d.tag);
  body.innerHTML = `
    <dl>
      <dt>Balise</dt><dd>${escapeHtml(d.tag || "")}</dd>
      <dt>Catégorie</dt><dd>${CATEGORY_LABELS[category] || category}</dd>
      <dt>État</dt><dd>${STATE_LABELS[d.state] || d.state}</dd>
      ${d.relevance_score !== undefined ? `<dt>Score de pertinence</dt><dd>${Number(d.relevance_score).toFixed(3)}</dd>` : ""}
      <dt>Texte</dt><dd>${escapeHtml(d.text_preview || "")}</dd>
      ${d.evidence ? `<dt>Évidence lue par le VLM</dt><dd>${escapeHtml(d.evidence)}</dd>` : ""}
    </dl>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}
function escapeAttr(str) { return escapeHtml(str).replace(/"/g, "&quot;"); }

// -- Légende + interrupteurs de relation ------------------------------------
function renderLegend() {
  const legend = document.getElementById("graph-legend");
  const categoryDots = Object.entries(CATEGORY_LABELS)
    .map(([key, label]) => `<span class="legend-group"><span class="dot" style="background:${PALETTE.category[key]}"></span>${label}</span>`)
    .join("");
  const stateRings = Object.entries(STATE_LABELS)
    .map(([key, label]) => `<span class="legend-group"><span class="ring" style="border-color:${PALETTE.state[key]}"></span>${label}</span>`)
    .join("");

  legend.innerHTML = `
    <div class="legend-group"><span class="legend-group-title">Catégorie (couleur)</span>${categoryDots}</div>
    <div class="legend-group"><span class="legend-group-title">État (bordure)</span>${stateRings}</div>
  `;
}

function renderRelationToggles() {
  const container = document.getElementById("relation-toggles");
  container.innerHTML = RELATIONS.map(
    (rel) => `
    <label>
      <input type="checkbox" data-relation="${rel.key}" ${rel.defaultOn ? "checked" : ""} />
      <span class="swatch" style="background:${PALETTE.edge[rel.key]}; ${rel.dashed ? "border-top: 2px dashed " + PALETTE.edge[rel.key] + "; background: transparent;" : ""}"></span>
      ${rel.label}
    </label>`
  ).join("");

  container.querySelectorAll("input[type=checkbox]").forEach((input) => {
    input.addEventListener("change", () => {
      if (!cy) return;
      cy.edges(`[relation = "${input.dataset.relation}"]`).style("display", input.checked ? "element" : "none");
    });
  });
}

// -- Panneau "pages Wikipédia liées" -----------------------------------------
function renderExternalLinks(links) {
  const list = document.getElementById("external-links-list");
  if (!links.length) {
    list.innerHTML = `<li class="placeholder">Aucun lien sortant trouvé sur cette page (au-delà des premières tuiles de texte).</li>`;
    return;
  }
  list.innerHTML = links
    .map((link) => {
      const nodeId = `ext::${link.url}`;
      return `
      <li>
        <span class="badge ${link.used_as_evidence ? "used" : "unused"}">
          ${link.used_as_evidence ? "Utilisé comme évidence" : "Trouvé sur la page"}
        </span>
        <span class="link-anchor">${escapeHtml(link.anchor_text || shortDomainLabel(link.url))}</span>
        <span class="link-url">${escapeHtml(link.url)}</span>
        <a class="open-btn" href="${escapeAttr(link.url)}" target="_blank" rel="noopener">Ouvrir sur Wikipédia ↗</a>
        &nbsp;·&nbsp;
        <a class="open-btn" href="#" data-focus-node="${escapeAttr(nodeId)}">Voir dans le graphe</a>
      </li>`;
    })
    .join("");

  list.querySelectorAll("[data-focus-node]").forEach((link) => {
    link.addEventListener("click", (evt) => {
      evt.preventDefault();
      focusNodeInGraph(link.dataset.focusNode);
    });
  });
}

function focusNodeInGraph(nodeId) {
  if (!cy) return;
  const ele = cy.$id(nodeId);
  if (ele.empty()) return;
  cy.edges('[relation = "links_to"]').style("display", "element");
  document.querySelector('input[data-relation="links_to"]').checked = true;
  cy.animate({ center: { eles: ele }, zoom: 1.4 }, { duration: 300 });
  ele.addClass("pulse");
  setTimeout(() => ele.removeClass("pulse"), 1200);
  renderNodeDetail(ele.data());
}

// -- Journal du contrôleur ---------------------------------------------------
function renderLog(log) {
  const tbody = document.querySelector("#log-table tbody");
  tbody.innerHTML = log
    .map(
      (entry) => `
      <tr>
        <td>${entry.step}</td>
        <td>${entry.action}</td>
        <td>${escapeHtml(entry.node_id)}</td>
        <td>${Number(entry.score).toFixed(3)}</td>
        <td>${entry.state_before} → ${entry.state_after}</td>
      </tr>`
    )
    .join("");
}

// -- Soumission du formulaire -------------------------------------------------
document.getElementById("ask-form").addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const url = document.getElementById("url-input").value.trim();
  const question = document.getElementById("question-input").value.trim();
  if (!url || !question) return;

  const submitBtn = document.getElementById("submit-btn");
  submitBtn.disabled = true;
  document.getElementById("results").hidden = true;
  setStatus(
    "Analyse en cours (extraction de la page, tuilage, contrôleur d'évidence, lecture VLM tuile par tuile)… " +
      "cela peut prendre de quelques dizaines de secondes à quelques minutes selon la page.",
    "info"
  );

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, question }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || `Erreur ${res.status}`);

    setStatus("", null);
    document.getElementById("results").hidden = false;
    document.getElementById("answer-text").textContent = payload.answer;
    const pageLink = document.getElementById("page-url-link");
    pageLink.href = payload.page_url;
    pageLink.textContent = payload.page_url;

    renderLegend();
    renderRelationToggles();
    buildCytoscape(payload.graph);
    renderExternalLinks(payload.external_links);
    renderLog(payload.log);
    document.getElementById("node-detail-body").innerHTML =
      '<p class="placeholder">Survole ou clique un noeud du graphe pour voir son contenu.</p>';
  } catch (err) {
    setStatus(`Échec de l'analyse : ${err.message}`, "error");
  } finally {
    submitBtn.disabled = false;
  }
});
