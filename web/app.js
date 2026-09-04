const RED = "#ff3b30";
const TEAL = "#06b6a8";
const BORDER = "#d2d2d7";
const TEXT = "#1d1d1f";

const MIN_SCALE = 0.5;
const MAX_SCALE = 3.5;

let DATA = null;

// Mutable view state for the neighborhood graph: node positions are fixed
// per render (computed once), scale/pan let the user zoom and drag without
// recomputing layout.
let graphState = null;

async function main() {
  const res = await fetch("data.json");
  DATA = await res.json();

  renderOverview();
  renderNodeSelect();
  renderErrorAnalysis();
  setupGraphInteractivity();

  document.getElementById("node-select").addEventListener("change", (e) => {
    renderAccountLookup(Number(e.target.value));
  });
  renderAccountLookup(DATA.test_node_ids[0]);

  setupNav();
}

function renderOverview() {
  const m = DATA.metrics;
  document.getElementById("hero-recall").textContent = m.recall.toFixed(3);
  document.getElementById("stat-precision").textContent = m.precision.toFixed(3);
  document.getElementById("stat-f1").textContent = m.f1.toFixed(3);
  document.getElementById("stat-fp").textContent = m.false_positive;
  document.getElementById("stat-mules").textContent = m.test_mule_count;
  document.getElementById("caveat-text").textContent =
    `Sample size — test set contains only ${m.test_mule_count} mule accounts. ` +
    `Read these numbers as directional signal, not a statistically precise benchmark.`;
}

function renderNodeSelect() {
  const select = document.getElementById("node-select");
  select.innerHTML = "";
  DATA.test_node_ids.forEach((id) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = `Account ${id}`;
    select.appendChild(opt);
  });
}

function renderAccountLookup(nodeId) {
  const node = DATA.nodes[String(nodeId)];
  document.getElementById("node-select").value = nodeId;

  const predChip = document.getElementById("pred-chip");
  predChip.textContent = node.predicted_mule ? "MULE" : "NORMAL";
  predChip.className = "chip " + (node.predicted_mule ? "chip-mule" : "chip-normal");

  document.getElementById("confidence-value").textContent = node.confidence.toFixed(3);

  const trueChip = document.getElementById("true-chip");
  trueChip.textContent = node.true_mule ? "MULE" : "NORMAL";
  trueChip.className = "chip " + (node.true_mule ? "chip-mule" : "chip-normal");

  const outcome = document.getElementById("outcome-text");
  let text, flagged;
  if (node.true_mule && node.predicted_mule) {
    text = "TRUE POSITIVE — real mule, correctly caught."; flagged = true;
  } else if (node.true_mule && !node.predicted_mule) {
    text = "FALSE NEGATIVE — real mule, missed."; flagged = false;
  } else if (!node.true_mule && node.predicted_mule) {
    text = "FALSE POSITIVE — normal account, flagged in error."; flagged = true;
  } else {
    text = "TRUE NEGATIVE — normal account, correctly cleared."; flagged = false;
  }
  outcome.textContent = text;
  outcome.className = "outcome" + (flagged ? " flagged" : "");

  initNeighborhoodGraph(node);
}

// ---- neighborhood graph: layout + interactive view state -----------------

function initNeighborhoodGraph(node) {
  const canvas = document.getElementById("graph-canvas");
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const radius = Math.min(w, h) * 0.36;

  const positions = { [node.id]: { x: cx, y: cy } };
  const n = node.neighbors.length;
  node.neighbors.forEach((nb, i) => {
    const angle = (2 * Math.PI * i) / Math.max(n, 1) - Math.PI / 2;
    positions[nb.id] = {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });

  graphState = {
    canvas,
    ctx: canvas.getContext("2d"),
    node,
    positions,
    scale: 1,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragMoved: false,
  };

  renderGraphFrame();
}

function renderGraphFrame() {
  if (!graphState) return;
  const { ctx, canvas, node, positions, scale, panX, panY } = graphState;
  const w = canvas.width, h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(scale, scale);

  const center = positions[node.id];

  // edges first, nodes on top
  node.neighbors.forEach((nb) => {
    const p = positions[nb.id];
    if (nb.direction === "in" || nb.direction === "both") {
      drawArrow(ctx, p.x, p.y, center.x, center.y, BORDER);
    }
    if (nb.direction === "out" || nb.direction === "both") {
      drawArrow(ctx, center.x, center.y, p.x, p.y, BORDER);
    }
  });

  drawNode(ctx, center.x, center.y, 15, node.predicted_mule ? RED : TEAL, true, node.id, false);

  node.neighbors.forEach((nb) => {
    const p = positions[nb.id];
    const clickable = String(nb.id) in DATA.nodes;
    drawNode(ctx, p.x, p.y, 9, nb.predicted_mule ? RED : TEAL, false, nb.id, clickable);
  });

  ctx.restore();
}

function drawNode(ctx, x, y, r, color, selected, label, clickable) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  if (selected) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = TEXT;
    ctx.stroke();
  } else if (clickable) {
    // a thin ring marks neighbors you can click into — otherwise it's not
    // obvious which nodes are inspectable vs. just context
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }
  ctx.fillStyle = TEXT;
  ctx.font = "11px 'SF Mono', 'IBM Plex Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText(String(label), x, y - r - 8);
}

function drawArrow(ctx, x1, y1, x2, y2, color) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const nodeGap = 18;
  const sx = x1 + nodeGap * Math.cos(angle);
  const sy = y1 + nodeGap * Math.sin(angle);
  const ex = x2 - nodeGap * Math.cos(angle);
  const ey = y2 - nodeGap * Math.sin(angle);

  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.stroke();

  const headLen = 7;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - headLen * Math.cos(angle - Math.PI / 6), ey - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(ex - headLen * Math.cos(angle + Math.PI / 6), ey - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

// screen (canvas pixel) coords -> world (pre-transform) coords
function toWorld(screenX, screenY) {
  return {
    x: (screenX - graphState.panX) / graphState.scale,
    y: (screenY - graphState.panY) / graphState.scale,
  };
}

function nodeAt(worldX, worldY) {
  const { node, positions } = graphState;
  const all = [{ id: node.id }, ...node.neighbors];
  for (const n of all) {
    const p = positions[n.id];
    const dx = p.x - worldX, dy = p.y - worldY;
    if (Math.sqrt(dx * dx + dy * dy) <= 14) return n.id;
  }
  return null;
}

function setupGraphInteractivity() {
  const canvas = document.getElementById("graph-canvas");

  canvas.addEventListener("wheel", (e) => {
    if (!graphState) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const mouseY = ((e.clientY - rect.top) / rect.height) * canvas.height;

    const worldBefore = toWorld(mouseX, mouseY);
    const zoomFactor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    graphState.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, graphState.scale * zoomFactor));

    // keep the point under the cursor fixed while zooming, instead of
    // zooming around the canvas corner
    graphState.panX = mouseX - worldBefore.x * graphState.scale;
    graphState.panY = mouseY - worldBefore.y * graphState.scale;
    renderGraphFrame();
  }, { passive: false });

  canvas.addEventListener("mousedown", (e) => {
    if (!graphState) return;
    graphState.isDragging = true;
    graphState.dragMoved = false;
    graphState.lastX = e.clientX;
    graphState.lastY = e.clientY;
    canvas.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!graphState || !graphState.isDragging) return;
    const dx = e.clientX - graphState.lastX;
    const dy = e.clientY - graphState.lastY;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) graphState.dragMoved = true;
    graphState.panX += dx;
    graphState.panY += dy;
    graphState.lastX = e.clientX;
    graphState.lastY = e.clientY;
    renderGraphFrame();
  });

  window.addEventListener("mouseup", () => {
    if (!graphState) return;
    graphState.isDragging = false;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!graphState || graphState.isDragging) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const screenY = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const world = toWorld(screenX, screenY);
    const hit = nodeAt(world.x, world.y);
    const clickable = hit !== null && hit !== graphState.node.id && String(hit) in DATA.nodes;
    canvas.style.cursor = clickable ? "pointer" : "grab";
  });

  canvas.addEventListener("click", (e) => {
    if (!graphState || graphState.dragMoved) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const screenY = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const world = toWorld(screenX, screenY);
    const hit = nodeAt(world.x, world.y);
    if (hit !== null && hit !== graphState.node.id && String(hit) in DATA.nodes) {
      renderAccountLookup(hit);
    }
  });

  canvas.style.cursor = "grab";
}

function renderErrorAnalysis() {
  const ea = DATA.error_analysis;
  const layeringPct = ea.layering_total ? (ea.layering_caught / ea.layering_total) * 100 : 0;
  const funnelPct = ea.funnel_total ? (ea.funnel_caught / ea.funnel_total) * 100 : 0;

  document.getElementById("layering-recall").textContent = `${Math.round(layeringPct)}%`;
  document.getElementById("layering-frac").textContent = `${ea.layering_caught}/${ea.layering_total} caught`;
  document.getElementById("funnel-recall").textContent = `${Math.round(funnelPct)}%`;
  document.getElementById("funnel-frac").textContent = `${ea.funnel_caught}/${ea.funnel_total} caught`;

  const list = document.getElementById("fp-list");
  list.innerHTML = "";
  if (ea.false_positives.length === 0) {
    list.innerHTML = '<div class="panel">No false positives in this test split.</div>';
    return;
  }
  ea.false_positives.forEach((fp) => {
    const row = document.createElement("div");
    row.className = "fp-row";
    row.innerHTML = `<span class="fp-node">NODE ${fp.node_id}</span><span class="fp-verdict">${fp.verdict}</span>`;
    list.appendChild(row);
  });
}

function setupNav() {
  const links = document.querySelectorAll(".nav-link");
  links.forEach((link) => {
    link.addEventListener("click", () => {
      links.forEach((l) => l.classList.remove("active"));
      link.classList.add("active");
    });
  });
}

main();
