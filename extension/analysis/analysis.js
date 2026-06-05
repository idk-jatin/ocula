// API base is loaded dynamically from extension settings
let API_BASE = "http://localhost:8000"; // fallback until storage resolves

chrome.storage.local.get("settings", (data) => {
  if (data.settings?.apiBase) API_BASE = data.settings.apiBase;
});

// ── DOM refs ──────────────────────────────────────────────────────
const inputEl    = document.getElementById("inputText");
const analyzeBtn = document.getElementById("analyzeBtn");
const charCount  = document.getElementById("charCount");
const loadingEl  = document.getElementById("loading");
const resultsEl  = document.getElementById("results");
const errorEl    = document.getElementById("errorBox");

// ── Char counter ──────────────────────────────────────────────────
inputEl.addEventListener("input", () => {
  const n = inputEl.value.length;
  charCount.textContent  = `${n} / 2000`;
  charCount.style.color  = n > 1800 ? "#F87171" : "#334155";
});

// ── Keyboard shortcut ─────────────────────────────────────────────
inputEl.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") analyze();
});

analyzeBtn.addEventListener("click", analyze);

// ── Main analyse function ─────────────────────────────────────────
async function analyze() {
  const text = inputEl.value.trim();
  if (!text) return;

  // Reset
  resultsEl.classList.add("hidden");
  errorEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");
  analyzeBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/explain`, {
      method:  "POST",
      headers: {
        "Content-Type":                "application/json",
        "ngrok-skip-browser-warning": "true"
      },
      body:    JSON.stringify({ text, top_k: 15 }),
      signal:  AbortSignal.timeout(30000)  // 30s timeout for SHAP
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    showError(err.message);
  } finally {
    loadingEl.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────
function renderResults(data) {
  const { label, confidence, probabilities, tokens, highlight, latency_ms } = data;

  // 1. Label banner
  const banner = document.getElementById("labelBanner");
  const META   = {
    hate:      { title: "Hate Speech",       cls: "b-hate" },
    offensive: { title: "Offensive Language", cls: "b-offensive" },
    normal:    { title: "Normal Content",     cls: "b-normal" },
  };
  const m = META[label] || META.normal;
  banner.className                              = `label-banner ${m.cls}`;
  document.getElementById("labelTitle").textContent = m.title;
  document.getElementById("labelConf").textContent  =
    `${Math.round(confidence * 100)}% confidence`;
  document.getElementById("latencyBadge").textContent =
    latency_ms ? `${(latency_ms / 1000).toFixed(1)}s` : "";

  // 2. Probability bars
  const probs = {
    hate:      probabilities.hate      || 0,
    offensive: probabilities.offensive || 0,
    normal:    probabilities.normal    || 0,
  };
  for (const [cls, val] of Object.entries(probs)) {
    const pct = Math.round(val * 100);
    const key = cls.charAt(0).toUpperCase() + cls.slice(1);
    document.getElementById(`bar${key}`).style.width    = `${pct}%`;
    document.getElementById(`pct${key}`).textContent = `${pct}%`;
  }

  // 3. SHAP-highlighted text
  const htEl = document.getElementById("highlightedText");
  if (highlight && highlight.trim()) {
    htEl.innerHTML = highlight;
  } else {
    htEl.textContent = inputEl.value.trim();
  }

  // 4. Token chart
  renderTokenChart(tokens, label);

  // Show
  resultsEl.classList.remove("hidden");
  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Token bar chart ───────────────────────────────────────────────
function renderTokenChart(tokens, label) {
  const container = document.getElementById("tokenChart");
  container.innerHTML = "";

  const top = (tokens || []).filter(t => t.score > 0.05).slice(0, 12);

  if (!top.length) {
    container.innerHTML =
      `<span style="color:#334155;font-size:13px">No significant tokens found.</span>`;
    return;
  }

  top.forEach(token => {
    const pct = Math.round(token.score * 100);
    const row = document.createElement("div");
    row.className = "tc-row";
    row.innerHTML = `
      <span class="tc-word" title="${esc(token.word)}">${esc(token.word)}</span>
      <div class="tc-track">
        <div class="tc-fill ${label}" style="width:${pct}%"></div>
      </div>
      <span class="tc-score">${pct}%</span>
    `;
    container.appendChild(row);
  });
}

// ── Error display ─────────────────────────────────────────────────
function showError(msg) {
  errorEl.textContent = `⚠ ${msg} — Make sure the OCULA server is running on ${API_BASE} (change in extension popup settings)`;
  errorEl.classList.remove("hidden");
}

// ── Utils ─────────────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
