// ── Helpers ───────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ── Load settings into UI ─────────────────────────────────────────
async function load() {
  const data = await chrome.storage.local.get("settings");
  const s    = data.settings;
  if (!s) return;

  // Toggle
  $("enabledToggle").checked = s.enabled;

  // Stats
  const { scanned, flagged } = s.stats;
  $("scannedCount").textContent = scanned;
  $("flaggedCount").textContent = flagged;
  $("flagRate").textContent     = scanned > 0
    ? Math.round(flagged / scanned * 100) + "%" : "0%";

  // Platforms
  $$("[data-platform]").forEach(el => {
    el.checked = s.platforms[el.dataset.platform] ?? true;
  });

  // Threshold
  $("thresholdSlider").value    = s.threshold;
  $("thresholdVal").textContent = Math.round(s.threshold * 100) + "%";

  // API base
  $("apiBase").value = s.apiBase;

  // Server status
  checkServer(s.apiBase);
}

// ── Save settings from UI ─────────────────────────────────────────
async function save() {
  const data = await chrome.storage.local.get("settings");
  const s    = data.settings;

  s.enabled   = $("enabledToggle").checked;
  s.threshold = parseFloat($("thresholdSlider").value);
  s.apiBase   = $("apiBase").value.trim();

  $$("[data-platform]").forEach(el => {
    s.platforms[el.dataset.platform] = el.checked;
  });

  await chrome.storage.local.set({ settings: s });

  // Visual confirmation
  $("saveBtn").textContent = "Saved ✓";
  setTimeout(() => { $("saveBtn").textContent = "Save"; }, 1500);

  checkServer(s.apiBase);
}

// ── Server health check ───────────────────────────────────────────
async function checkServer(apiBase) {
  const dot = $("serverDot");
  dot.className   = "dot dot-gray";
  dot.title       = "Checking...";
  try {
    const res  = await fetch(`${apiBase}/health`, {
      signal:  AbortSignal.timeout(2500),
      headers: { "ngrok-skip-browser-warning": "true" }
    });
    const data = await res.json();
    if (data.model_loaded) {
      dot.className = "dot dot-green";
      dot.title     = `Connected · Model loaded (${data.gpu_name || "CPU"})`;
    } else {
      dot.className = "dot dot-red";
      dot.title     = "Server running but model not loaded yet";
    }
  } catch {
    dot.className = "dot dot-red";
    dot.title     = "Server unreachable — is uvicorn running?";
  }
}

// ── Events ────────────────────────────────────────────────────────
$("thresholdSlider").addEventListener("input", e => {
  $("thresholdVal").textContent = Math.round(e.target.value * 100) + "%";
});

$("saveBtn").addEventListener("click", save);

$("resetBtn").addEventListener("click", async () => {
  chrome.runtime.sendMessage({ type: "RESET_STATS" }, () => {
    $("scannedCount").textContent = "0";
    $("flaggedCount").textContent = "0";
    $("flagRate").textContent     = "0%";
  });
});

$("analysisBtn").addEventListener("click", () => {
  chrome.tabs.create({
    url: chrome.runtime.getURL("analysis/analysis.html")
  });
});

// Reload popup stats when window gains focus
window.addEventListener("focus", load);

// Init
load();
