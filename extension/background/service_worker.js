const DEFAULT_SETTINGS = {
  enabled:   true,
  apiBase:   "http://localhost:8000",
  threshold: 0.65,
  platforms: {
    twitter:  true,
    reddit:   true
  },
  stats: { scanned: 0, flagged: 0 }
};

// ── Init ──────────────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ settings: DEFAULT_SETTINGS });
  console.log("OCULA installed.");
});

// ── Message router ────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  if (msg.type === "PREDICT") {
    getSettings().then(settings => {
      if (!settings.enabled) {
        sendResponse({ skip: true });
        return;
      }
      fetchPredict(msg.text, settings.apiBase, settings.threshold)
        .then(result => {
          if (result.flagged) updateStats(true);
          else                updateStats(false);
          sendResponse(result);
        })
        .catch(err => sendResponse({ error: err.message, flagged: false }));
    });
    return true; // keep channel open for async response
  }

  if (msg.type === "GET_SETTINGS") {
    getSettings().then(s => sendResponse(s));
    return true;
  }

  if (msg.type === "RESET_STATS") {
    getSettings().then(settings => {
      settings.stats = { scanned: 0, flagged: 0 };
      chrome.storage.local.set({ settings });
      sendResponse({ ok: true });
    });
    return true;
  }
});

// ── Helpers ───────────────────────────────────────────────────────
async function getSettings() {
  const data = await chrome.storage.local.get("settings");
  return data.settings || DEFAULT_SETTINGS;
}

async function fetchPredict(text, apiBase, threshold) {
  try {
    const res = await fetch(`${apiBase}/predict`, {
      method:  "POST",
      headers: {
        "Content-Type":                "application/json",
        "ngrok-skip-browser-warning": "true"
      },
      body:    JSON.stringify({ text }),
      signal:  AbortSignal.timeout(5000)   // 5s max wait
    });

    if (!res.ok) throw new Error(`API error ${res.status}`);

    const data    = await res.json();
    const flagged = (data.label !== "normal") && (data.confidence >= threshold);
    return { ...data, flagged, threshold };

  } catch (e) {
    // Server not running or timeout — fail silently
    return { error: e.message, flagged: false };
  }
}

async function updateStats(wasFlagged) {
  const data     = await chrome.storage.local.get("settings");
  const settings = data.settings || DEFAULT_SETTINGS;
  settings.stats.scanned += 1;
  if (wasFlagged) settings.stats.flagged += 1;
  chrome.storage.local.set({ settings });
}
