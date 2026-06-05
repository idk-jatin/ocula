// Shared overlay logic — loaded before every content script
// Handles: API calls, badge injection, word highlighting

window.OCULA = {

  scanned: new Set(),   // post IDs already processed
  pending: new Set(),   // requests in-flight

  // ── Main entry point called by each content script ──────────────
  processPost(postEl, text, postId) {
    if (!text || text.trim().length < 5) return;
    if (this.scanned.has(postId))        return;
    if (this.pending.has(postId))        return;

    this.pending.add(postId);

    chrome.runtime.sendMessage(
      { type: "PREDICT", text: text.trim() },
      (response) => {
        this.pending.delete(postId);

        if (chrome.runtime.lastError) return;
        if (!response || response.skip || response.error) {
          this.scanned.add(postId);
          return;
        }

        if (response.flagged) {
          this.injectOverlay(postEl, response);
        }

        this.scanned.add(postId);
      }
    );
  },

  // ── Inject badge + border ────────────────────────────────────────
  injectOverlay(postEl, result) {
    if (!postEl || postEl.querySelector(".ocula-badge")) return;

    const label  = result.label;       // "hate" | "offensive"
    const conf   = Math.round(result.confidence * 100);
    const tokens = (result.top_tokens || []).filter(t => t.score > 0.2).slice(0, 5);

    // 1. Left border
    postEl.classList.add(
      label === "hate" ? "ocula-border-hate" : "ocula-border-offensive"
    );

    // 2. Badge
    const badge = document.createElement("span");
    badge.className = `ocula-badge ocula-badge-${label}`;

    const icon  = label === "hate" ? "🔴" : "🟡";
    const title = label === "hate" ? "Hate" : "Offensive";

    const tokensHTML = tokens.length
      ? tokens.map(t =>
          `<span class="ocula-token ocula-token-${label}">${escHtml(t.word)}</span>`
        ).join(" ")
      : `<span style="color:#64748B;font-size:11px">No specific triggers identified</span>`;

    badge.innerHTML = `
      ${icon}&nbsp;${title}&nbsp;${conf}%
      <span class="ocula-tooltip">
        <div class="ocula-tooltip-title">Top contributing words</div>
        <div>${tokensHTML}</div>
        <div class="ocula-conf">Confidence: ${conf}% &nbsp;·&nbsp; Powered by MuRIL</div>
      </span>
    `;

    // 3. Insert badge near author/header
    const headerSelectors = [
      '[data-testid="User-Name"]',
      '#header-author',
      '.Comment__author',
      '.author-text',
      '[class*="author"]',
      '[class*="username"]',
      'h3', 'h4',
    ];

    let inserted = false;
    for (const sel of headerSelectors) {
      const header = postEl.querySelector(sel);
      if (header) {
        header.appendChild(badge);
        inserted = true;
        break;
      }
    }

    if (!inserted) {
      // Fallback: absolute position top-right
      postEl.style.position = "relative";
      badge.style.cssText   = "position:absolute;top:8px;right:8px;z-index:9000;";
      postEl.appendChild(badge);
    }

    // 4. Inline word highlights
    if (tokens.length > 0) {
      this.highlightWords(postEl, tokens, label);
    }
  },

  // ── Wrap matching words in highlight spans ───────────────────────
  highlightWords(postEl, tokens, label) {
    const highSet = new Set(tokens.filter(t => t.score >= 0.7).map(t => t.word.toLowerCase()));
    const medSet  = new Set(tokens.filter(t => t.score >= 0.4 && t.score < 0.7).map(t => t.word.toLowerCase()));

    if (highSet.size === 0 && medSet.size === 0) return;

    // Target text containers (avoid messing with badge/tooltip)
    const candidates = postEl.querySelectorAll(
      '[data-testid="tweetText"], #content-text, .md, ' +
      'div[lang], p[lang], [class*="comment-body"], ' +
      '[class*="post-body"], [class*="message"]'
    );

    const container = candidates.length ? candidates[0] : null;
    if (!container) return;

    this._walkAndWrap(container, highSet, medSet, label);
  },

  _walkAndWrap(root, highSet, medSet, label) {
    // Collect text nodes first (avoid live-collection issues)
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const p = node.parentElement;
        if (!p) return NodeFilter.FILTER_REJECT;
        const tag = p.tagName;
        if (["SCRIPT","STYLE","NOSCRIPT"].includes(tag)) return NodeFilter.FILTER_REJECT;
        if (p.closest(".ocula-badge, .ocula-tooltip, .ocula-word-high, .ocula-word-med"))
          return NodeFilter.FILTER_REJECT;
        if (node.textContent.trim().length < 2) return NodeFilter.FILTER_SKIP;
        return NodeFilter.FILTER_ACCEPT;
      }
    });

    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    textNodes.forEach(node => {
      const parts = node.textContent.split(/(\s+)/);
      let changed  = false;
      const frag   = document.createDocumentFragment();

      parts.forEach(part => {
        const key = part.toLowerCase().replace(/[^a-z0-9\u0900-\u097F]/gi, "");
        if (key && highSet.has(key)) {
          const s = document.createElement("span");
          s.className  = `ocula-word-high ${label}`;
          s.textContent = part;
          frag.appendChild(s);
          changed = true;
        } else if (key && medSet.has(key)) {
          const s = document.createElement("span");
          s.className  = `ocula-word-med ${label}`;
          s.textContent = part;
          frag.appendChild(s);
          changed = true;
        } else {
          frag.appendChild(document.createTextNode(part));
        }
      });

      if (changed) node.parentNode.replaceChild(frag, node);
    });
  }
};

function escHtml(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
