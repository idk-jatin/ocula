// OCULA — Reddit content script
// Supports both new Reddit and old Reddit

const COMMENT_SELECTORS = [
  "shreddit-comment",                  // new Reddit (web components)
  '[data-testid="comment"]',           // new Reddit posts
  ".Comment",                          // old Reddit
  ".thing.comment",                    // old Reddit fallback
];

function getPostId(el) {
  return (
    el.getAttribute("thingid") ||
    el.getAttribute("id") ||
    el.querySelector("a[href*='/comments/']")?.href.match(/comments\/([^/?#]+)/)?.[1] ||
    btoa(el.innerText.slice(0, 50)).slice(0, 16)
  );
}

function getPostText(el) {
  const tries = [
    "[slot='text-body']",
    ".RichTextJSON-root",
    ".md p",
    ".usertext-body .md",
    "p",
  ];
  for (const sel of tries) {
    const found = el.querySelector(sel);
    if (found?.innerText?.trim()) return found.innerText.trim();
  }
  return null;
}

function scanEl(el) {
  const id   = getPostId(el);
  const text = getPostText(el);
  if (!id || !text) return;
  window.OCULA.processPost(el, text, `rd_${id}`);
}

function scanAll() {
  COMMENT_SELECTORS.forEach(sel => {
    try { document.querySelectorAll(sel).forEach(scanEl); } catch (_) {}
  });
}

let timer;
const debounce = (fn, ms = 800) => { clearTimeout(timer); timer = setTimeout(fn, ms); };

setTimeout(scanAll, 1500);
const observer = new MutationObserver(() => debounce(scanAll));
observer.observe(document.body, { childList: true, subtree: true });
