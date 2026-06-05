// OCULA — Facebook content script
// Note: Facebook DOM selectors are fragile — Meta updates React frequently

const FB_TEXT_SELECTORS = [
  '[data-ad-preview="message"]',
  'div[dir="auto"][data-content-len]',
  'div[dir="auto"][style*="text-align"]',
];

function getElId(el) {
  return (
    el.getAttribute("data-ft")?.slice(0, 24) ||
    el.closest("[data-pagelet]")?.getAttribute("data-pagelet") ||
    btoa(el.innerText.slice(0, 50)).slice(0, 16)
  );
}

function scanEl(el) {
  const text = el.innerText?.trim();
  if (!text || text.length < 5) return;

  const container = (
    el.closest('[role="article"]') ||
    el.closest("div[data-ft]") ||
    el.parentElement
  );

  const id = getElId(container || el);
  window.OCULA.processPost(container || el, text, `fb_${id}`);
}

function scanAll() {
  FB_TEXT_SELECTORS.forEach(sel => {
    try { document.querySelectorAll(sel).forEach(scanEl); } catch (_) {}
  });
}

let timer;
const debounce = (fn, ms = 1000) => { clearTimeout(timer); timer = setTimeout(fn, ms); };

setTimeout(scanAll, 2500);
const observer = new MutationObserver(() => debounce(scanAll));
observer.observe(document.body, { childList: true, subtree: true });
