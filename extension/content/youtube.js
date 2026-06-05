// OCULA — YouTube content script (comments section)

const COMMENT_SEL = "ytd-comment-renderer";
const TEXT_SEL    = "#content-text";

function getCommentId(el) {
  // YouTube comment IDs are in the rendered anchor
  const anchor = el.querySelector("a[id]");
  if (anchor?.id) return anchor.id;
  const text = el.querySelector(TEXT_SEL);
  return text ? btoa(text.innerText.slice(0,50)).slice(0,16) : null;
}

function scanComment(el) {
  const id = getCommentId(el);
  if (!id) return;

  const textEl = el.querySelector(TEXT_SEL);
  if (!textEl) return;

  const text = textEl.innerText.trim();
  if (!text) return;

  window.OCULA.processPost(el, text, `yt_${id}`);
}

function scanAll() {
  document.querySelectorAll(COMMENT_SEL).forEach(scanComment);
}

let timer;
const debounce = (fn, ms = 900) => { clearTimeout(timer); timer = setTimeout(fn, ms); };

// YouTube loads comments lazily on scroll — we need MutationObserver
const observer = new MutationObserver(() => debounce(scanAll));
observer.observe(document.body, { childList: true, subtree: true });

// Also do a delayed first scan
setTimeout(scanAll, 3000);
