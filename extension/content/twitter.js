// OCULA — Twitter / X content script

const TWEET_SEL      = 'article[data-testid="tweet"]';
const TWEET_TEXT_SEL = '[data-testid="tweetText"]';

function getTweetId(article) {
  const link = article.querySelector('a[href*="/status/"]');
  if (link) {
    const m = link.href.match(/status\/(\d+)/);
    if (m) return m[1];
  }
  // Fallback: hash of first 60 chars of text
  const t = article.querySelector(TWEET_TEXT_SEL);
  return t ? btoa(t.innerText.slice(0, 60)).slice(0, 20) : null;
}

function scanTweet(article) {
  const id = getTweetId(article);
  if (!id) return;

  const textEl = article.querySelector(TWEET_TEXT_SEL);
  if (!textEl) return;

  const text = textEl.innerText.trim();
  if (!text) return;

  window.OCULA.processPost(article, text, `tw_${id}`);
}

function scanAll() {
  document.querySelectorAll(TWEET_SEL).forEach(scanTweet);
}

// Debounce
let timer;
const debounce = (fn, ms = 700) => { clearTimeout(timer); timer = setTimeout(fn, ms); };

// Initial scan
setTimeout(scanAll, 1500);

// Watch for new tweets as user scrolls
const observer = new MutationObserver(() => debounce(scanAll));
observer.observe(document.body, { childList: true, subtree: true });
