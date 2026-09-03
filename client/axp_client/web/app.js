import {initSearch} from './search.js';
import {initAsk} from './ask.js';
import {getBuildInfo} from './api.js';

const searchTab = document.querySelector('#search-tab');
const askTab = document.querySelector('#ask-tab');
const searchPanel = document.querySelector('#search-panel');
const askPanel = document.querySelector('#ask-panel');
const ask = initAsk();

function selectMode(mode) {
  const asking = mode === 'ask';
  searchPanel.hidden = asking; askPanel.hidden = !asking;
  searchTab.classList.toggle('active', !asking); askTab.classList.toggle('active', asking);
  searchTab.setAttribute('aria-selected', String(!asking)); askTab.setAttribute('aria-selected', String(asking));
  if (asking) ask.open();
}
searchTab.addEventListener('click', () => selectMode('search'));
askTab.addEventListener('click', () => selectMode('ask'));
initSearch();

getBuildInfo().then(info => {
  document.querySelectorAll('[data-build-version]').forEach(node => { node.textContent = info.version; });
  const badge = document.querySelector('#build-version');
  if (info.commit) badge.title = `Build ${info.commit}`;
}).catch(() => { /* The server-side fallback normally makes this unreachable. */ });
