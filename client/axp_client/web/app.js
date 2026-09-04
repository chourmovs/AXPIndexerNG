import {initSearch} from './search.js';
import {initAsk} from './ask.js';
import {getBuildInfo, getStartup} from './api.js';

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

const startupStable = new Set(['ready', 'ready_with_warmup_warning', 'failed', 'unconfigured']);
async function refreshStartup() {
  const state = await getStartup(); const searchReady = state.search.state === 'ready';
  document.querySelector('#search-input').disabled = !searchReady;
  document.querySelector('#search-form button').disabled = !searchReady;
  const searchStatus = document.querySelector('#search-startup'); searchStatus.hidden = searchReady;
  if (state.search.state === 'failed') searchStatus.textContent = `Search unavailable · ${state.search.error || 'initialization failed'}`;
  const ai = state.local_ai, name = ai.model_name || 'local model', device = ai.device || 'Intel GPU';
  const labels = {initializing:`○ Local AI starting · ${name}`, probing_gpu:'◐ Checking Intel GPU…',
    loading_model:`◐ Loading ${name} on ${device}…`, warming_model:`◐ Warming ${name}…`,
    unconfigured:'○ Local answer model not configured', failed:`⚠ Local AI initialization failed · ${ai.error || 'Retry from Manage local AI'}`};
  const ready = ai.state === 'ready' || ai.state === 'ready_with_warmup_warning';
  document.querySelector('#ask-health').textContent = ready ? `● Local AI ready · ${name} · ${device}${ai.offloaded_layers != null ? ` · ${ai.offloaded_layers}/${ai.total_layers || '?'} layers offloaded` : ''}` : labels[ai.state] || labels.initializing;
  document.querySelector('#ask-input').disabled = !ready;
  if (!ready) document.querySelector('#ask-submit').disabled = true;
  if (!startupStable.has(state.search.state) || !startupStable.has(ai.state)) setTimeout(() => refreshStartup().catch(() => setTimeout(refreshStartup, 1000)), 850);
}
refreshStartup().catch(() => setTimeout(refreshStartup, 1000));

getBuildInfo().then(info => {
  document.querySelectorAll('[data-build-version]').forEach(node => { node.textContent = info.version; });
  const badge = document.querySelector('#build-version');
  if (info.commit) badge.title = `Build ${info.commit}`;
}).catch(() => { /* The server-side fallback normally makes this unreachable. */ });
