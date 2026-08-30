import {askHealth, askStream, retryAskModel, localModels, modelAction} from './api.js';
import {createDocumentActions} from './documents.js';
import {element} from './ui.js';

const progressLabels = {retrieval_started: 'Searching indexed documents…', retrieval_complete: 'Retrieved candidate passages…',
  model_load_started: 'Loading local AI model…', model_load_heartbeat: 'Loading local AI model…', model_load_complete: 'Local AI model loaded.',
  context_preparation_started: 'Preparing evidence…', context_ready: 'Evidence prepared.', generation_started: 'Generating answer locally…',
  generation_heartbeat: 'Generating answer locally…', generation_complete: 'Local generation complete.', validation_started: 'Validating citations…'};
const errors = {chat_busy: 'AXP is already generating an answer. Please wait for the current question to finish.',
  local_generation_failed: 'The local answer model could not complete this request.',
  chat_model_unavailable: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  model_missing: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  model_invalid: 'The configured local model is invalid.', model_load_failed: 'The local answer model could not be loaded.',
  context_preparation_failed: 'AXP could not prepare the indexed evidence.', validation_failed: 'AXP could not validate the local answer.',
  stream_incomplete: 'AXP lost the local processing stream unexpectedly.', stream_internal_error: 'AXP could not complete this request.'};

function renderAnswerText(container, answer, turnId) {
  const matcher = /\[S(\d+)\]/g; let offset = 0;
  for (const match of String(answer || '').matchAll(matcher)) { container.append(document.createTextNode(answer.slice(offset, match.index)));
    const link = element('button', 'citation-chip', `S${match[1]}`); link.type = 'button';
    link.addEventListener('click', () => { const source = document.querySelector(`#source-${turnId}-S${match[1]}`); if (!source) return;
      source.scrollIntoView({behavior: 'smooth', block: 'center'}); source.classList.add('highlighted'); setTimeout(() => source.classList.remove('highlighted'), 1600); });
    container.append(link); offset = match.index + match[0].length; }
  container.append(document.createTextNode(String(answer || '').slice(offset)));
}
function renderDocuments(turn, heading, rows, sourceCards = false) {
  if (!rows?.length) return; const section = element('section', 'evidence'); section.append(element('h4', '', heading));
  for (const row of rows) { const card = element('article', 'source-card'); if (sourceCards) card.id = `source-${turn}-${row.id}`;
    const title = row.title || row.filename || row.path || 'Indexed document';
    card.append(element('strong', '', `${sourceCards ? `[${row.id}] ` : ''}${title}`));
    const detail = [row.page_no ? `Page ${row.page_no}` : '', row.section_heading || '', !sourceCards && !row.snippet ? 'Content not indexed' : ''].filter(Boolean).join(' · ');
    if (detail) card.append(element('small', '', detail)); card.append(createDocumentActions(row.document_id)); section.append(card); }
  return section;
}
function renderResponse(article, response, turn) {
  const answer = element('div', 'answer-text');
  if (response.status === 'answered' && response.answerable) renderAnswerText(answer, response.answer, turn);
  else if (response.status === 'ungrounded_generation') answer.textContent = 'AXP found related evidence but could not produce a sufficiently grounded answer. Try rephrasing the question.';
  else answer.textContent = "I couldn't find enough information in the indexed documents to answer this reliably.";
  article.append(answer);
  const documents = response.status === 'answered' ? renderDocuments(turn, 'Sources', response.sources, true) :
    renderDocuments(turn, 'Related documents', response.related_documents, false);
  if (documents) article.append(documents);
  if (response.timings) { const seconds = (response.timings.total_ms / 1000).toFixed(1); const count = response.sources?.length || 0;
    article.append(element('small', 'answer-meta', `${count} source${count === 1 ? '' : 's'} · answered locally in ${seconds} s`)); }
}
export function initAsk() {
  const form = document.querySelector('#ask-form'), input = document.querySelector('#ask-input'), submit = document.querySelector('#ask-submit');
  const history = document.querySelector('#chat-history'), progress = document.querySelector('#ask-progress'), health = document.querySelector('#ask-health');
  let checked = false, busy = false, turn = 0, timer = null, phaseStarted = 0, lastBackend = 0;
  const formatElapsed = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')} elapsed`;
  let liveLabel = '';
  const updateProgress = () => { const elapsed = Math.floor((Date.now() - phaseStarted) / 1000), quiet = Math.floor((Date.now() - lastBackend) / 1000);
    progress.replaceChildren(element('strong', '', liveLabel), element('small', '', formatElapsed(elapsed)),
      element('small', quiet >= 10 ? 'stalled' : '', quiet >= 10 ? `No backend heartbeat for ${quiet} s — processing may be stalled.` : `Backend heartbeat: ${quiet < 1 ? '< 1' : quiet} s ago`)); };
  input.addEventListener('input', () => { submit.disabled = busy || !input.value.trim(); });
  input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (!submit.disabled) form.requestSubmit(); } });
  document.querySelector('#clear-chat').addEventListener('click', () => { history.replaceChildren(); progress.textContent = ''; });
  form.addEventListener('submit', async event => { event.preventDefault(); const question = input.value.trim(); if (!question || busy) return;
    busy = true; submit.disabled = true; input.disabled = true; turn += 1; const current = turn;
    const user = element('article', 'turn user-turn'); user.append(element('h3', '', 'You'), element('p', '', question));
    const axp = element('article', 'turn axp-turn'); axp.append(element('h3', '', 'AXP'), element('p', 'working', 'Working…')); history.append(user, axp); input.value = '';
    phaseStarted = lastBackend = Date.now(); liveLabel = 'Starting local processing…'; updateProgress(); timer = setInterval(updateProgress, 1000);
    try { await askStream(question, message => { lastBackend = Date.now();
        if (progressLabels[message.event]) { liveLabel = progressLabels[message.event]; updateProgress(); }
        else if (message.event === 'gate_complete') { liveLabel = message.answerable ? 'Evidence is sufficient…' : 'Evidence is insufficient…'; updateProgress(); }
        else if (message.event === 'final') { axp.querySelector('.working')?.remove(); renderResponse(axp, message.response, current); }
        else if (message.event === 'error') throw Object.assign(new Error(errors[message.error] || 'AXP could not complete this request.'), {code: message.error}); });
    } catch (exception) { axp.querySelector('.working')?.remove(); axp.append(element('p', 'inline-error', errors[exception.code] || exception.message || 'AXP could not complete this request.')); }
    finally { clearInterval(timer); progress.replaceChildren();
      if (!axp.querySelector('.answer-text, .inline-error')) axp.append(element('p', 'inline-error', 'AXP could not complete this request.'));
      busy = false; input.disabled = false; submit.disabled = !input.value.trim(); input.focus(); }
  });
  const manager = document.querySelector('#model-manager'), list = document.querySelector('#model-list');
  document.querySelector('#manage-ai').addEventListener('click', async () => { manager.hidden = !manager.hidden; if (manager.hidden) return;
    const catalog = await localModels(); list.replaceChildren(...catalog.models.map(model => {
      const card = element('article', 'model-card'); const title = element('strong', '', `${model.name}${model.active ? ' · ACTIVE' : ''}`);
      const details = element('p', 'muted', `${model.profile === 'fast' ? 'Fast · Recommended for standard workstations' : 'Balanced'} · ${model.display_size} · ${model.license}`);
      const source = element('small', '', `${model.repository} · ${model.quantization}`); const button = element('button', 'compact', model.installed ? (model.active ? 'Active' : 'Activate') : 'Download & activate');
      button.disabled = model.active; button.addEventListener('click', async () => { if (!model.installed && !confirm(`Download ${model.name}?\n\nSize: approximately ${model.display_size}\nSource: approved Hugging Face model repository\nStored locally in AXP model cache`)) return;
        await modelAction(model.id, model.installed ? 'activate' : 'download', {activate: true}); await refreshHealth(); });
      if (model.download) { const progress = element('progress'); progress.max=100; progress.value=model.download.percentage; card.append(title, details, source, progress, element('small','',`${model.download.percentage.toFixed(1)}% · ${formatRate(model.download.bytes_per_second)} · ${formatEta(model.download.eta_seconds)}`),button); }
      else card.append(title, details, source, button); return card; })); });
  function formatRate(value){ return value ? `${(value/1048576).toFixed(1)} MB/s` : 'Connecting'; }
  function formatEta(value){ return value == null ? '' : `about ${Math.ceil(value)} s remaining`; }
  async function refreshHealth(){ try { const state = await askHealth();
      if (state.model_state === 'loaded') health.textContent = `● Local AI ready · Model loaded${state.model_name ? ` · ${state.model_name}` : ''}`;
      else if (state.model_state === 'loading') health.textContent = '● Loading local model…';
      else if (state.model_state === 'failed') { health.replaceChildren(document.createTextNode('⚠ Local model load failed ')); const retry = element('button', 'retry-model', 'Retry model');
        retry.type = 'button'; retry.addEventListener('click', async () => { retry.disabled = true; try { await retryAskModel(); health.textContent = '● Local model detected · Not loaded yet'; } catch (_) { retry.disabled = false; } }); health.append(retry); }
      else if (state.available) health.textContent = `● Local model detected · Not loaded yet${state.model_name ? ` · ${state.model_name}` : ''}`;
      else health.textContent = state.reason === 'model_invalid' ? '⚠ Local model is invalid' : '○ Local answer model not configured';
    } catch (_) { health.textContent = '⚠ Local AI health is unavailable'; } }
  let healthTimer; return {open: async () => { checked = true; await refreshHealth(); clearInterval(healthTimer); healthTimer=setInterval(() => { if (!document.querySelector('#ask-panel').hidden) refreshHealth(); }, 7500); }};
}
