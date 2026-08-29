import {askHealth, askStream} from './api.js';
import {createDocumentActions} from './documents.js';
import {element} from './ui.js';

const progressLabels = {retrieval_started: 'Searching indexed documents…', retrieval_complete: 'Checking whether the evidence is sufficient…',
  gate_complete: 'Evidence check complete…', context_ready: 'Preparing evidence…', generation_started: 'Generating locally…',
  validation_started: 'Validating citations…'};
const errors = {chat_busy: 'AXP is already generating an answer. Please wait for the current question to finish.',
  local_generation_failed: 'The local answer model could not complete this request.',
  chat_model_unavailable: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  model_missing: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  model_invalid: 'The configured local model is invalid.', model_load_failed: 'The local model could not be loaded.'};

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
  let checked = false, busy = false, turn = 0;
  input.addEventListener('input', () => { submit.disabled = busy || !input.value.trim(); });
  input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (!submit.disabled) form.requestSubmit(); } });
  document.querySelector('#clear-chat').addEventListener('click', () => { history.replaceChildren(); progress.textContent = ''; });
  form.addEventListener('submit', async event => { event.preventDefault(); const question = input.value.trim(); if (!question || busy) return;
    busy = true; submit.disabled = true; input.disabled = true; turn += 1; const current = turn;
    const user = element('article', 'turn user-turn'); user.append(element('h3', '', 'You'), element('p', '', question));
    const axp = element('article', 'turn axp-turn'); axp.append(element('h3', '', 'AXP')); history.append(user, axp); input.value = '';
    try { await askStream(question, message => { if (progressLabels[message.event]) {
          progress.textContent = message.event === 'generation_started' && !message.model_was_loaded ? 'Loading local model and generating locally…' : progressLabels[message.event];
        } else if (message.event === 'final') { progress.textContent = ''; renderResponse(axp, message.response, current); }
        else if (message.event === 'error') throw Object.assign(new Error(errors[message.error] || 'AXP could not complete this request.'), {code: message.error}); });
    } catch (exception) { progress.textContent = ''; axp.append(element('p', 'inline-error', errors[exception.code] || exception.message || 'AXP could not complete this request.')); }
    finally { busy = false; input.disabled = false; submit.disabled = !input.value.trim(); input.focus(); }
  });
  return {open: async () => { if (checked) return; checked = true; try { const state = await askHealth();
      if (state.available) health.textContent = state.model_loaded ? `● Local AI ready${state.model_name ? ` · ${state.model_name}` : ''}` : '● Local AI ready · Model will load on first answer';
      else health.textContent = state.reason === 'model_invalid' ? '⚠ Local model is invalid' : state.reason === 'model_load_failed' ? '⚠ Local model could not be loaded' : '○ Local answer model not configured';
    } catch (_) { health.textContent = '⚠ Local AI health is unavailable'; } }};
}
