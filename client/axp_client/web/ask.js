import {askHealth, askStream, retryAskModel, localModels, modelAction, setInferenceDevice} from './api.js';
import {createDocumentActions} from './documents.js';
import {element} from './ui.js';

const progressLabels = {retrieval_started: 'Searching indexed documents…', retrieval_complete: 'Retrieved candidate passages…',
  model_load_started: 'Loading local AI model…', model_load_heartbeat: 'Loading local AI model…', model_load_complete: 'Local AI model loaded.',
  context_preparation_started: 'Preparing evidence…', context_ready: 'Evidence prepared.', generation_started: 'Generating answer locally…',
  generation_heartbeat: 'Generating answer locally…', generation_complete: 'Local generation complete.', validation_started: 'Validating citations…'};
const errors = {chat_busy: 'AXP is already generating an answer. Please wait for the current question to finish.',
  local_generation_failed: 'The local answer model could not complete this request.',
  chat_model_unavailable: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  backend_cpu_incompatible: 'This AXP build requires CPU instructions unavailable on this PC.',
  backend_missing: 'The local AI runtime is not installed correctly.',
  model_missing: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  model_invalid: 'The configured local model is invalid.', model_load_failed: 'The local answer model could not be loaded.',
  context_preparation_failed: 'AXP could not prepare the indexed evidence.', validation_failed: 'AXP could not validate the local answer.',
  stream_incomplete: 'AXP lost the local processing stream unexpectedly.', stream_internal_error: 'AXP could not complete this request.'};
const downloadErrors = {network_error: 'Download blocked or unavailable on this network. You can still import an approved local GGUF file.',
  tls_error: 'The secure connection could not be verified. AXP did not weaken TLS validation.',
  integrity_mismatch: 'Download verification failed. The file did not match the release catalog and was not installed.',
  invalid_gguf: 'The downloaded file is not a valid GGUF model and was not installed.',
  insufficient_disk: 'There is not enough free disk space for this model.',
  download_cancelled: 'Download cancelled. The verified partial data can be resumed later.'};
const activeDownloadStates = new Set(['queued', 'connecting', 'downloading', 'verifying', 'installing']);

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
  const managerError = document.querySelector('#model-manager-error'); let downloadTimer;
  const action = async (model, name, body={}) => { managerError.textContent = ''; try { await modelAction(model.id, name, body); await renderManager(); await refreshHealth(); }
    catch (exception) { managerError.textContent = downloadErrors[exception.code] || exception.message; } };
  const makeButton = (label, handler, secondary=false) => { const button=element('button', `${secondary ? 'secondary ' : ''}compact`, label); button.type='button'; button.addEventListener('click', handler); return button; };
  async function renderManager(){
    clearTimeout(downloadTimer); const catalog = await localModels(); const cards=[]; let downloading=false;
    for (const model of catalog.models) { const card=element('article','model-card');
      const stateLabel = !model.selected ? '' : model.model_loaded ? ' · ACTIVE · READY' :
        model.model_state === 'loading' ? ' · SELECTED · LOADING' : model.model_state === 'failed' ? ' · SELECTED · LOAD FAILED' : ' · SELECTED';
      card.append(element('strong','',`${model.name}${stateLabel}`),
        element('p','muted',`${model.profile === 'fast' ? 'Fast · Recommended for standard workstations' : 'Balanced'} · ${model.display_size} · ${model.license}`),
        element('small','',`${model.repository} · ${model.quantization}`));
      const job=model.download;
      if (job && activeDownloadStates.has(job.state)) { downloading=true; const bar=element('progress'); bar.max=100; bar.value=job.percentage;
        card.append(element('strong','download-state',downloadLabel(job.state)),bar,
          element('small','',`${job.percentage.toFixed(1)}% · ${formatBytes(job.bytes_downloaded)} / ${formatBytes(job.bytes_total)} · ${formatRate(job.bytes_per_second)}${formatEta(job.eta_seconds)}`),
          makeButton('Cancel',()=>action(model,'cancel'),true));
      } else { if (job?.state === 'failed' || job?.state === 'cancelled') card.append(element('p','inline-error',downloadErrors[job.error] || `Download ${job.state}.`));
        if (!model.installed) card.append(makeButton(model.partial_bytes ? 'Resume download' : 'Download & activate',()=>{ if (model.partial_bytes || confirmDownload(model)) action(model,'download',{activate:true}); }));
        else if (!model.active) card.append(makeButton('Activate',()=>action(model,'activate')),makeButton('Remove',()=>{ if (confirm(`Remove ${model.name} from AXP?`)) action(model,'remove'); },true));
        else card.append(element('small','active-status',model.model_loaded ? 'Ready' : model.model_state === 'failed' ? 'Load failed' : 'Selected · Not ready')); }
      cards.push(card);
    }
    if (catalog.custom_model) { const custom=element('article','model-card'); custom.append(element('strong','',`Custom local model${catalog.custom_model.active ? ' · ACTIVE' : ''}`),
      element('p','muted',catalog.custom_model.filename),element('small','',catalog.custom_model.installed ? 'Installed' : 'Configured file is missing')); cards.push(custom); }
    list.replaceChildren(...cards); const requested=catalog.device.inference_device_requested || 'auto';
    document.querySelectorAll('input[name="device"]').forEach(radio=>{ radio.checked=radio.value===requested; });
    const intel=document.querySelector('input[name="device"][value="intel_gpu"]'); intel.disabled=!catalog.hardware.accelerator_available;
    document.querySelector('#intel-device-status').textContent=catalog.hardware.accelerator_available ? '— available' : `— unavailable (${catalog.hardware.accelerator_reason || 'not installed'})`;
    if (downloading && !manager.hidden) downloadTimer=setTimeout(renderManager,750);
  }
  document.querySelector('#manage-ai').addEventListener('click', async () => { manager.hidden=!manager.hidden; clearTimeout(downloadTimer); if (!manager.hidden) await renderManager(); });
  document.querySelectorAll('input[name="device"]').forEach(radio=>radio.addEventListener('change',async()=>{ try { await setInferenceDevice(radio.value); await renderManager(); await refreshHealth(); }
    catch(exception){ managerError.textContent=exception.code==='intel_gpu_unavailable' ? 'Intel GPU inference is unavailable; CPU remains active.' : exception.message; await renderManager(); } }));
  function confirmDownload(model){ return confirm(`Download ${model.name}?\n\nSize: approximately ${model.display_size}\nSource: approved Hugging Face model repository\nStored locally in AXP model cache`); }
  function downloadLabel(state){ return {queued:'Queued…',connecting:'Connecting…',downloading:'Downloading…',verifying:'Verifying SHA-256…',installing:'Installing model…'}[state] || state; }
  function formatBytes(value){ return value >= 1e9 ? `${(value/1e9).toFixed(2)} GB` : `${(value/1e6).toFixed(1)} MB`; }
  function formatRate(value){ return value ? `${(value/1048576).toFixed(1)} MB/s` : 'Calculating speed'; }
  function formatEta(value){ if(value==null)return ''; const seconds=Math.ceil(value); return ` · ~${Math.floor(seconds/60)}m ${String(seconds%60).padStart(2,'0')}s remaining`; }
  async function refreshHealth(){ try { const state = await askHealth();
      if (state.model_state === 'loaded') health.textContent = `● Local AI ready · ${state.active_model_name || state.model_name || 'Local model'} · ${state.inference_device_effective === 'intel_gpu' ? 'Intel GPU' : 'CPU'}${state.fallback_reason ? ' · Intel GPU unavailable' : ''}`;
      else if (state.model_state === 'loading') health.textContent = '● Loading local model…';
      else if (state.model_state === 'failed') { health.replaceChildren(document.createTextNode(`⚠ ${state.failure_reason || 'Local model load failed'} `));
        if (state.retryable === true) { const retry = element('button', 'retry-model', 'Retry model');
          retry.type = 'button'; retry.addEventListener('click', async () => { retry.disabled = true; try { await retryAskModel(); health.textContent = '● Local model selected · Not ready'; } catch (_) { retry.disabled = false; } }); health.append(retry); } }
      else if (state.available) health.textContent = `● Local model detected · Not loaded yet${state.model_name ? ` · ${state.model_name}` : ''}`;
      else if (state.reason === 'backend_cpu_incompatible') health.textContent = `⚠ ${errors.backend_cpu_incompatible}`;
      else health.textContent = state.reason === 'model_invalid' ? '⚠ Local model is invalid' : '○ Local answer model not configured';
    } catch (_) { health.textContent = '⚠ Local AI health is unavailable'; } }
  let healthTimer; return {open: async () => { checked = true; await refreshHealth(); clearInterval(healthTimer); healthTimer=setInterval(() => { if (!document.querySelector('#ask-panel').hidden) refreshHealth(); }, 7500); }};
}
