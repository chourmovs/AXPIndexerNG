import {askHealth, askStream, cancelAskGeneration, retryAskModel, localModels, modelAction, setInferenceDevice,
  downloadIntelRuntime, cancelIntelRuntimeDownload, retryIntelProbe, removeIntelRuntime,
  startIntelBenchmark, cancelIntelBenchmark} from './api.js';
import {createDocumentActions} from './documents.js';
import {element} from './ui.js';

const progressLabels = {retrieval_started: 'Searching indexed documents…', retrieval_complete: 'Retrieved candidate passages…',
  model_load_started: 'Loading local AI model…', model_load_heartbeat: 'Loading local AI model…', model_load_complete: 'Local AI model loaded.',
  model_load_progress: 'Starting Intel GPU backend…',
  context_preparation_started: 'Preparing evidence…', context_ready: 'Evidence prepared', generation_started: 'Evaluating prompt locally…',
  context_reduced_for_latency: 'Reducing context to meet latency target…',
  generation_skipped: 'Relevant evidence ready; local answer skipped.',
  generation_complete: 'Local generation complete.', validation_started: 'Validating citations…'};
const errors = {chat_busy: 'AXP is already generating an answer. Please wait for the current question to finish.',
  local_generation_failed: 'The local answer model could not complete this request.',
  chat_model_unavailable: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  backend_cpu_incompatible: 'This AXP build requires CPU instructions unavailable on this PC.',
  backend_missing: 'The local AI runtime is not installed correctly.',
  model_missing: 'Ask AXP requires a locally provisioned GGUF model. Document Search remains fully available.',
  model_invalid: 'The configured local model is invalid.', model_load_failed: 'The local answer model could not be loaded.',
  model_template_incompatible: 'The selected model uses a chat template that is incompatible with this AXP runtime.',
  context_preparation_failed: 'AXP could not prepare the indexed evidence.', validation_failed: 'AXP could not validate the local answer.',
  stream_incomplete: 'AXP lost the local processing stream unexpectedly.', stream_internal_error: 'AXP could not complete this request.'};
const downloadErrors = {network_error: 'Download blocked or unavailable on this network. You can still import an approved local GGUF file.',
  tls_error: 'The secure connection could not be verified. AXP did not weaken TLS validation.',
  integrity_mismatch: 'Download verification failed. The file did not match the release catalog and was not installed.',
  invalid_gguf: 'The downloaded file is not a valid GGUF model and was not installed.',
  insufficient_disk: 'There is not enough free disk space for this model.',
  download_cancelled: 'Download cancelled. The verified partial data can be resumed later.'};
const activeDownloadStates = new Set(['queued', 'connecting', 'downloading', 'verifying', 'installing', 'probing']);

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
  else if (response.status === 'local_generation_skipped_latency_budget') answer.textContent = 'Local answer skipped because estimated generation latency exceeds the interactive budget. Relevant evidence is shown instead.';
  else if (response.status === 'ungrounded_generation') answer.textContent = 'AXP found related evidence but could not produce a sufficiently grounded answer. Try rephrasing the question.';
  else if (response.status === 'insufficient_evidence' && response.context?.search_depth !== 1)
    answer.textContent = "I couldn't find enough information in the first search pass to answer this reliably. You can try Search more to inspect a broader evidence set.";
  else answer.textContent = "I couldn't find enough information in the indexed documents to answer this reliably.";
  article.append(answer);
  const documents = ['answered','local_generation_skipped_latency_budget'].includes(response.status) ? renderDocuments(turn, 'Sources', response.sources, true) :
    renderDocuments(turn, 'Related documents', response.related_documents, false);
  if (documents) article.append(documents);
  if (response.timings) { const generation=response.generation || {}; const seconds = ((generation.generation_ms || response.timings.total_ms) / 1000).toFixed(1); const count = response.sources?.length || 0;
    const device=generation.inference_device_effective === 'intel_gpu' ? 'Intel GPU' : generation.inference_device_effective === 'none' ? 'No inference device' : 'CPU';
    const reduced=response.context?.context_reduced_for_latency ? ' · reduced context' : '';
    const expanded=response.context?.search_depth === 1 ? 'Expanded search · ' : '';
    const evidence=response.context?.selected_evidence_tokens ?? '—';
    const meta=element('small', 'answer-meta', `${expanded}${response.context?.selected_documents ?? count} document${(response.context?.selected_documents ?? count) === 1 ? '' : 's'} · ${evidence} evidence tokens · ${device}${reduced} · ${seconds} s`);
    if(generation.completion_tokens != null) meta.append(document.createElement('br'), document.createTextNode(
      `TTFT ${(generation.time_to_first_token_ms/1000).toFixed(1)} s · ${generation.completion_tokens} tokens · ${generation.decode_tokens_per_second.toFixed(1)} tok/s`));
    article.append(meta); }
  if (['answered','insufficient_evidence','ungrounded_generation','local_generation_skipped_latency_budget'].includes(response.status) &&
      response.context?.search_depth !== 1) {
    const more=element('button','secondary compact search-more','Search more'); more.type='button';
    more.addEventListener('click',()=>article.dispatchEvent(new CustomEvent('search-more',{bubbles:true})));
    article.append(more);
  }
}
export function initAsk() {
  const form = document.querySelector('#ask-form'), input = document.querySelector('#ask-input'), submit = document.querySelector('#ask-submit');
  const history = document.querySelector('#chat-history'), progress = document.querySelector('#ask-progress'), health = document.querySelector('#ask-health');
  let checked = false, busy = false, turn = 0, timer = null, phaseStarted = 0, progressMode = 'pipeline', progressData = {};
  const formatElapsed = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')} elapsed`;
  let liveLabel = '';
  const updateProgress = () => { const elapsed = Math.floor(progressData.elapsed_s ?? ((Date.now() - phaseStarted) / 1000));
    const rows=[element('strong', '', liveLabel), element('small', '', formatElapsed(elapsed))];
    if(progressMode==='waiting'){ rows.push(element('small','','Waiting for first token'));
      if(elapsed>=60) rows.push(element('small','stalled','Local CPU inference is very slow on this machine. You can cancel this generation.'));
      else if(elapsed>=10) rows.push(element('small','stalled','Prompt evaluation is taking longer than usual.')); }
    if(progressMode==='generating') rows.push(element('small','',`First token: ${(progressData.time_to_first_token_ms/1000).toFixed(1)} s`),
      element('small','',`${progressData.generated_fragments} output fragments`));
    if(progressMode==='context') rows.push(element('small','',`${Number(progressData.evidence_tokens).toLocaleString()} / ${Number(progressData.evidence_budget_tokens).toLocaleString()} evidence tokens`),
      element('small','',`${progressData.documents} documents · ${progressData.blocks} evidence blocks`));
    if(progressData.cancelButton) rows.push(progressData.cancelButton); progress.replaceChildren(...rows); };
  input.addEventListener('input', () => { submit.disabled = busy || !input.value.trim(); });
  input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (!submit.disabled) form.requestSubmit(); } });
  document.querySelector('#clear-chat').addEventListener('click', () => { history.replaceChildren(); progress.textContent = ''; });
  form.addEventListener('submit', async event => { event.preventDefault(); const question = input.value.trim(); if (!question || busy) return;
    busy = true; submit.disabled = true; input.disabled = true; turn += 1; const current = turn;
    const user = element('article', 'turn user-turn'); user.append(element('h3', '', 'You'), element('p', '', question));
    const axp = element('article', 'turn axp-turn'); axp.dataset.question=question; axp.append(element('h3', '', 'AXP'), element('p', 'working', 'Working…')); history.append(user, axp); input.value = '';
    phaseStarted = Date.now(); progressMode='pipeline'; progressData={}; liveLabel = 'Starting local processing…'; updateProgress(); timer = setInterval(updateProgress, 1000);
    try { await askStream(question, message => {
        if (progressLabels[message.event]) { liveLabel = progressLabels[message.event]; progressMode='pipeline'; progressData={}; updateProgress(); }
        if(message.event==='context_ready'){ liveLabel='Evidence prepared'; progressMode='context'; progressData=message; updateProgress(); }
        else if(message.event==='model_load_progress'){ const labels={spawning:'Starting Intel GPU backend…',runtime_initializing:'Initializing SYCL / Level Zero…',model_opening:'Loading local model…',tensor_loading:'Loading model tensors…',gpu_allocating:'Allocating Intel GPU buffers…',gpu_offloading:'Offloading model to Intel GPU…',waiting_health:'Waiting for local GPU server…'};
          const cancelButton=progressData.cancelButton||element('button','secondary compact','Cancel'); cancelButton.type='button';
          if(!progressData.cancelButton) cancelButton.addEventListener('click',()=>cancelAskGeneration());
          liveLabel=labels[message.phase]||progressLabels.model_load_progress; progressData={...message,cancelButton}; updateProgress(); }
        else if(message.event==='generation_started'){ const cancelButton=element('button','secondary compact','Cancel generation'); cancelButton.type='button';
          cancelButton.addEventListener('click',async()=>{ cancelButton.disabled=true; cancelButton.textContent='Cancellation requested…'; try{await cancelAskGeneration();}catch(_){/* stream remains authoritative */} });
          phaseStarted=Date.now(); liveLabel='Evaluating prompt locally…'; progressMode='waiting'; progressData={cancelButton}; updateProgress(); }
        else if(message.event==='generation_waiting_first_token'){ progressMode='waiting'; progressData={...progressData,...message}; updateProgress(); }
        else if(message.event==='generation_progress'){ liveLabel='Generating locally…'; progressMode='generating'; progressData={...progressData,...message}; updateProgress(); }
        else if (message.event === 'gate_complete') { liveLabel = message.answerable ? 'Evidence is sufficient…' : 'Evidence is insufficient…'; updateProgress(); }
        else if (message.event === 'final') { axp.querySelector('.working')?.remove(); renderResponse(axp, message.response, current); }
        else if (message.event === 'cancelled') { axp.querySelector('.working')?.remove(); axp.append(element('p','generation-cancelled','Generation cancelled.')); }
        else if (message.event === 'error') throw Object.assign(new Error(errors[message.error] || 'AXP could not complete this request.'), {code: message.error}); });
    } catch (exception) { axp.querySelector('.working')?.remove(); axp.append(element('p', 'inline-error', errors[exception.code] || exception.message || 'AXP could not complete this request.')); }
    finally { clearInterval(timer); progress.replaceChildren();
      if (!axp.querySelector('.answer-text, .inline-error, .generation-cancelled')) axp.append(element('p', 'inline-error', 'AXP could not complete this request.'));
      busy = false; input.disabled = false; submit.disabled = !input.value.trim(); input.focus(); }
  });
  history.addEventListener('search-more', async event => {
    const article=event.target.closest('.axp-turn'); const question=article?.dataset.question;
    if (!article || !question || busy) return;
    const button=article.querySelector('.search-more'); button.disabled=true; busy=true;
    progress.replaceChildren(element('strong','','Expanding search…'));
    try { let expanded=null; await askStream(question, message=>{
      const labels={retrieval_complete:'Ranking documents…',context_preparation_started:'Preparing wider evidence…',
        context_ready:'Evaluating expanded context…',generation_started:'Generating expanded answer…'};
      if(labels[message.event]) progress.replaceChildren(element('strong','',labels[message.event]));
      if(message.event==='final') expanded=message.response;
      if(message.event==='error') throw Object.assign(new Error(errors[message.error]||message.error),{code:message.error});
    },1);
      if(expanded){ const heading=article.querySelector('h3'); article.replaceChildren(heading); renderResponse(article,expanded,article.id||turn); }
    } catch(exception) { button.disabled=false; progress.replaceChildren(element('small','inline-error',errors[exception.code]||exception.message)); }
    finally { busy=false; if(article.querySelector('.search-more')) article.querySelector('.search-more').disabled=false;
      if(!progress.querySelector('.inline-error')) progress.replaceChildren(); }
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
        model.model_state === 'loading' ? ' · SELECTED · LOADING' : model.model_state === 'failed' ?
          ` · SELECTED · LOAD FAILED${model.failure_type === 'model_template_incompatible' ? ' — INCOMPATIBLE CHAT TEMPLATE' : ''}` : ' · SELECTED';
      const badge = model.experimental ? 'EXPERIMENTAL' : model.recommended ? 'RECOMMENDED' : model.profile === 'balanced' ? 'BALANCED' : '';
      card.append(element('strong','',`${model.name}${badge ? ` · ${badge}` : ''}${stateLabel}`),
        element('p','muted',`${model.profile === 'fast' ? 'Fast' : 'Balanced'} · ${model.display_size} · ${model.license}`),
        element('small','',`${model.repository} · ${model.quantization}`));
      if (model.experimental) card.append(element('p','muted','Optimized for fast local inference / RAG. Review model license terms before organizational deployment.'));
      const job=model.download;
      if (job && activeDownloadStates.has(job.state)) { downloading=true; const bar=element('progress'); bar.max=100; bar.value=job.percentage;
        card.append(element('strong','download-state',downloadLabel(job.state)),bar,
          element('small','',`${job.percentage.toFixed(1)}% · ${formatBytes(job.bytes_downloaded)} / ${formatBytes(job.bytes_total)} · ${formatRate(job.bytes_per_second)}${formatEta(job.eta_seconds)}`),
          makeButton('Cancel',()=>action(model,'cancel'),true));
      } else { if (job?.state === 'failed' || job?.state === 'cancelled') card.append(element('p','inline-error',downloadErrors[job.error] || `Download ${job.state}.`));
        if (!model.installed) card.append(makeButton(model.partial_bytes ? 'Resume download' : 'Download & activate',()=>{ if (model.partial_bytes || confirmDownload(model)) action(model,'download',{activate:true}); }));
        else if (!model.active) card.append(makeButton('Activate',()=>action(model,'activate')),makeButton('Remove',()=>{ if (confirm(`Remove ${model.name} from AXP?`)) action(model,'remove'); },true));
        else card.append(element('small','active-status',model.model_loaded ? 'Ready' : model.model_state === 'failed' ?
          (model.failure_type === 'model_template_incompatible' ? 'Load failed — incompatible chat template' : 'Load failed') : 'Selected · Not ready')); }
      cards.push(card);
    }
    if (catalog.custom_model) { const custom=element('article','model-card'); custom.append(element('strong','',`Custom local model${catalog.custom_model.active ? ' · ACTIVE' : ''}`),
      element('p','muted',catalog.custom_model.filename),element('small','',catalog.custom_model.installed ? 'Installed' : 'Configured file is missing')); cards.push(custom); }
    list.replaceChildren(...cards); const requested=catalog.device.inference_device_requested || 'auto';
    document.querySelectorAll('input[name="device"]').forEach(radio=>{ radio.checked=radio.value===requested; });
    const intel=document.querySelector('input[name="device"][value="intel_gpu"]'); intel.disabled=!catalog.hardware.accelerator_available;
    const intelProven=catalog.device.inference_device_effective==='intel_gpu';
    document.querySelector('#intel-device-status').textContent=intelProven ? '— Ready — hardware acceleration confirmed' :
      catalog.hardware.intel_gpu_detected ? '— Detected — inference acceleration not confirmed' : `— unavailable (${catalog.hardware.accelerator_reason || 'not installed'})`;
    const accelerator=catalog.hardware.accelerator || {}; const installed=accelerator.installed;
    const acceleratorJob=accelerator.download; const acceleratorDownloadActive=activeDownloadStates.has(acceleratorJob?.state);
    downloading=downloading || acceleratorDownloadActive;
    document.querySelector('#download-intel-runtime').hidden=installed || !catalog.hardware.intel_gpu_detected || acceleratorDownloadActive;
    document.querySelector('#retry-intel-probe').hidden=!installed || catalog.hardware.sycl_probe_ok || acceleratorDownloadActive;
    const cancelRuntime=document.querySelector('#cancel-intel-runtime');
    cancelRuntime.hidden=!['queued','connecting','downloading'].includes(acceleratorJob?.state);
    const runtimeProgress=document.querySelector('#intel-runtime-progress'); runtimeProgress.replaceChildren();
    if(acceleratorDownloadActive){ const phase={queued:'Preparing Intel GPU runtime download…',connecting:'Connecting securely…',downloading:'Downloading Intel GPU runtime…',verifying:'Verifying Intel GPU runtime integrity…',installing:'Installing Intel GPU runtime…',probing:'Detecting Intel SYCL / Level Zero GPU…'}[acceleratorJob.state];
      runtimeProgress.append(element('strong','download-state',phase));
      if(acceleratorJob.state==='downloading'){ const bar=element('progress'); bar.max=100; bar.value=acceleratorJob.percentage;
        runtimeProgress.append(bar,element('small','',`${acceleratorJob.percentage.toFixed(1)}% · ${formatBytes(acceleratorJob.bytes_downloaded)} / ${formatBytes(acceleratorJob.bytes_total)} · ${formatRate(acceleratorJob.bytes_per_second)}${formatEta(acceleratorJob.eta_seconds)}`)); }}
    document.querySelector('#benchmark-intel').hidden=!catalog.hardware.accelerator_available;
    const benchmarkActive=catalog.benchmark && !['idle','complete','complete_with_errors','failed','cancelled'].includes(catalog.benchmark.state);
    document.querySelector('#benchmark-intel').disabled=benchmarkActive;
    document.querySelector('#cancel-benchmark').hidden=!benchmarkActive;
    document.querySelector('#remove-intel-runtime').hidden=!installed;
    document.querySelector('#intel-runtime-description').textContent = !catalog.hardware.intel_gpu_detected ? 'No Intel GPU detected.' :
      installed ? `${catalog.hardware.intel_gpu_name} · Intel GPU runtime installed · ${catalog.hardware.sycl_probe_ok ? 'SYCL / Level Zero available' : probeErrorMessage(catalog.hardware.sycl_probe_error)}` :
      `${catalog.hardware.intel_gpu_name} detected · SYCL runtime not installed · approximately 120 MB · Official llama.cpp Windows SYCL runtime · Experimental`;
    if ((downloading || benchmarkActive) && !manager.hidden) downloadTimer=setTimeout(renderManager,750);
  }
  document.querySelector('#manage-ai').addEventListener('click', async () => { manager.hidden=!manager.hidden; clearTimeout(downloadTimer); if (!manager.hidden) await renderManager(); });
  document.querySelectorAll('input[name="device"]').forEach(radio=>radio.addEventListener('change',async()=>{ try { await setInferenceDevice(radio.value); await renderManager(); await refreshHealth(); }
    catch(exception){ managerError.textContent=exception.code==='intel_gpu_unavailable' ? 'Intel GPU inference is unavailable; CPU remains active.' : exception.message; await renderManager(); } }));
  document.querySelector('#download-intel-runtime').addEventListener('click',async()=>{ try { await downloadIntelRuntime(); await renderManager(); }
    catch(exception){ if(exception.code==='accelerator_download_busy'){ managerError.textContent='Intel GPU runtime download is already in progress.'; await renderManager(); } else managerError.textContent=exception.message; } });
  document.querySelector('#cancel-intel-runtime').addEventListener('click',async()=>{ try { await cancelIntelRuntimeDownload(); await renderManager(); }
    catch(exception){ managerError.textContent=exception.message; } });
  document.querySelector('#retry-intel-probe').addEventListener('click',async()=>{ try { await retryIntelProbe(); await renderManager(); await refreshHealth(); }
    catch(exception){ managerError.textContent=exception.message; } });
  document.querySelector('#remove-intel-runtime').addEventListener('click',async()=>{ try { await removeIntelRuntime(); await renderManager(); }
    catch(exception){ managerError.textContent=exception.message; } });
  document.querySelector('#benchmark-intel').addEventListener('click',async()=>{ try { await startIntelBenchmark('quick'); await renderManager(); }
    catch(exception){ managerError.textContent=exception.message; } });
  document.querySelector('#cancel-benchmark').addEventListener('click',async()=>{ try { await cancelIntelBenchmark(); await renderManager(); }
    catch(exception){ managerError.textContent=exception.message; } });
  function confirmDownload(model){ return confirm(`Download ${model.name}?\n\nSize: approximately ${model.display_size}\nSource: approved Hugging Face model repository\nStored locally in AXP model cache`); }
  function downloadLabel(state){ return {queued:'Queued…',connecting:'Connecting…',downloading:'Downloading…',verifying:'Verifying SHA-256…',installing:'Installing model…'}[state] || state; }
  function probeErrorMessage(code){ return {intel_sycl_probe_timeout:'Intel GPU detection timed out.',intel_sycl_device_not_found:'The SYCL runtime started, but no Intel GPU device was reported.',intel_gpu_driver_or_level_zero_unavailable:'The Intel GPU runtime could not access Level Zero. Check the Intel graphics driver and retry detection.',intel_sycl_probe_command_failed:'The Intel GPU runtime started but device detection failed.',intel_sycl_runtime_invalid:'The installed Intel GPU runtime is incomplete or invalid.'}[code] || 'Intel GPU probe did not succeed'; }
  function formatBytes(value){ return value >= 1e9 ? `${(value/1e9).toFixed(2)} GB` : `${(value/1e6).toFixed(1)} MB`; }
  function formatRate(value){ return value ? `${(value/1048576).toFixed(1)} MB/s` : 'Calculating speed'; }
  function formatEta(value){ if(value==null)return ''; const seconds=Math.ceil(value); return ` · ~${Math.floor(seconds/60)}m ${String(seconds%60).padStart(2,'0')}s remaining`; }
  async function refreshHealth(){ try { const state = await askHealth();
      if (state.model_state === 'loaded') health.textContent = `● Local AI ready · ${state.active_model_name || state.model_name || 'Local model'} · ${state.inference_device_effective === 'intel_gpu' ? `Intel GPU · hardware acceleration confirmed${state.offloaded_layers ? ` · ${state.offloaded_layers}/${state.total_layers || '?'} layers offloaded` : ''}` : state.inference_device_requested === 'intel_gpu' ? 'Intel GPU detected — GPU inference not confirmed' : 'CPU'}`;
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
