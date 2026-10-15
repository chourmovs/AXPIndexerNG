import {searchDocuments, searchProgress, cancelSearch} from './api.js';
import {SearchDiagnostics, SearchGeneration} from './search-diagnostics.js';
import {createDocumentActions} from './documents.js';
import {appendHighlighted, element} from './ui.js';
const stopwords = new Set(['and', 'are', 'the', 'for', 'with', 'les', 'des', 'une', 'dans', 'pour', 'avec']);
function queryTerms(query) { const phrases = [...query.matchAll(/"([^"]+)"/gu)].map(x => x[1].trim()).filter(Boolean);
  const words = query.match(/[\p{L}\p{N}_-]+/gu) || []; return [...new Set([...phrases, ...words.filter(x => x.length >= 3 && !stopwords.has(x.toLocaleLowerCase()))])].sort((a, b) => b.length - a.length); }
const percent = value => Math.round(Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0)) * 100);
function renderResult(row, terms) { const article = element('article', 'result-card'); const header = element('div', 'result-header');
  const title = element('h3'); appendHighlighted(title, row.title || row.filename || row.path, terms);
  const score = element('span', 'relevance', `Passage ${percent(row.passage_score)}%`); const details = [];
  if (row.passage_score != null) details.push(`Passage match: ${percent(row.passage_score)}%`);
  if (row.document_score != null) details.push(`Document score: ${Number(row.document_score).toFixed(3)}`);
  if (row.document_query_coverage != null) details.push(`Query coverage: ${percent(row.document_query_coverage)}%`);
  if (row.document_metadata_coverage != null) details.push(`Metadata coverage: ${percent(row.document_metadata_coverage)}%`);
  if (row.vector_similarity != null) details.push(`Vector similarity: ${percent(row.vector_similarity)}%`);
  if (row.content_lexical_coverage != null) details.push(`Content lexical coverage: ${percent(row.content_lexical_coverage)}%`);
  if (row.title_coverage != null) details.push(`Title coverage: ${percent(row.title_coverage)}%`);
  if (row.filename_coverage != null) details.push(`Filename coverage: ${percent(row.filename_coverage)}%`);
  if (row.convergence_score != null) details.push(`Convergence: ${percent(row.convergence_score)}%`);
  score.title = details.join('\n'); header.append(title, score);
  const path = element('div', 'path', row.path); const source = element('small', '', `Source: ${row.source_label || row.source_path || '—'}`);
  const snippet = element('p'); appendHighlighted(snippet, row.snippet, terms); article.append(header, path, source, snippet, createDocumentActions(row.document_id)); return article; }
function diagnosticSummary(diagnostics) {
  const d = diagnostics || {}; const counts = d.catalog_snapshot_status;
  const age = d.catalog_snapshot_age_ms == null ? '—' : `${(d.catalog_snapshot_age_ms / 1000).toFixed(1)} s`;
  const text = !counts || counts === 'unavailable' ? 'Compteurs indisponibles' :
    `Compteurs échantillonnés${counts === 'aged' ? ' (anciens)' : ''} : ${d.total_chunks ?? '—'} passages, ${d.total_vectors ?? '—'} vecteurs — âge ${age}`;
  return element('small', 'search-diagnostics', `${text}${d.catalog_snapshot_refreshing ? ' — calcul en arrière-plan' : ''}. ` +
    `Portée : ${d.catalog_snapshot_scope || 'catalogue entier'}. ` +
    `Cache identité : ${d.identity_cache?.status || 'non renseigné'}, âge ${d.identity_cache?.age_ms == null ? '—' : `${d.identity_cache.age_ms.toFixed(1)} ms`}. Données de recherche disponibles.`);
}
export function initSearch() {
  const form = document.querySelector('#search-form'), results = document.querySelector('#results');
  const cancel = document.querySelector('#cancel-search'), diagnostic = new SearchDiagnostics();
  const generation = new SearchGeneration(); let active = null;
  const stop = state => {
    if (!state || state.done) return;
    state.done = true; state.controller.abort(); state.pollController?.abort(); clearInterval(state.timer);
    cancelSearch(state.token.requestId).catch(() => {});
  };
  cancel.addEventListener('click', () => {
    const state = active; if (!state || state.done) return;
    stop(state); const cancelledToken = generation.begin(state.token.requestId); cancel.hidden = true;
    state.browser.cancellation_requested = true;
    diagnostic.update(state.trace, {...state.browser, elapsed_ms:performance.now()-state.started});
    results.textContent = 'Recherche annulée.';
    const follow = async attempt => {
      if (attempt >= 20 || !generation.accepts(cancelledToken)) return;
      const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 1000);
      try {
        const trace = await searchProgress(state.token.requestId, controller.signal);
        if (!generation.accepts(cancelledToken)) return;
        diagnostic.update(trace, {...state.browser, elapsed_ms:performance.now()-state.started});
        if (trace.status !== 'running') return;
      } catch (_) { /* The browser cancellation remains effective without progress. */ }
      finally { clearTimeout(timeout); }
      setTimeout(() => void follow(attempt+1), 250);
    };
    void follow(0);
  });
  form.addEventListener('submit', async event => {
    event.preventDefault(); const query = document.querySelector('#search-input').value.trim(); if (!query) return;
    stop(active);
    const requestId = crypto.randomUUID(), token = generation.begin(requestId);
    const state = {token, controller:new AbortController(), started:performance.now(), browser:{},
      trace:{request_id:requestId, status:'running', spans:[]}, done:false, polling:false};
    active = state; diagnostic.reset(requestId); cancel.hidden = false; results.textContent = 'Searching…';
    const refresh = () => {
      if (!generation.accepts(token) || state.done) return;
      state.browser.elapsed_ms = performance.now()-state.started;
      diagnostic.update(state.trace, state.browser);
    };
    const poll = async () => {
      if (state.polling || state.done || !generation.accepts(token)) return;
      state.polling = true; const controller = new AbortController(); state.pollController = controller;
      const timeout = setTimeout(() => controller.abort(), 1500);
      try {
        const trace = await searchProgress(requestId, controller.signal);
        if (!state.done && generation.accepts(token)) { state.trace = trace; state.browser.progress_failed = false; refresh(); }
      } catch (error) { if (generation.accepts(token) && !state.done && error.status !== 404) state.browser.progress_failed = true; }
      finally { clearTimeout(timeout); state.polling = false; }
    };
    state.timer = setInterval(() => { refresh(); void poll(); }, 250);
    const httpStarted = performance.now(); state.browser.browser_pre_request_ms = httpStarted-state.started;
    try {
      const response = await searchDocuments(query, requestId, state.controller.signal);
      if (!generation.accepts(token) || state.done) return;
      state.browser.http_ms = performance.now()-httpStarted;
      const renderStarted = performance.now(), rows = response.results;
      results.replaceChildren(...(rows.length ? rows.map(row => renderResult(row, queryTerms(query))) : [element('p', '', 'No sufficiently relevant result.')]), diagnosticSummary(response.diagnostics));
      state.browser.render_ms = performance.now()-renderStarted;
      state.browser.first_result_timing_method = 'Two animation frames: upper bound after a paint opportunity; render_ms measures DOM preparation only';
      state.trace = response.request_trace;
      state.browser.http_outside_server_ms = Math.max(0, state.browser.http_ms-state.trace.elapsed_ms);
      // Give results a paint opportunity before building the diagnostic tree.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (!generation.accepts(token)) return;
        state.browser.first_result_ms = rows.length ? performance.now()-state.started : null;
        state.browser.elapsed_ms = performance.now()-state.started;
        const measurement = {request:requestId, query, result_count:rows.length,
          browser_response_ms:renderStarted-state.started,
          browser_dom_ready_ms:renderStarted-state.started+state.browser.render_ms,
          browser_first_result_ms:state.browser.first_result_ms,
          first_result_definition:'two animation frames after final result DOM insertion',
          server_timings:response.timings};
        window.axpSearchMeasurement = measurement;
        window.dispatchEvent(new CustomEvent('axp-search-measured', {detail:measurement}));
        diagnostic.update(state.trace, state.browser);
      }));
    } catch (error) {
      if (generation.accepts(token) && !state.done) {
        results.textContent = 'Search is temporarily unavailable.';
        diagnostic.update(error.body?.request_trace || {...state.trace, status:'error', active_stage:null}, {...state.browser, elapsed_ms:performance.now()-state.started});
      }
    } finally {
      state.done = true; clearInterval(state.timer); state.pollController?.abort();
      if (generation.accepts(token)) cancel.hidden = true;
    }
  });
}
