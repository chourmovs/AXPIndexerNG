import {element} from './ui.js';

const milliseconds = value => Number.isFinite(value) ? `${value.toFixed(1)} ms` : 'Non mesuré';
const statuses = {running:'En cours', completed:'Terminé', error:'Erreur', cancelled:'Annulé', skipped:'Ignoré'};

export class SearchDiagnostics {
  constructor() {
    this.panel = document.querySelector('#search-diagnostics');
    this.summary = document.querySelector('#search-diagnostic-summary');
    this.tree = document.querySelector('#search-diagnostic-tree');
    this.expanded = new Set(); this.data = null;
    try { this.panel.open = localStorage.getItem('searchDiagnosticCollapsed') !== 'true'; } catch (_) { /* optional storage */ }
    this.panel.addEventListener('toggle', () => {
      try { localStorage.setItem('searchDiagnosticCollapsed', String(!this.panel.open)); } catch (_) { /* optional storage */ }
    });
    document.querySelector('#copy-search-diagnostic').addEventListener('click', async () => {
      const text = JSON.stringify(this.data, null, 2);
      const fallback = document.querySelector('#search-diagnostic-json');
      try { await navigator.clipboard.writeText(text); fallback.hidden = true; }
      catch (_) { fallback.value = text; fallback.hidden = false; fallback.focus(); fallback.select(); }
    });
  }
  reset(requestId) {
    this.expanded.clear();
    const fallback = document.querySelector('#search-diagnostic-json'); fallback.hidden = true; fallback.value = '';
    this.update({request_id:requestId, status:'running', spans:[]}, {});
  }
  update(trace, browser) {
    this.data = {server:trace, browser};
    const lines = [
      `Requête ${trace.request_id} · ${trace.build?.version || 'version en attente'} · commit ${trace.build?.commit || 'inconnu'}`,
      `Premier résultat rendu : ${milliseconds(browser.first_result_ms)} · navigateur/HTTP : ${milliseconds(browser.http_ms)} · préparation du rendu : ${milliseconds(browser.render_ms)}`,
      `Serveur : ${milliseconds(trace.elapsed_ms)} · ${statuses[trace.status] || trace.status}${trace.active_stage ? ` · ${trace.active_stage} (${milliseconds(trace.active_stage_elapsed_ms)})` : ''}`,
      `Temps écoulé côté navigateur : ${milliseconds(browser.elapsed_ms)} · attente HTTP hors serveur : ${milliseconds(browser.http_outside_server_ms)}`,
      trace.server_scope || 'Périmètre serveur : en attente de la première progression.',
      'HTTP côté navigateur inclut transport et décodage JSON. L’attente hors serveur est un résidu, pas une mesure de réseau pur. Les horloges restent séparées.',
    ];
    if (browser.progress_failed) lines.push('Canal de progression indisponible ; la recherche continue.');
    if (browser.cancellation_requested) lines.push('Requête annulée côté navigateur ; annulation serveur demandée. Le statut serveur est celui de la dernière progression reçue.');
    if (trace.truncated) lines.push('Trace tronquée : limite de mémoire atteinte.');
    if (trace.catalog) {
      const c = trace.catalog, identity = c.identity_cache;
      lines.push(c.catalog_snapshot_status === 'unavailable' ? 'Compteurs indisponibles ; calcul en arrière-plan.' :
        `Compteurs échantillonnés : ${c.total_chunks ?? '?'} passages, ${c.total_vectors ?? '?'} vecteurs · état ${c.catalog_snapshot_status || 'inconnu'} · âge ${milliseconds(c.catalog_snapshot_age_ms)}.`);
      lines.push(`Portée des compteurs : ${c.catalog_snapshot_scope || 'non renseignée'}. Données de recherche : ${c.search_data?.status || 'non renseigné'}.`);
      if (identity) lines.push(`Cache identité : ${identity.status} · âge ${milliseconds(identity.age_ms)} · génération ${identity.generation || '—'} · ${identity.invalidation_reason || 'sans invalidation'}.`);
    }
    if (trace.spans?.length) lines.push('Coûts propres les plus élevés : ' + [...trace.spans].sort((a,b) => b.self_ms-a.self_ms)
      .slice(0,3).map(row => `${row.name} (${milliseconds(row.self_ms)})`).join(' ; '));
    this.summary.replaceChildren(...lines.map(text => element('p', '', text)));
    const rows = trace.spans || [];
    const costly = new Set([...rows].sort((a,b) => b.self_ms-a.self_ms).slice(0,3).map(row => row.span_id));
    const branch = parent => rows.filter(row => row.parent_span_id === parent).map(row => {
      const details = element('details', costly.has(row.span_id) ? 'costly' : '');
      details.open = this.expanded.has(row.span_id);
      details.addEventListener('toggle', () => {
        if (!details.isConnected) return;
        if (details.open) this.expanded.add(row.span_id); else this.expanded.delete(row.span_id);
      });
      const title = element('summary');
      for (const value of [row.name, row.status === 'skipped' ? '—' : milliseconds(row.inclusive_ms),
        row.status === 'skipped' ? '—' : milliseconds(row.self_ms), String(row.calls),
        Object.entries(row.counters || {}).map(([k,v]) => `${k}: ${v}`).join(', ') || '—', statuses[row.status] || row.status]) {
        title.append(element('span', '', value));
      }
      details.append(title, element('small', '', `${row.routine} · début +${milliseconds(row.start_ms)} · max ${milliseconds(row.max_ms)}`));
      if (Object.keys(row.details || {}).length) details.append(element('pre', '', JSON.stringify(row.details, null, 2)));
      details.append(...branch(row.span_id)); return details;
    });
    const header = element('div', 'diagnostic-columns');
    for (const name of ['Routine','Inclusive','Propre','Appels','Volumes','Statut']) header.append(element('strong', '', name));
    this.tree.replaceChildren(header, ...branch(null));
  }
}

// A generation guard protects both progress and final responses, including failures.
export class SearchGeneration {
  constructor() { this.current = null; }
  begin(requestId) { const token = {requestId}; this.current = token; return token; }
  accepts(token) { return this.current === token; }
}
