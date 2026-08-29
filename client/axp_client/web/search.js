import {searchDocuments} from './api.js';
import {createDocumentActions} from './documents.js';
import {appendHighlighted, element} from './ui.js';
const stopwords = new Set(['and', 'are', 'the', 'for', 'with', 'les', 'des', 'une', 'dans', 'pour', 'avec']);
function queryTerms(query) { const phrases = [...query.matchAll(/"([^"]+)"/gu)].map(x => x[1].trim()).filter(Boolean);
  const words = query.match(/[\p{L}\p{N}_-]+/gu) || []; return [...new Set([...phrases, ...words.filter(x => x.length >= 3 && !stopwords.has(x.toLocaleLowerCase()))])].sort((a, b) => b.length - a.length); }
const percent = value => Math.round(Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0)) * 100);
function renderResult(row, terms) { const article = element('article', 'result-card'); const header = element('div', 'result-header');
  const title = element('h3'); appendHighlighted(title, row.title || row.filename || row.path, terms);
  const score = element('span', 'relevance', `Relevance ${percent(row.relevance_score)}%`); const details = [];
  if (row.vector_similarity != null) details.push(`Vector similarity: ${percent(row.vector_similarity)}%`);
  if (row.lexical_coverage != null) details.push(`Lexical coverage: ${percent(row.lexical_coverage)}%`); score.title = details.join('\n'); header.append(title, score);
  const path = element('div', 'path', row.path); const source = element('small', '', `Source: ${row.source_label || row.source_path || '—'}`);
  const snippet = element('p'); appendHighlighted(snippet, row.snippet, terms); article.append(header, path, source, snippet, createDocumentActions(row.document_id)); return article; }
export function initSearch() { const form = document.querySelector('#search-form'); const results = document.querySelector('#results');
  form.addEventListener('submit', async event => { event.preventDefault(); const query = document.querySelector('#search-input').value.trim(); if (!query) return;
    results.textContent = 'Searching…'; try { const rows = await searchDocuments(query); if (!rows.length) { results.textContent = 'No sufficiently relevant result.'; return; }
      const terms = queryTerms(query); results.replaceChildren(...rows.map(row => renderResult(row, terms)));
    } catch (_) { results.textContent = 'Search is temporarily unavailable.'; } }); }
