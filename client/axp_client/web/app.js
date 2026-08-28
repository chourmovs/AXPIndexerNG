const form = document.querySelector('form');
const results = document.querySelector('#results');
const stopwords = new Set(['and', 'are', 'the', 'for', 'with', 'les', 'des', 'une', 'dans', 'pour', 'avec']);

function queryTerms(query) {
  const phrases = [...query.matchAll(/"([^"]+)"/gu)].map(match => match[1].trim()).filter(Boolean);
  const words = query.match(/[\p{L}\p{N}_-]+/gu) || [];
  return [...new Set([...phrases, ...words.filter(word => word.length >= 3 && !stopwords.has(word.toLocaleLowerCase()))])]
    .sort((left, right) => right.length - left.length);
}

function appendHighlighted(element, value, terms) {
  const text = String(value || '');
  if (!terms.length) { element.append(document.createTextNode(text)); return; }
  const escaped = terms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const matcher = new RegExp(`(${escaped.join('|')})`, 'giu');
  let offset = 0;
  for (const match of text.matchAll(matcher)) {
    element.append(document.createTextNode(text.slice(offset, match.index)));
    const mark = document.createElement('mark');
    mark.textContent = match[0];
    element.append(mark);
    offset = match.index + match[0].length;
  }
  element.append(document.createTextNode(text.slice(offset)));
}

function percent(value) {
  const number = Number(value);
  return Math.round(Math.max(0, Math.min(1, Number.isFinite(number) ? number : 0)) * 100);
}

function renderResult(row, terms) {
  const article = document.createElement('article');
  const header = document.createElement('div'); header.className = 'result-header';
  const title = document.createElement('h2'); appendHighlighted(title, row.title || row.filename || row.path, terms);
  const score = document.createElement('span'); score.className = 'relevance'; score.textContent = `Relevance ${percent(row.relevance_score)}%`;
  const details = [];
  if (row.vector_similarity != null) details.push(`Vector similarity: ${percent(row.vector_similarity)}%`);
  if (row.lexical_coverage != null) details.push(`Lexical coverage: ${percent(row.lexical_coverage)}%`);
  score.title = details.join('\n'); header.append(title, score);
  const path = document.createElement('div'); path.className = 'path'; path.textContent = row.path;
  const source = document.createElement('small'); source.textContent = `Source: ${row.source_label || row.source_path || '—'}`;
  const snippet = document.createElement('p'); appendHighlighted(snippet, row.snippet, terms);
  const action = document.createElement('button'); action.type = 'button'; action.className = 'open-file'; action.textContent = 'Open file';
  const error = document.createElement('span'); error.className = 'action-error';
  action.addEventListener('click', async () => {
    error.textContent = ''; action.disabled = true;
    try {
      const response = await fetch(`/api/document/${encodeURIComponent(row.document_id)}/open`, {method: 'POST'});
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || 'The file could not be opened.');
    } catch (exception) { error.textContent = exception.message; }
    finally { action.disabled = false; }
  });
  article.append(header, path, source, snippet, action, error);
  return article;
}

form.addEventListener('submit', async event => {
  event.preventDefault(); results.textContent = 'Searching…';
  const query = form.querySelector('input').value;
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const rows = await response.json();
    if (!rows.length) { results.textContent = 'No sufficiently relevant result.'; return; }
    const terms = queryTerms(query);
    results.replaceChildren(...rows.map(row => renderResult(row, terms)));
  } catch (_) { results.textContent = 'Search is temporarily unavailable.'; }
});
