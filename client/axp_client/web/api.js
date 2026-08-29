export class ApiError extends Error {
  constructor(code, message, status = 0) { super(message || code); this.code = code; this.status = status; }
}

async function jsonRequest(url, options = {}) {
  let response;
  try { response = await fetch(url, options); } catch (_) { throw new ApiError('network_error', 'AXP is temporarily unavailable.'); }
  let body = {};
  try { body = await response.json(); } catch (_) { if (!response.ok) throw new ApiError('http_error', 'AXP request failed.', response.status); }
  if (!response.ok) throw new ApiError(body.error || 'http_error', body.error || 'AXP request failed.', response.status);
  return body;
}
export const searchDocuments = query => jsonRequest(`/api/search?q=${encodeURIComponent(query)}`);
export const askHealth = () => jsonRequest('/api/ask/health', {cache: 'no-store'});
export const openDocument = id => jsonRequest(`/api/document/${encodeURIComponent(id)}/open`, {method: 'POST'});
export const openDocumentDirectory = id => jsonRequest(`/api/document/${encodeURIComponent(id)}/open-dir`, {method: 'POST'});

export async function askStream(question, onEvent) {
  let response;
  try { response = await fetch('/api/ask/stream', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question})}); } catch (_) { throw new ApiError('network_error', 'AXP is temporarily unavailable.'); }
  if (!response.ok) {
    let body = {}; try { body = await response.json(); } catch (_) { /* normalized below */ }
    throw new ApiError(body.error || 'http_error', body.error || 'AXP request failed.', response.status);
  }
  if (!response.body) throw new ApiError('stream_unavailable', 'The answer stream is unavailable.');
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = '';
  while (true) {
    const {value, done} = await reader.read(); pending += decoder.decode(value || new Uint8Array(), {stream: !done});
    const lines = pending.split('\n'); pending = lines.pop() || '';
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) break;
  }
  if (pending.trim()) onEvent(JSON.parse(pending));
}
