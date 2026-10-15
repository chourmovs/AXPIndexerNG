import {test} from 'node:test';
import assert from 'node:assert/strict';
import {initSearch} from '../client/axp_client/web/search.js';

class Node {
  constructor(tag) {this.tag = tag; this.children = []; this.listeners = {}; this.value = ''; this.text = '';}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this.children = nodes; this.text = '';}
  set textContent(value) {this.text = value; this.children = [];}
  get textContent() {return this.text + this.children.map(x => x.textContent ?? x).join('');}
  addEventListener(event, handler) {this.listeners[event] = handler;}
  setAttribute() {}
}
function setup() {
  const nodes = Object.fromEntries(['search-form', 'results', 'search-input', 'cancel-search', 'search-diagnostics', 'search-diagnostic-summary', 'search-diagnostic-tree', 'copy-search-diagnostic', 'search-diagnostic-json'].map(id => [id, new Node(id)]));
  const frames = [], requests = [], events = [];
  let clock = 0;
  globalThis.document = {querySelector: selector => nodes[selector.slice(1)],
    createElement: tag => new Node(tag), createTextNode: text => ({textContent: text})};
  globalThis.window = {dispatchEvent: event => events.push(event)};
  globalThis.CustomEvent = class {constructor(type, {detail}) {this.type = type; this.detail = detail;}};
  globalThis.performance = {now: () => clock};
  globalThis.requestAnimationFrame = callback => frames.push(callback);
  globalThis.fetch = (url, options) => url.includes('/api/search/cancel') ? Promise.resolve({ok:true, json:async()=>({})}) : new Promise((resolve, reject) => requests.push({url, options, resolve, reject}));
  initSearch();
  return {nodes, frames, requests, events, time: value => {clock = value;},
    submit: query => {nodes['search-input'].value = query; return nodes['search-form'].listeners.submit({preventDefault() {}});},
    frame: () => {const pending = frames.splice(0); pending.forEach(callback => callback());}};
}
const row = {document_id: 1, title: 'Reactor', snippet: 'pressure', path: '/file', passage_score: 0.8};
function response(request, rows = [row]) {request.resolve({ok: true, json: async () => ({results: rows,
  timings: {http_search_total_ms: 40}, request_trace:{elapsed_ms:40, spans:[]}, diagnostics: {catalog_snapshot_status: 'unavailable',
    catalog_snapshot_refreshing: true, identity_cache: {status: 'miss', age_ms:0}}})});}

test('search renders with unavailable counts; browser clock stays separate from server', async () => {
  const env = setup(); const submitted = env.submit('pressure'); env.time(70); response(env.requests[0]); await submitted;
  assert.match(env.nodes.results.textContent, /Reactor/);
  assert.match(env.nodes.results.textContent, /Compteurs indisponibles/);
  assert.match(env.nodes.results.textContent, /Cache identité : miss/);
  assert.equal(env.events.length, 0); env.time(80); env.frame(); assert.equal(env.events.length, 0);
  env.time(96); env.frame(); const measured = env.events[0].detail;
  assert.equal(measured.browser_first_result_ms, 96);
  assert.equal(measured.server_timings.http_search_total_ms, 40);
  assert.equal(measured.browser_response_ms, 70);
});

test('a late response or error cannot replace a newer query', async () => {
  const env = setup(); const old = env.submit('old'); const current = env.submit('current');
  assert.equal(env.requests[0].options.signal.aborted, true);
  response(env.requests[1]); await current; const rendered = env.nodes.results.textContent;
  // Simulate a transport that ignores abort and replies late.
  response(env.requests[0], [{...row, title: 'Obsolete'}]); await old;
  assert.equal(env.nodes.results.textContent, rendered);
  env.frame(); env.frame(); assert.equal(env.events.length, 1); assert.equal(env.events[0].detail.query, 'current');
  const failure = env.submit('failure'); const final = env.submit('final');
  env.requests[2].reject(new Error('late network error')); await failure;
  response(env.requests[3]); await final; assert.match(env.nodes.results.textContent, /Reactor/);
});

test('old animation frames are ignored, and empty results have no first-result duration', async () => {
  const env = setup(); const first = env.submit('first'); response(env.requests[0]); await first;
  const empty = env.submit('empty'); response(env.requests[1], []); await empty;
  env.frame(); env.frame(); assert.equal(env.events.length, 1);
  assert.equal(env.events[0].detail.query, 'empty'); assert.equal(env.events[0].detail.browser_first_result_ms, null);
});
