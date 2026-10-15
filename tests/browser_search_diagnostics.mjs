// Optional real-browser integration: node tests/browser_search_diagnostics.mjs
// Requires Playwright + Chromium; Python/HTTP coverage lives in test_search_diagnostics.py.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
const {chromium} = createRequire(import.meta.url)('playwright');
const root = new URL('../client/axp_client/web/', import.meta.url);
const server = createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  const payloads = {
    '/api/version':{version:'v0.3.7-alpha4', commit:'61480d6'},
    '/api/startup':{search:{state:'ready'}, local_ai:{state:'unconfigured'}},
    '/api/ask/health':{available:false}, '/api/skills':{skills:[]}, '/api/models':{models:[]},
    '/api/search/cancel':{status:'cancellation_requested'},
  };
  if (url.pathname in payloads) {
    response.setHeader('Content-Type','application/json'); response.end(JSON.stringify(payloads[url.pathname])); return;
  }
  try {
    const name = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
    if (!/^[a-z-]+\.(js|css|html)$/.test(name)) {response.writeHead(404).end(); return;}
    response.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html');
    response.end(await readFile(new URL(name, root)));
  } catch {response.writeHead(404).end();}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || undefined,
  args:['--no-sandbox']});
try {
  const page = await browser.newPage(); const errors = []; page.on('pageerror', error => errors.push(error));
  const pending = new Map(); let progressFails = false;
  const trace = (id, status='completed') => ({request_id:id, status, elapsed_ms:100,
    active_stage:status === 'running' ? 'Vecteurs en cours' : null, active_stage_elapsed_ms:30,
    build:{version:'v0.3.7-alpha4',commit:'61480d6'}, server_scope:'Test fixture scope',
    spans:[{span_id:'1', parent_span_id:null, name:'Recherche', routine:'test.search', start_ms:0, inclusive_ms:100,
      self_ms:30, max_ms:100, calls:1, status, counters:{candidates:2}, details:{cache:'miss'}}]});
  const respond = async (query, title=query) => {
    const item = pending.get(query); assert.ok(item, `pending ${query}`);
    await item.route.fulfill({json:{results:[{document_id:1, title, path:'/fixture', snippet:'Evidence', passage_score:.8}],
      request_trace:trace(item.id)}}).catch(() => {});
  };
  await page.route('**/api/search?*', route => {
    const params = new URL(route.request().url()).searchParams;
    pending.set(params.get('q'), {route, id:params.get('request_id')});
  });
  await page.route('**/api/search/progress?*', route => {
    if (progressFails) return route.abort();
    const id = new URL(route.request().url()).searchParams.get('request_id');
    return route.fulfill({json:trace(id, 'running')});
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const submit = async query => {
    await page.locator('#search-input').fill(query);
    await page.locator('#search-form button').first().click();
    await page.waitForFunction(() => document.querySelector('#cancel-search').hidden === false);
    await new Promise(resolve => setTimeout(resolve, 30));
  };
  await submit('old');
  await page.waitForFunction(() => document.querySelector('#search-diagnostic-summary').textContent.includes('Vecteurs en cours'));
  assert.match(await page.locator('#results').textContent(), /Searching/);
  await submit('new'); await respond('new', 'NEW RESULT');
  await page.waitForFunction(() => document.querySelector('#results').textContent.includes('NEW RESULT'));
  const newId = pending.get('new').id;
  await page.waitForFunction(id => document.querySelector('#search-diagnostic-summary').textContent.includes(id), newId);
  await page.waitForFunction(() => !document.querySelector('#search-diagnostic-summary').textContent.includes('Premier résultat rendu : Non mesuré'));
  await page.evaluate(() => Object.defineProperty(navigator, 'clipboard', {configurable:true,
    value:{writeText:async () => {throw new Error('clipboard disabled');}}}));
  await page.locator('#copy-search-diagnostic').click();
  const copied = JSON.parse(await page.locator('#search-diagnostic-json').inputValue());
  assert.equal(copied.server.request_id, newId);
  assert.ok(copied.browser.first_result_ms >= copied.browser.render_ms);
  await page.locator('#search-diagnostic-tree details').first().locator('summary').first().click();
  assert.equal(await page.locator('#search-diagnostic-tree details').first().evaluate(node => node.open), true);
  await respond('old', 'OBSOLETE RESULT');
  await page.waitForTimeout(300);
  assert.match(await page.locator('#results').textContent(), /NEW RESULT/);
  assert.doesNotMatch(await page.locator('#results').textContent(), /OBSOLETE/);
  assert.match(await page.locator('#search-diagnostic-summary').textContent(), new RegExp(newId));
  progressFails = true;
  await submit('channel failure');
  await page.waitForFunction(() => document.querySelector('#search-diagnostic-summary').textContent.includes('Canal de progression indisponible'));
  await respond('channel failure', 'RESULT DESPITE CHANNEL FAILURE');
  await page.waitForFunction(() => document.querySelector('#results').textContent.includes('RESULT DESPITE CHANNEL FAILURE'));
  await submit('cancelled'); await page.locator('#cancel-search').click(); await respond('cancelled', 'CANCELLED RESPONSE');
  await page.waitForTimeout(100);
  assert.equal(await page.locator('#results').textContent(), 'Recherche annulée.');
  await page.locator('#search-diagnostics > summary').click();
  await page.reload();
  assert.equal(await page.locator('#search-diagnostics').evaluate(node => node.open), false);
  assert.deepEqual(errors, []);
  console.log('Browser checks passed: live progress, successive requests, old responses, JSON copy fallback, routine tree, channel failure, cancellation, persisted panel.');
} finally {
  await browser.close(); await new Promise(resolve => server.close(resolve));
}
