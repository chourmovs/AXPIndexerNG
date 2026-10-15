#!/usr/bin/env node
// Dependency-free Chromium/CDP measurement against an already running AXP server.
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, readFile, writeFile, rm} from 'node:fs/promises';
import {resolve, join} from 'node:path';
const options = {};
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i];
  if (key === '--no-sandbox') options[key] = true;
  else options[key] = process.argv[++i];
}
if (!options['--url'] || !options['--output']) throw new Error('--url and --output required');
const repeats = Number(options['--repeats'] || 7);
if (!Number.isInteger(repeats) || repeats < 1) throw new Error('repeats must be positive');
const queries = options['--queries-json'] ? JSON.parse(await readFile(options['--queries-json'], 'utf8')) :
  ['DOMINO', 'reactor pressure', 'DOMINO pressure', 'DOMINO storage pressure', 'unobtainium calibration'];
const work = resolve(options['--work-dir'] || 'work'); await mkdir(work, {recursive: true});
const profile = await mkdtemp(join(work, 'chromium-'));
const chrome = spawn(options['--chrome'] || 'chromium', ['--headless=new', '--disable-gpu',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, ...(options['--no-sandbox'] ? ['--no-sandbox'] : []), 'about:blank'],
{stdio: ['ignore', 'ignore', 'pipe']});
let socket;
try {
  const endpoint = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Chromium startup timeout')), 15000);
    let log = '';
    chrome.stderr.on('data', chunk => { log += chunk; const match = log.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); } });
    chrome.on('error', error => {clearTimeout(timer); reject(error);});
    chrome.on('exit', code => {clearTimeout(timer); reject(new Error(`Chromium exited ${code}: ${log.slice(-1000)}`));});
  });
  const origin = new URL(endpoint); const targets = await (await fetch(`http://${origin.host}/json/list`)).json();
  socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
  const pending = new Map(); let sequence = 0;
  socket.onmessage = event => {
    const message = JSON.parse(event.data); const request = pending.get(message.id);
    if (request) { pending.delete(message.id); clearTimeout(request.timer);
      if (message.error) request.reject(new Error(JSON.stringify(message.error))); else request.resolve(message.result); }
  };
  function command(method, params = {}) {
    return new Promise((resolve, reject) => {const id = ++sequence;
      const timer = setTimeout(() => {pending.delete(id); reject(new Error(`CDP timeout: ${method}`));}, 60000);
      pending.set(id, {resolve, reject, timer}); socket.send(JSON.stringify({id, method, params}));});
  }
  async function evaluate(expression) {
    const result = await command('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  await command('Page.navigate', {url: options['--url']});
  await evaluate(`new Promise((resolve, reject) => {
    const start = performance.now(); const timer = setInterval(() => {
      if (document.querySelector('#search-input') && !document.querySelector('#search-input').disabled) {
        clearInterval(timer); resolve(true);
      } else if (performance.now() - start > 30000) {clearInterval(timer); reject(new Error('Search startup timeout'));}
    }, 50);
  })`);
  const rows = [];
  for (let rep = 0; rep <= repeats; rep++) {
    for (const query of queries) {
      const result = await evaluate(`new Promise((resolve, reject) => {
        const timer = setTimeout(() => {window.removeEventListener('axp-search-measured', done); reject(new Error('Search failed or timed out'));}, 55000);
        function done(event) {clearTimeout(timer); window.removeEventListener('axp-search-measured', done);
          resolve({...event.detail, titles: [...document.querySelectorAll('.result-card h3')].map(x => x.textContent),
            diagnostic_text: document.querySelector('.search-diagnostics')?.textContent});}
        window.addEventListener('axp-search-measured', done);
        document.querySelector('#search-input').value = ${JSON.stringify(query)};
        document.querySelector('#search-form').requestSubmit();
      })`);
      rows.push({rep, phase: rep === 0 ? 'first_round' : 'repeat', ...result});
    }
  }
  const quantile = (a, p) => {const x = (a.length - 1) * p, i = Math.floor(x); return a[i] + (a[Math.ceil(x)] - a[i]) * (x - i);};
  const summary = {};
  for (const query of queries) {
    summary[query] = {};
    for (const key of ['browser_first_result_ms', 'browser_response_ms', 'server_http_search_total_ms']) {
      const values = rows.filter(r => r.query === query && r.phase === 'repeat').map(r =>
        key === 'server_http_search_total_ms' ? r.server_timings?.http_search_total_ms : r[key]).filter(x => x != null).sort((a,b) => a-b);
      summary[query][key] = values.length ? {n: values.length, median: quantile(values, 0.5),
        min: values[0], max: values.at(-1), iqr: quantile(values, 0.75) - quantile(values, 0.25)} : {n: 0, median: null};
    }
  }
  await writeFile(options['--output'], JSON.stringify({url: options['--url'],
    environment_label: options['--environment-label'] || 'daemon state unobserved',
    browser: await command('Browser.getVersion'),
    notes: ['first_round contains the first HTTP request only for its first query',
      'first result means two animation frames after final DOM insertion; empty results have no first-result time',
      'headless browser on this host; no claim about the user browser or corpus'], rows, summary}, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally {
  socket?.close(); chrome.kill();
  await new Promise(resolve => {if (chrome.exitCode != null) resolve(); else chrome.once('exit', resolve);});
  await rm(profile, {recursive: true, force: true});
}
