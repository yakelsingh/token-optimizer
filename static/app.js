/* TokenWise front-end – plain JavaScript, no build step. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const EXAMPLES = [
  "Hello! I was wondering if you could please kindly explain, in order to understand it really well, how RAG reduces the number of tokens that are sent to the model? Thank you so much!",
  "What is a token?",
  "Compare model routing and caching, and design a strategy to keep an app with 50,000 requests a day cheap. Why does each technique save money?"
];

const state = { fmt: 'text', model: 'auto', docs: [], picked: new Set(), inr: 88, last: null };

/* ---------- helpers ---------- */
const usd = v => '$' + (Math.abs(v) >= 1 ? v.toFixed(2) : Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toFixed(5));
const inr = v => '₹' + (v * state.inr).toFixed(Math.abs(v * state.inr) >= 1 ? 2 : 4);
const num = n => Number(n).toLocaleString('en-IN');
async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
  return j;
}

/* ---------- tabs ---------- */
$('#nav').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#nav button').forEach(x => x.classList.toggle('on', x === b));
  $$('.tab').forEach(t => t.classList.toggle('on', t.id === 'tab-' + b.dataset.tab));
  if (b.dataset.tab === 'stats') loadStats();
  if (b.dataset.tab === 'kb') loadDocs();
});

/* ---------- segmented controls & examples ---------- */
function seg(id, key) {
  $(id).addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    state[key] = b.dataset.v;
    $$('button', $(id)).forEach(x => x.classList.toggle('on', x === b));
  });
}
seg('#fmt', 'fmt'); seg('#mdl', 'model');
$('#examples').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $('#prompt').value = EXAMPLES[+b.dataset.ex]; $('#prompt').focus();
});

/* ---------- documents ---------- */
async function loadDocs() {
  state.docs = await api('/api/documents');
  const ids = new Set(state.docs.map(d => d.id));
  [...state.picked].forEach(i => !ids.has(i) && state.picked.delete(i));
  state.docs.forEach(d => { if (!state.knownDocs?.has(d.id)) state.picked.add(d.id); });
  state.knownDocs = ids;
  renderDocs();
}
function renderDocs() {
  $('#docPick').innerHTML = state.docs.length
    ? state.docs.map(d => `<button type="button" data-id="${d.id}" class="${state.picked.has(d.id) ? 'on' : ''}" aria-pressed="${state.picked.has(d.id)}">${esc(d.name)}</button>`).join('')
    : '<span class="hint">No documents yet – add one in the Knowledge base tab.</span>';
  $('#docList').innerHTML = state.docs.length ? state.docs.map(d => `
    <div class="doc"><div><b>${esc(d.name)}</b><span>${num(d.tokens)} tokens · ${num(d.words)} words · ${d.chunks} chunks</span></div>
    <button class="btn danger" data-del="${d.id}" aria-label="Remove ${esc(d.name)}">Remove</button></div>`).join('')
    : '<p class="hint">The knowledge base is empty. Add a document to try retrieval.</p>';
}
$('#docPick').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  const id = b.dataset.id;
  state.picked.has(id) ? state.picked.delete(id) : state.picked.add(id);
  renderDocs();
});
$('#docList').addEventListener('click', async e => {
  const b = e.target.closest('[data-del]'); if (!b) return;
  await api('/api/documents/' + b.dataset.del, { method: 'DELETE' });
  loadDocs();
});
$('#docFile').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  $('#docText').value = await f.text();
  if (!$('#docName').value) $('#docName').value = f.name;
});
$('#docAdd').addEventListener('click', async () => {
  const msg = $('#docMsg'); msg.style.color = '';
  try {
    const d = await api('/api/documents', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: $('#docName').value || 'Untitled', text: $('#docText').value }) });
    msg.textContent = `Added “${d.name}” – ${d.chunks} chunks, ${num(d.tokens)} tokens.`;
    $('#docName').value = ''; $('#docText').value = '';
    loadDocs();
  } catch (err) { msg.textContent = err.message; msg.style.color = 'var(--red)'; }
});

/* ---------- run ---------- */
async function run() {
  const prompt = $('#prompt').value.trim();
  const err = $('#err'); err.hidden = true;
  if (!prompt) { err.textContent = 'Type a prompt or pick an example first.'; err.hidden = false; return; }
  const btn = $('#run'); btn.disabled = true; btn.firstElementChild.textContent = 'Optimizing…';
  const techniques = {}; $$('#switches input').forEach(i => techniques[i.dataset.t] = i.checked);
  try {
    const res = await api('/api/optimize', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, techniques, format: state.fmt, model: state.model,
        max_words: $('#maxw').value || 'auto', top_k: $('#topk').value || 3, doc_ids: [...state.picked] }) });
    state.inr = res.inr_rate; state.last = res; render(res);
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  finally { btn.disabled = false; btn.firstElementChild.textContent = 'Optimize & run'; }
}
$('#run').addEventListener('click', run);
$('#prompt').addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run(); });

/* ---------- render result ---------- */
function render(r) {
  const u = r.usage, b = u.baseline, o = u.optimized, s = r.savings;
  const row = (label, a, c, sv) => `<tr><td>${label}</td><td>${a}</td><td>${c}</td><td class="sv">${sv}</td></tr>`;
  const dTok = (x, y) => x - y > 0 ? '−' + num(x - y) : '0';
  const stages = r.stages.map(st => `
    <li class="${st.status}"><span class="badge">${st.no}</span>
      <div><b>${esc(st.name)}</b><span class="d">${esc(st.detail)}</span></div>
      <span class="amt">${st.status === 'off' ? 'off' : st.saved_cost > 0 ? '−' + usd(st.saved_cost) : '—'}</span></li>`).join('');
  const diff = r.prompt_diff.map((d, i) => d.op === 'del' ? `<del style="--i:${i}">${esc(d.t)}</del>` : esc(d.t)).join('');
  const chunks = r.retrieved.length ? r.retrieved.map(c => `
    <div class="chunk"><small>${esc(c.doc)} · chunk ${c.idx + 1} · BM25 ${c.score} · ${c.tokens} tokens</small>${esc(c.text)}</div>`).join('')
    : '<p class="hint">No chunks were retrieved for this request.</p>';
  const m = r.model;
  const isJson = r.answer.trim().startsWith('{') || r.answer.trim().startsWith('```');

  $('#results').innerHTML = `
  <div class="res-top">
    <section class="receipt" aria-label="Savings receipt">
      <header><h2>Request receipt</h2><span>${r.latency_ms} ms · ${r.mode} mode</span></header>
      <div class="big"><strong id="bigNum">−0%</strong><span>cost · ${s.tokens_pct}% fewer tokens (${num(s.tokens_saved)} saved)</span></div>
      <table class="rt">
        <tr><th></th><th>Baseline</th><th>TokenWise</th><th>Saved</th></tr>
        ${row('Model', esc(TIER_NAME[b.tier] || b.tier), r.cache.hit ? 'none (cache)' : esc(m.model.replace('claude-', '')), '')}
        ${row('Input tokens', num(b.input), num(o.input), dTok(b.input, o.input))}
        ${row('Output tokens', num(b.output), num(o.output), dTok(b.output, o.output))}
        ${row('Total tokens', num(b.input + b.output), num(o.input + o.output), dTok(b.input + b.output, o.input + o.output))}
        <tr class="tot"><td>Cost</td><td>${usd(b.cost)}</td><td>${usd(o.cost)}</td><td class="sv">−${usd(s.cost_saved)}</td></tr>
        <tr class="tot"><td></td><td>${inr(b.cost)}</td><td>${inr(o.cost)}</td><td class="sv">−${inr(s.cost_saved)}</td></tr>
      </table>
      <div class="proj">At <input id="reqs" type="number" min="1" value="10000" aria-label="Requests per month"> requests a month you would save <b id="projOut"></b></div>
    </section>
  </div>

  <div class="stack">
    <section class="card"><h3>Answer ${r.cache.hit ? `<span class="tag hit">from cache · ${Math.round(r.cache.similarity * 100)}% match</span>` : `<span class="tag model">${esc(m.model.replace('claude-', ''))}</span>`}</h3>
      <div class="answer ${isJson ? 'json' : ''}">${esc(r.answer)}</div></section>

    <section class="card"><h3>1 · Your prompt after optimization</h3>
      <div class="diff">${diff}</div>
      <div class="opt"><small>Sent to the model</small>${esc(r.optimized_prompt)}</div></section>

    <section class="card"><h3>Pipeline</h3><ul class="stages">${stages}</ul></section>

    <section class="card"><h3>Details</h3>
      <details><summary>Retrieved chunks (${r.retrieved.length})</summary>${chunks}</details>
      <details><summary>Why the ${esc(TIER_NAME[m.tier])} model${m.complexity !== null ? ` · complexity ${m.complexity}/100` : ''}</summary>
        ${m.complexity !== null ? `<div class="meter" role="img" aria-label="Complexity ${m.complexity} of 100"><i style="width:${m.complexity}%"></i></div>` : ''}
        <ul class="reasons">${m.reasons.map(x => `<li>${esc(x)}</li>`).join('')}</ul></details>
    </section>
  </div>`;

  const upd = () => { const n = +$('#reqs').value || 0; $('#projOut').textContent = `${usd(s.cost_saved * n)} (${inr(s.cost_saved * n)})`; };
  $('#reqs').addEventListener('input', upd); upd();
  countUp($('#bigNum'), s.cost_pct);
  $('#results').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
}
const TIER_NAME = { small: 'Small', medium: 'Medium', large: 'Large' };
function countUp(el, to) {
  if (reduceMotion || to === 0) { el.textContent = '−' + to + '%'; return; }
  const t0 = performance.now(), dur = 800;
  (function f(t) {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    el.textContent = '−' + (to * e).toFixed(1).replace(/\.0$/, '') + '%';
    if (p < 1) requestAnimationFrame(f); else el.textContent = '−' + to + '%';
  })(t0);
}

/* ---------- analytics ---------- */
async function loadStats() {
  const s = await api('/api/stats'); state.inr = s.inr_rate;
  $('#tiles').innerHTML = `
    <div class="tile"><small>Requests</small><strong>${s.requests}</strong></div>
    <div class="tile g"><small>Tokens saved</small><strong>${s.tokens_pct}%</strong></div>
    <div class="tile g"><small>Cost saved</small><strong>${usd(s.base_cost - s.opt_cost)}</strong></div>
    <div class="tile"><small>Cache hits</small><strong>${s.cache_hits}</strong></div>`;

  // tokens per request (SVG grouped bars)
  const h = s.series;
  if (!h.length) { $('#chartTokens').innerHTML = '<p class="hint">Run a prompt in the Playground to see data here.</p>'; }
  else {
    const W = 520, H = 190, pad = 24, max = Math.max(...h.map(x => x.base_tokens), 1);
    const bw = Math.min(16, (W - pad * 2) / h.length / 2.4);
    const step = (W - pad * 2) / h.length;
    const bars = h.map((x, i) => {
      const bh = x.base_tokens / max * (H - 40), oh = x.opt_tokens / max * (H - 40), cx = pad + i * step + step / 2;
      return `<rect x="${cx - bw - 1}" y="${H - 20 - bh}" width="${bw}" height="${bh}" rx="3" fill="#e5484d" opacity=".85"><title>Baseline ${x.base_tokens} tokens</title></rect>
              <rect x="${cx + 1}" y="${H - 20 - oh}" width="${bw}" height="${Math.max(oh, 1)}" rx="3" fill="#0e9f6e"><title>TokenWise ${x.opt_tokens} tokens</title></rect>`;
    }).join('');
    $('#chartTokens').innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Baseline vs optimized tokens per request">
      <line x1="${pad}" x2="${W - pad}" y1="${H - 20}" y2="${H - 20}" stroke="currentColor" opacity=".2"/>${bars}
      <text x="${pad}" y="12" font-size="11" fill="currentColor" opacity=".6">${num(max)} tokens</text></svg>
      <div class="legend"><span><i style="background:#e5484d"></i>Baseline</span><span><i style="background:#0e9f6e"></i>TokenWise</span></div>`;
  }

  // waterfall
  const names = { prompt: 'Prompt', cache: 'Cache', output: 'Output limit', rag: 'Retrieval', model: 'Model choice' };
  const wmax = Math.max(...Object.values(s.waterfall), 1e-9);
  $('#chartWf').innerHTML = `<div class="bars">${Object.keys(names).map(k => `
    <div class="bar"><span>${names[k]}</span><div class="track"><i style="width:${s.waterfall[k] / wmax * 100}%"></i></div><span>${usd(s.waterfall[k])}</span></div>`).join('')}</div>`;

  // tiers
  const tmax = Math.max(...Object.values(s.tiers), s.cache_hits, 1);
  $('#chartTier').innerHTML = `<div class="bars">${['small', 'medium', 'large'].map(t => `
    <div class="bar"><span>${TIER_NAME[t]}</span><div class="track"><i style="width:${s.tiers[t] / tmax * 100}%"></i></div><span>${s.tiers[t]}</span></div>`).join('')}
    <div class="bar"><span>Cache (no model)</span><div class="track"><i style="width:${s.cache_hits / tmax * 100}%;background:var(--blue)"></i></div><span>${s.cache_hits}</span></div></div>`;

  // history
  $('#histTable').innerHTML = h.length ? `<table><tr><th>Prompt</th><th>Tokens</th><th>Saved</th></tr>${[...h].reverse().map(x =>
    `<tr><td title="${esc(x.prompt)}">${esc(x.prompt)}</td><td>${num(x.opt_tokens)} / ${num(x.base_tokens)}</td><td class="g">${x.base_cost ? Math.round(100 * (x.base_cost - x.opt_cost) / x.base_cost) : 0}%</td></tr>`).join('')}</table>`
    : '<p class="hint">No requests yet.</p>';
}
$('#resetStats').addEventListener('click', async () => { await api('/api/reset', { method: 'POST' }); loadStats(); });

/* ---------- boot ---------- */
(async function () {
  try {
    const h = await api('/api/health');
    const p = $('#modePill'); p.textContent = h.mode + ' mode'; p.classList.toggle('live', h.mode === 'live');
  } catch (e) {}
  loadDocs();
})();
