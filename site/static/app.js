/* CS2 Integrity Watch — plain JS, no build step, offline-capable.
   Contract: docs/implementation-plan.md §3.
   Every fetch has a timeout and falls back to the bundled JSON, so the page
   still renders when the API is gone (that is the q35 requirement). */

const $ = (id) => document.getElementById(id);
const state = { matches: [], series: {}, current: null, online: true, summary: {} };

async function getJSON(url, opts = {}, timeout = 3000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const r = await fetch(url, { ...opts, signal: ctl.signal });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
  } finally {
    clearTimeout(timer);
  }
}

/* ---------- data loading, with static fallback ---------- */
async function loadData() {
  try {
    state.matches = await getJSON('/api/matches');
    state.summary = await getJSON('/api/summary');
    setMode(true);
  } catch (e) {
    state.matches = await getJSON('/data/matches.json');
    try { state.summary = await getJSON('/data/summary.json'); } catch (_) {}
    setMode(false);
  }
  renderList();
  $('floor').textContent = (state.summary && state.summary.resolution_floor_s) || 60;
}

async function loadSeries(uid) {
  if (state.series[uid]) return state.series[uid];
  let doc;
  try {
    doc = await getJSON(`/api/series/${encodeURIComponent(uid)}`);
  } catch (e) {
    doc = await getJSON(`/data/series/${encodeURIComponent(uid)}.json`);
  }
  state.series[uid] = doc;
  return doc;
}

function setMode(online) {
  state.online = online;
  const el = $('mode');
  el.className = 'mode' + (online ? '' : ' static');
  el.textContent = online ? 'live API · artifacts + model in memory'
                          : 'static mode · live inference unavailable (bundle only)';
}

/* ---------- list ---------- */
function renderList() {
  const q = ($('search').value || '').toLowerCase().trim();
  const rows = state.matches.filter(m =>
    !q || [m.team1, m.team2, m.event, m.map_name].join(' ').toLowerCase().includes(q));
  const list = $('matches');
  list.innerHTML = '';
  const alerts = state.matches.reduce((a, m) => a + (m.n_alerts || 0), 0);
  $('counts').textContent = `${state.matches.length} maps · ${alerts} alerts · ` +
    `${state.matches.filter(m => m.has_injection).length} with injected feed`;
  rows.forEach((m) => {
    const li = document.createElement('li');
    if (m.map_uid === state.current) li.className = 'sel';
    li.innerHTML = `
      <div>
        <div class="t">${esc(m.team1)} <span class="e">vs</span> ${esc(m.team2)}</div>
        <div class="e">${esc(m.map_name)} · ${esc((m.event || '').slice(0, 34))}</div>
      </div>
      <span class="badge ${m.severity || 'none'}">${
        m.n_alerts ? `${m.n_alerts}·${m.severity}` : '—'}</span>`;
    li.onclick = () => select(m.map_uid);
    list.appendChild(li);
  });
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ---------- detail ---------- */
async function select(uid) {
  state.current = uid;
  renderList();
  const meta = state.matches.find(m => m.map_uid === uid) || {};
  const doc = await loadSeries(uid);

  $('empty').hidden = true;
  $('map').hidden = false;
  $('teams').textContent = `${doc.meta.team1} vs ${doc.meta.team2}`;
  $('meta').textContent =
    `${doc.meta.map_name} · ${doc.meta.event || ''} · match ${doc.meta.match_id} · ` +
    `tickrate ${doc.meta.tickrate} · floor ${doc.meta.resolution_floor_s}s · ` +
    `${doc.rounds.length} rounds`;
  const sev = $('sev');
  sev.textContent = meta.n_alerts ? `${meta.n_alerts} alert(s) · ${meta.severity}`
                                  : 'no divergence above threshold';
  sev.className = 'sev';

  drawChart(doc);
  drawGap(doc);
  drawAlerts(doc);
  drawPlayers(doc);
  prepareLive(doc);
}

const NS = 'http://www.w3.org/2000/svg';
const mk = (tag, attrs = {}) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

function axes(svg, w, h, pad) {
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    const y = pad.t + (1 - p) * (h - pad.t - pad.b);
    svg.appendChild(mk('line', { x1: pad.l, x2: w - pad.r, y1: y, y2: y,
      stroke: '#1b2836', 'stroke-width': 1 }));
    const t = mk('text', { x: pad.l - 8, y: y + 4, fill: '#6f89a6', 'font-size': 10,
      'text-anchor': 'end' });
    t.textContent = p.toFixed(2);
    svg.appendChild(t);
  }
}

function drawChart(doc) {
  const svg = $('chart');
  const w = 820, h = 260, pad = { l: 46, r: 14, t: 14, b: 24 };
  svg.innerHTML = '';
  axes(svg, w, h, pad);
  const rounds = doc.rounds;
  const n = Math.max(rounds.length, 2);
  const x = i => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const y = p => pad.t + (1 - p) * (h - pad.t - pad.b);

  const line = (key, color) => {
    let d = '', pen = false;
    rounds.forEach((r, i) => {
      const v = r[key];
      if (v == null) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    svg.appendChild(mk('path', { d, fill: 'none', stroke: color, 'stroke-width': 2,
      'stroke-linejoin': 'round' }));
  };
  line('model_p', '#5aa9ff');
  line('market_p', '#ffb14e');

  rounds.forEach((r, i) => {
    if (r.market_p == null) return;
    svg.appendChild(mk('circle', {
      cx: x(i), cy: y(r.market_p), r: r.injected ? 4.5 : 2,
      fill: r.injected ? '#ff5d6c' : '#ffb14e',
      stroke: r.injected ? '#ffd0d5' : 'none', 'stroke-width': 1 }));
  });

  rounds.forEach((r, i) => {
    if (i % 4 !== 0 && i !== rounds.length - 1) return;
    const t = mk('text', { x: x(i), y: h - 8, fill: '#6f89a6', 'font-size': 10,
      'text-anchor': 'middle' });
    t.textContent = `R${r.round_num}`;
    svg.appendChild(t);
  });

  doc.alerts.forEach(a => {
    const i = rounds.findIndex(r => r.round_num === a.round_num);
    if (i < 0) return;
    svg.appendChild(mk('line', { x1: x(i), x2: x(i), y1: pad.t, y2: h - pad.b,
      stroke: '#ff5d6c66', 'stroke-dasharray': '3 3' }));
  });
}

function drawGap(doc) {
  const svg = $('gapchart');
  const w = 820, h = 90, pad = { l: 46, r: 14, t: 8, b: 18 };
  svg.innerHTML = '';
  const rounds = doc.rounds, n = Math.max(rounds.length, 2);
  const x = i => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const mid = pad.t + (h - pad.t - pad.b) / 2;
  const scale = (h - pad.t - pad.b) / 2 / 0.6;
  const y = v => mid - v * scale;
  [0.15, -0.15].forEach(v => svg.appendChild(mk('line', {
    x1: pad.l, x2: w - pad.r, y1: y(v), y2: y(v), stroke: '#2c4560',
    'stroke-dasharray': '2 4' })));
  svg.appendChild(mk('line', { x1: pad.l, x2: w - pad.r, y1: mid, y2: mid,
    stroke: '#1b2836' }));
  const bw = Math.max(3, (w - pad.l - pad.r) / n - 2);
  rounds.forEach((r, i) => {
    if (r.gap == null) return;
    const top = Math.min(mid, y(r.gap)), hgt = Math.abs(y(r.gap) - mid);
    svg.appendChild(mk('rect', { x: x(i) - bw / 2, y: top, width: bw, height: hgt,
      fill: r.injected ? '#ff5d6c' : (r.gap > 0 ? '#ffb14e' : '#5aa9ff'), opacity: .85 }));
  });
  const lbl = mk('text', { x: pad.l - 8, y: mid + 4, fill: '#6f89a6', 'font-size': 10,
    'text-anchor': 'end' });
  lbl.textContent = '0';
  svg.appendChild(lbl);
}

function drawAlerts(doc) {
  const box = $('alerts');
  box.innerHTML = '';
  $('alert-count').textContent = `${doc.alerts.length} alert(s)`;
  if (!doc.alerts.length) {
    box.innerHTML = '<div class="hint">No alert: no sustained divergence above the ' +
      '15-point threshold with the market moving first.</div>';
    return;
  }
  doc.alerts.forEach(a => {
    const c = a.criteria || {};
    const div = document.createElement('div');
    div.className = `alert ${a.severity}`;
    const contribs = (a.contributions || []).map(c2 =>
      `<li>${esc(c2.name)} = ${c2.value} → ${c2.contribution >= 0 ? '+' : ''}${c2.contribution.toFixed(4)}</li>`
    ).join('');
    div.innerHTML = `
      <h4>Round ${a.round_num}${a.span[1] > a.span[0] ? `–${a.span[1]}` : ''} ·
          ${a.severity} severity</h4>
      <div class="row">
        <span>gap ${fmtPct(a.gap)}</span>
        <span>residual ${fmtPct(a.residual_gap)}</span>
        <span>baseline ${fmtPct(a.baseline_gap)}</span>
        <span>floor ${a.floor_s}s</span>
      </div>
      <div class="row" style="margin-top:6px">
        <span class="crit"><span class="${c.residual_ge_15 ? 'ok' : 'no'}">●</span> residual ≥ 15 pts</span>
        <span class="crit"><span class="${c.persistence_rounds >= 2 ? 'ok' : 'no'}">●</span> persisted ${c.persistence_rounds} rounds</span>
        <span class="crit"><span class="${c.market_moved_first ? 'ok' : 'no'}">●</span> market moved first</span>
        <span class="crit"><span class="${c.no_explaining_event ? 'ok' : 'no'}">●</span> no explaining game event</span>
      </div>
      <ul>${contribs || '<li>no attribution available</li>'}</ul>
      <div class="disc">${esc(a.disclaimer || '')}</div>`;
    box.appendChild(div);
  });
}

function fmtPct(v) {
  if (v == null) return '—';
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)} pts`;
}

function drawPlayers(doc) {
  const t = $('players');
  const rows = doc.players || [];
  t.innerHTML = '<tr><th>rank</th><th>player</th><th>rounds</th><th>z</th></tr>' +
    rows.slice(0, 10).map(p => `
      <tr><td>${p.rank}</td><td>${esc(p.name)}</td><td>${p.rounds_flagged}</td>
      <td class="${p.z_mean < 0 ? 'neg' : 'pos'}">${p.z_mean.toFixed(2)}</td></tr>`).join('');
  if (!rows.length) t.innerHTML += '<tr><td colspan="4" class="hint">no flagged rounds</td></tr>';
}

function prepareLive(doc) {
  const a = doc.alerts[0];
  const r = a ? doc.rounds.find(x => x.round_num === a.round_num) : doc.rounds[0];
  if (!r) return;
  $('lr-round').value = r.round_num;
  $('lr-s1').value = r.score_team1; $('lr-s2').value = r.score_team2;
  $('lr-e1').value = Math.round((r.equip_diff_team1_minus_team2 >= 0 ? 11000 : 9000));
  $('lr-e2').value = 11000;
  $('lr-out').textContent = state.online
    ? 'press “Score this state” to run inference on the host'
    : 'static mode: live inference needs the API';
}

$('lr-go').onclick = async () => {
  if (!state.online) { $('lr-out').textContent = 'static mode: API unreachable'; return; }
  const doc = state.series[state.current];
  const body = { round_state: {
    score_team1: +$('lr-s1').value, score_team2: +$('lr-s2').value,
    round_num: +$('lr-round').value,
    team1_side: (doc && doc.rounds.find(r => r.round_num === +$('lr-round').value) || {}).team1_side || 'ct',
    equip_team1: +$('lr-e1').value, equip_team2: +$('lr-e2').value,
    buy_team1: $('lr-b1').value, buy_team2: $('lr-b2').value,
    map_name: doc ? doc.meta.map_name : 'de_overpass',
  }};
  $('lr-out').textContent = 'scoring…';
  try {
    const out = await getJSON('/api/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }, 5000);
    $('lr-out').textContent = JSON.stringify({
      p_logit: out.p_logit, p_xgb: out.p_xgb,
      latency_ms: out.latency_ms, arch: out.arch, host: out.hostname,
      python: out.python,
      top_contributions: (out.contributions || []).slice(0, 3),
    }, null, 2);
  } catch (e) {
    $('lr-out').textContent = `scoring failed: ${e.message}`;
  }
};

$('search').oninput = renderList;

(async function boot() {
  await loadData();
  try {
    const h = await getJSON('/api/health');
    const el = $('health');
    el.className = 'chip ' + (h.ok ? 'ok' : 'warn');
    el.textContent = `${h.arch} · py${h.python} · ${h.n_maps} maps · ${h.uptime_s}s`;
  } catch (_) {
    const el = $('health');
    el.className = 'chip warn';
    el.textContent = 'static mode';
  }
  if (state.matches.length) select(state.matches[0].map_uid);
})();
