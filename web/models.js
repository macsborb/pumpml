// Vue "Modeles" : performance reelle (paper trading au prix Jupiter) du modele en service et de chaque modele.
// S'appuie sur index.html ($, esc, sol, cls, api, openToken) et ai.js (C, chart, timeAxis, hhmm, idToMs, pct, fmt, showModel).

(function () {
  const st = document.createElement('style');
  st.textContent = `
  .mh{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
  .mh h1{margin:0;font-size:20px}
  .mh .sub{color:var(--muted);font-size:13px;margin-top:4px}
  .chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
  .chip{font-size:11px;padding:3px 9px;border-radius:999px;background:var(--panel2);border:1px solid var(--line);color:var(--muted)}
  .chip.gold{color:var(--amber);border-color:rgba(245,185,66,.5)}
  .chip.live{color:var(--violet);border-color:rgba(165,123,255,.5)}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:14px}
  .tile{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
  .tile .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
  .tile .v{font-size:22px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
  .tile .s{font-size:12px;color:var(--muted);margin-top:2px}
  .mx td,.mx th{padding:7px 10px}
  .mx th{font-size:12px;color:var(--text);font-weight:600;background:var(--panel2)}
  .mx td:first-child{color:var(--muted)}
  .mx tr.grp td{background:#151925;color:var(--violet);font-size:11px;text-transform:uppercase;letter-spacing:.5px;text-align:left}
  .mx tbody tr{cursor:default}
  .lb th{cursor:pointer;white-space:nowrap}
  .lb th.sorted{color:var(--text)}
  .lb td{white-space:nowrap}
  .muted{color:var(--muted)}
  .warn{background:rgba(245,185,66,.08);border:1px solid rgba(245,185,66,.35);border-radius:10px;padding:8px 12px;font-size:13px;margin-top:10px}
  `;
  document.head.appendChild(st);
})();

const FAM = { xgb: 'XGBoost', lgb: 'LightGBM' };
const OBJ = { pnl_weighted: 'pondere par le gain', binary: 'gagnant / perdant', regression: 'regression du gain' };
const mName = id => id ? 'Modele du ' + hhmm(idToMs(id)) : '-';
const dur = h => h == null ? '-' : h < 1 ? Math.round(h * 60) + ' min' : h < 48 ? h.toFixed(1).replace('.', ',') + ' h' : (h / 24).toFixed(1).replace('.', ',') + ' j';
const pc = (v, d = 0) => v == null ? '-' : (v > 0 ? '+' : '') + (100 * v).toFixed(d) + ' %';
const MODEL_COLORS = [C.violet, C.blue, C.green, C.amber, '#ff8ad8', '#4fd1c5', '#c0c66b', '#ff9f5b'];
let mdSort = { key: 'start_ms', dir: -1 }, mdData = null, mdSig = null;

function verdictLive(w, stake) {
  const n = w.closed || 0;
  if (!n) return { cls: 'wait', t: 'Pas encore d achat depuis sa mise en service', s: 'Les chiffres apparaitront au premier achat.' };
  const edge = (w.sim_buy_mean != null && w.sim_all_mean != null) ? w.sim_buy_mean - w.sim_all_mean : null;
  const res = `${sol(w.pnl, 3)} SOL en ${n} achats (${pc(w.roi, 1)} de la mise, ${pct(w.win_rate)} de trades gagnants).`;
  const tri = edge == null ? '' : edge > 0
    ? ` Son tri est meilleur que le hasard : ses achats font en moyenne ${sol(edge, 4)} SOL de mieux par coin qu acheter tout (simulation).`
    : ` Son tri ne fait pas mieux que le hasard : ses achats font ${Math.abs(edge).toFixed(4)} SOL par coin de moins qu acheter tout (simulation).`;
  const small = n < 30 ? ` Moins de 30 achats : resultat encore fragile.` : '';
  return { cls: w.pnl > 0 ? 'ok' : 'ko', t: (w.pnl > 0 ? 'Gagnant' : 'Perdant') + ' depuis sa mise en service', s: res + tri + small };
}

function row(label, wins, f, tip = '') {
  return `<tr><td${tip ? ` title="${esc(tip)}"` : ''}>${label}</td>${wins.map(w => `<td>${f(w)}</td>`).join('')}</tr>`;
}

function matrix(ws, stake) {
  const g = t => `<tr class="grp"><td colspan="${ws.length + 1}">${t}</td></tr>`;
  const v = (x, d = 3) => `<span class="${cls(x)}">${sol(x, d)}</span>`;
  return `<table class="mx"><thead><tr><th>Periode</th>${ws.map(w => `<th>${esc(w.label)}</th>`).join('')}</tr></thead><tbody>
    ${g('Resultat reel (paper, prix Jupiter)')}
    ${row('Gain realise (SOL)', ws, w => `<b>${v(w.pnl)}</b>`, 'Somme des gains et pertes des positions fermees, frais compris')}
    ${row('Rendement sur la mise', ws, w => `<span class="${cls(w.roi)}">${pc(w.roi, 1)}</span>`, `Gain / total mise (${stake} SOL par achat)`)}
    ${row('Achats executes', ws, w => w.trades ? `${w.trades}${w.open ? ` <span class="muted">(${w.open} ouv.)</span>` : ''}` : '<span class="muted">0</span>', 'Achats passes au prix Jupiter (fermes + ouverts)')}
    ${row('Achats non executes', ws, w => w.not_executed ? `<span class="muted">${w.not_executed}</span>` : '<span class="muted">0</span>', 'Pas de cotation Jupiter, ou limite de positions atteinte')}
    ${row('Positions ouvertes (latent)', ws, w => w.open ? v(w.unrealized) : '<span class="muted">-</span>', 'Gain non realise des positions encore ouvertes')}
    ${g('Qualite des trades')}
    ${row('Trades gagnants', ws, w => pct(w.win_rate), 'Part des positions fermees avec un gain')}
    ${row('Gain moyen par achat', ws, w => v(w.per_trade, 4))}
    ${row('Gain moyen d un gagnant', ws, w => v(w.avg_win, 4))}
    ${row('Perte moyenne d un perdant', ws, w => v(w.avg_loss, 4))}
    ${row('Facteur de profit', ws, w => w.profit_factor == null ? '-' : `<span class="${w.profit_factor >= 1 ? 'pos' : 'neg'}">${w.profit_factor.toFixed(2).replace('.', ',')}</span>`, 'Gains totaux / pertes totales. Au-dessus de 1 = gagnant.')}
    ${row('Meilleur / pire achat', ws, w => w.best == null ? '-' : `${v(w.best)} / ${v(w.worst)}`)}
    ${row('Pire baisse (drawdown)', ws, w => w.closed ? v(w.max_drawdown) : '-', 'Plus grosse baisse du gain cumule depuis un sommet')}
    ${g('Le tri du modele vs acheter tout (meme simulation)')}
    ${row('Coins notes', ws, w => w.coins ?? '-', 'Coins analyses par le modele sur la periode (resultat connu entre parentheses)')}
    ${row('Gain simule moyen : ses achats', ws, w => v(w.sim_buy_mean, 4), 'Simulation par la courbe, la meme pour tous les coins')}
    ${row('Gain simule moyen : tous les coins', ws, w => v(w.sim_all_mean, 4), 'Si on avait achete chaque coin note (reference)')}
    ${row('Avantage du tri', ws, w => (w.sim_buy_mean == null || w.sim_all_mean == null) ? '-' : v(w.sim_buy_mean - w.sim_all_mean, 4), 'Positif = le modele choisit mieux que le hasard')}
  </tbody></table>`;
}

async function loadModels() {
  const el = $('#v-models');
  if (!el.innerHTML) el.innerHTML = `
    <div class="panel" id="mdHead"></div>
    <div class="panel" style="margin-top:16px"><div class="ph"><h2>Performance par periode</h2></div>
      <p class="what">Le modele en service, sur les dernieres heures. Les achats sont rattaches au modele qui les a decides ; chaque colonne compte les achats faits pendant la periode.</p>
      <div id="mdMatrix" style="overflow-x:auto"></div></div>
    <div class="two" style="margin-top:16px">
      ${panel('chEquity', 'Gain cumule de tous les achats', 'Chaque point est une position fermee ; sa couleur indique le modele qui a achete.', 'Une courbe qui monte = de l argent gagne. Les changements de couleur montrent quand un nouveau modele a pris le relais.', 300)}
      ${panel('chHourly', 'Gain par heure (modele en service, 48 h)', 'Gain realise des positions fermees, heure par heure.', 'Vert = heure gagnante, rouge = perdante. Survole une barre pour le nombre d achats.', 300)}
    </div>
    <div class="two" style="margin-top:16px">
      ${panel('chExits', 'Pourquoi il vend (depuis sa mise en service)', 'Nombre de positions par raison de sortie, et ce que chacune a rapporte.', 'Beaucoup de ventes a 2 min = des achats ou rien ne se passe. Les stops montrent les mauvais choix.', 260)}
      <div class="panel"><h2>Derniers achats du modele en service</h2><p class="what">Clique sur une ligne pour ouvrir la fiche du coin.</p><div id="mdRecent" style="overflow-x:auto;max-height:300px"></div></div>
    </div>
    <div class="panel" style="margin-top:16px"><h2>Classement de tous les modeles</h2>
      <p class="what"><b>En direct</b> = ce qu il a fait sur de vrais coins quand il etait en service (prix Jupiter). <b>Test</b> = ce qu il aurait fait sur les heures jamais vues a l entrainement. Clique un en-tete pour trier, une ligne pour le detail.</p>
      <div id="mdBoard" style="overflow-x:auto"></div></div>
    <div id="mdDetail" style="margin-top:16px"></div>`;
  const d = await api('/api/perf');
  mdData = d;
  if (!models) try { models = await api('/api/models'); } catch (e) {}
  const ws = d.windows, since = ws[ws.length - 1], meta = d.focus_meta || {}, w24 = ws.find(w => w.hours === 24) || {};
  const v = verdictLive(since, d.stake);
  const fam = FAM[meta.family] || meta.family || '?';
  $('#mdHead').innerHTML = `<div class="mh"><div>
      <div class="muted" style="font-size:12px;text-transform:uppercase;letter-spacing:.5px">Modele en service</div>
      <h1>${mName(d.focus)} ${d.focus === d.champion ? '👑' : ''}</h1>
      <div class="sub">En service depuis ${hhmm(meta.start_ms)} (${dur(since.hours)}) · achete si son score depasse ${fmt(meta.threshold, 3)}</div>
      <div class="chips"><span class="chip ${d.focus === d.champion ? 'gold' : ''}">${d.focus === d.champion ? 'Champion' : 'Non promu (faute de mieux)'}</span>
        <span class="chip">${fam}${meta.ensemble_size > 1 ? ' · ensemble de ' + meta.ensemble_size : ''}</span><span class="chip">objectif ${OBJ[meta.objective] || meta.objective || '?'}</span>
        <span class="chip">appris sur ${meta.train_tokens ?? '?'} coins</span><span class="chip">${meta.deep ? 'recherche profonde' : 'entrainement horaire'}</span></div>
    </div>
    <div class="verdict ${v.cls}" style="max-width:560px"><div class="vt">${v.t}</div><div>${v.s}</div></div></div>
    <div class="tiles">
      ${[['Derniere heure', ws[0]], ['4 dernieres heures', ws[1]], ['24 dernieres heures', w24], ['Depuis sa mise en service', since]].map(([l, w]) =>
        `<div class="tile"><div class="l">${l}</div><div class="v ${cls(w.pnl)}">${w.closed ? sol(w.pnl, 3) + ' SOL' : '-'}</div>
        <div class="s">${w.trades || 0} achats · ${w.closed ? pct(w.win_rate) + ' gagnants' : 'aucun ferme'}</div></div>`).join('')}
      <div class="tile"><div class="l">Test avant mise en service</div><div class="v ${cls(meta.valid_pnl)}">${sol(meta.valid_pnl, 3)} SOL</div>
        <div class="s">${meta.valid_trades ?? '-'} achats sur ${meta.valid_tokens ?? '-'} coins · tout acheter ${sol(meta.valid_buy_all_pnl, 2)}</div></div>
    </div>`;
  $('#mdMatrix').innerHTML = matrix(ws, d.stake);

  // derniers achats
  $('#mdRecent').innerHTML = d.recent.length ? `<table><thead><tr><th>Coin</th><th>Achat</th><th title="Score du modele / seuil">Score</th><th>Statut</th><th>Gain</th></tr></thead><tbody>${
    d.recent.map(p => `<tr onclick="openToken('${p.mint}')"><td>${esc(p.name || p.mint.slice(0, 8))} <span class="sym">$${esc(p.symbol || '')}</span></td>
      <td>${hhmm(p.entry_wall_ms)}</td><td>${fmt(p.final, 2)} / ${fmt(p.threshold, 2)}</td>
      <td>${p.status === 'open' ? '<span class="badge b-watch">ouverte</span>' : esc((d.exits[p.exit_reason] || p.exit_reason || '').split(' (')[0])}</td>
      <td class="${cls(p.status === 'open' ? (p.value || 0) - (p.cost || 0) : p.pnl)}">${sol(p.status === 'open' ? (p.value || 0) - (p.cost || 0) : p.pnl, 4)}</td></tr>`).join('')}</tbody></table>`
    : '<div class="empty" style="padding:30px 0">Aucun achat pour l instant.</div>';

  board(d);

  const sig = JSON.stringify([d.equity.length, d.hourly.length, d.focus]);
  if (sig === mdSig && charts.chEquity) return;
  mdSig = sig;

  // gain cumule, une couleur par modele
  const ids = [...new Set(d.equity.map(e => e[2]))];
  const col = id => id === d.champion ? C.amber : MODEL_COLORS[ids.indexOf(id) % MODEL_COLORS.length];
  const top = ids.slice(-7);
  chart('chEquity', { type: 'line', data: { datasets: [
    { label: 'Gain cumule', data: d.equity.map(e => ({ x: e[0], y: e[1], e })), borderColor: '#5a6178', borderWidth: 1.5, pointRadius: 0, order: 2 },
    ...top.map(id => ({ label: mName(id).replace('Modele du ', ''), data: d.equity.filter(e => e[2] === id).map(e => ({ x: e[0], y: e[1], e })),
      showLine: false, pointRadius: 2.5, borderColor: col(id), backgroundColor: col(id), order: 1 })),
  ] }, options: { scales: { x: timeAxis, y: { title: { display: true, text: 'SOL cumules' } } },
    interaction: { mode: 'nearest', intersect: false },
    plugins: { tooltip: { callbacks: { title: it => hhmm(it[0].parsed.x), label: c => {
      const e = c.raw.e; return e ? [`${e[3] || 'coin'} : ${sol(e[4], 4)} SOL`, `cumul ${sol(e[1], 3)} SOL`, mName(e[2])] : ''; } } } } } });

  const hr = d.hourly;
  chart('chHourly', { type: 'bar', data: { labels: hr.map(h => new Date(h.h).toLocaleString('fr-FR', { day: '2-digit', hour: '2-digit' }) + 'h'),
    datasets: [{ label: 'Gain (SOL)', data: hr.map(h => h.pnl), backgroundColor: hr.map(h => h.pnl >= 0 ? C.green + 'cc' : C.red + 'cc') }] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => { const h = hr[c.dataIndex]; return [`${sol(h.pnl, 4)} SOL`, `${h.n} achats, ${h.wins} gagnants`]; } } } },
      scales: { y: { title: { display: true, text: 'SOL' } } } } });

  const rs = Object.entries(since.reasons || {}).sort((a, b) => b[1].n - a[1].n);
  chart('chExits', { type: 'bar', data: { labels: rs.map(([k]) => d.exits[k] || k),
    datasets: [{ label: 'Positions', data: rs.map(([, r]) => r.n), backgroundColor: rs.map(([, r]) => r.pnl >= 0 ? C.green + 'cc' : C.red + 'cc') }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false }, zoom: { zoom: { wheel: { enabled: false } }, pan: { enabled: false } },
      tooltip: { callbacks: { label: c => { const r = rs[c.dataIndex][1]; return [`${r.n} positions`, `total ${sol(r.pnl, 4)} SOL`, `moyenne ${sol(r.pnl / r.n, 4)} SOL`]; } } } },
      scales: { x: { title: { display: true, text: 'positions' }, beginAtZero: true } } } });
}

const COLS = [
  ['id', 'Modele', 'Date de l entrainement'], ['status', 'Statut', 'Champion = meilleur modele prouve ; en service = celui qui decide maintenant'],
  ['start_ms', 'En service', 'Periode pendant laquelle il a decide en direct'], ['hours', 'Duree', ''],
  ['trades', 'Achats', 'Achats executes en direct'], ['pnl', 'Gain direct', 'Gain realise en direct, prix Jupiter (SOL)'],
  ['win_rate', 'Gagnants', 'Part des achats gagnants en direct'], ['per_trade', 'Par achat', 'Gain moyen par achat en direct'],
  ['profit_factor', 'Fact. profit', 'Gains / pertes en direct (> 1 = gagnant)'], ['edge', 'Avantage tri', 'Gain simule moyen de ses achats - celui de tous les coins'],
  ['v_pnl', 'Gain test', 'Gain sur les heures jamais vues a l entrainement'], ['v_trades', 'Achats test', ''],
  ['v_auc', 'AUC test', 'Qualite du classement (0,5 = hasard)'], ['v_all', 'Tout acheter (test)', 'Reference sur les memes heures'],
  ['family', 'Type', ''],
];

function board(d) {
  const rows = d.models.map(m => ({ ...m, edge: (m.sim_buy_mean != null && m.sim_all_mean != null) ? m.sim_buy_mean - m.sim_all_mean : null,
    v_pnl: m.valid.valid_pnl, v_trades: m.valid.valid_trades, v_auc: m.valid.valid_auc, v_all: m.valid.valid_buy_all_pnl,
    family: (FAM[m.valid.family] || m.valid.family || '') + (m.valid.deep ? ' · profond' : ''),
    status: m.live ? 2 : m.champion ? 1 : 0 }));
  const k = mdSort.key;
  rows.sort((a, b) => { const x = a[k], y = b[k]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * mdSort.dir; });
  $('#mdBoard').innerHTML = `<table class="lb"><thead><tr>${COLS.map(([c, l, t]) => `<th data-k="${c}" title="${esc(t)}" class="${c === k ? 'sorted' : ''}">${l}${c === k ? (mdSort.dir < 0 ? ' ▼' : ' ▲') : ''}</th>`).join('')}</tr></thead><tbody>${
    rows.map(m => `<tr data-id="${m.id}" class="${m.id === selModel ? 'sel' : ''}">
      <td>${hhmm(idToMs(m.id))}</td>
      <td>${m.live ? '<span class="chip live">en service</span> ' : ''}${m.champion ? '<span class="chip gold">👑 champion</span>' : m.valid.promoted ? '<span class="chip">promu</span>' : m.live ? '' : '<span class="muted">-</span>'}</td>
      <td>${m.start_ms ? hhmm(m.start_ms) + ' → ' + (m.live ? 'maintenant' : hhmm(m.end_ms)) : '<span class="muted">jamais</span>'}</td>
      <td>${m.start_ms ? dur(m.hours) : '-'}</td>
      <td>${m.trades || '<span class="muted">0</span>'}</td>
      <td class="${cls(m.pnl)}">${m.closed ? sol(m.pnl, 3) : '-'}</td>
      <td>${m.closed ? pct(m.win_rate) : '-'}</td>
      <td class="${cls(m.per_trade)}">${m.closed ? sol(m.per_trade, 4) : '-'}</td>
      <td>${m.profit_factor == null ? '-' : m.profit_factor.toFixed(2).replace('.', ',')}</td>
      <td class="${cls(m.edge)}">${m.edge == null ? '-' : sol(m.edge, 4)}</td>
      <td class="${cls(m.v_pnl)}">${sol(m.v_pnl, 3)}</td><td>${m.v_trades ?? '-'}</td><td>${fmt(m.v_auc, 2)}</td>
      <td class="${cls(m.v_all)}">${sol(m.v_all, 2)}</td><td class="muted">${esc(m.family)}</td></tr>`).join('')}</tbody></table>`;
  $('#mdBoard thead').onclick = e => { const c = e.target.closest('th')?.dataset.k; if (!c) return;
    mdSort = { key: c, dir: mdSort.key === c ? -mdSort.dir : -1 }; board(mdData); };
  $('#mdBoard tbody').onclick = async e => { const tr = e.target.closest('tr'); if (!tr) return;
    if (!models) models = await api('/api/models');
    document.querySelectorAll('#mdBoard tr').forEach(r => r.classList.toggle('sel', r === tr));
    if (models.history.some(h => h.id === tr.dataset.id)) { await showModel(tr.dataset.id, 'mdDetail'); $('#mdDetail').scrollIntoView({ behavior: 'smooth' }); } };
}

setInterval(() => { if (view === 'models') loadModels(); }, 60000);
