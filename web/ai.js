// Vues "IA" (apprentissage, modeles, comparaison) et "Exemples" (coins importes par CA) + recherche.
// S'appuie sur les helpers de index.html : $, esc, sol, cls, ago, api, openToken, sendFb, view.
// Graphiques : Chart.js + plugin de zoom (molette / selection / pincement, double-clic pour revenir).

const C = { violet: '#a57bff', green: '#2fd27a', red: '#ff5c6c', amber: '#f5b942', blue: '#5b8cff', muted: '#8a93a8', grid: '#232838' };
let models = null, selModel = null, exTimer = null, aiSig = null;
const charts = {};
const fmt = (v, d = 3) => v == null ? '-' : Number(v).toFixed(d);
const pct = v => v == null ? '-' : `${(100 * v).toFixed(0)} %`;
const hhmm = ms => new Date(ms).toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
const idToMs = id => { const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/.exec(id); return m ? new Date(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime() : 0; };

// ---------- noms lisibles des indicateurs ----------
const FEAT = {
  n_trades: 'Nombre de trades (60 s)', n_buys: 'Nombre d\'achats (60 s)', n_sells: 'Nombre de ventes (60 s)',
  buy_sol: 'SOL achetes (60 s)', sell_sol: 'SOL vendus (60 s)', net_sol: 'SOL entres - sortis (60 s)',
  sell_buy_ratio: 'Ventes / achats', uniq_buyers: 'Acheteurs differents (60 s)', uniq_sellers: 'Vendeurs differents',
  uniq_traders: 'Wallets differents', avg_buy_sol: 'Taille moyenne d\'un achat', max_buy_sol: 'Plus gros achat',
  small_buys_frac: 'Part de petits achats (<0,05 SOL)', dev_buy_sol: 'Achat du createur (SOL)',
  dev_sold_frac: 'Part vendue par le createur', dev_n_trades: 'Trades du createur', top1_share: 'Part du 1er holder',
  top5_share: 'Part des 5 premiers holders', n_holders: 'Nombre de holders', early_buyers: 'Snipers (1ers blocs)',
  early_sol: 'SOL des snipers', repeat_traders_frac: 'Wallets qui tradent en boucle (bots)',
  price_mult: 'Prix a 60 s / prix de depart', max_mult: 'Plus haut / prix de depart', drawdown: 'Chute depuis le plus haut',
  progress: 'Avancement de la courbe', mcap_sol: 'Market cap a 60 s (SOL)', recent_trades: 'Trades des 15 dernieres s',
  recent_net_sol: 'SOL nets des 15 dernieres s', recent_price_chg: 'Variation de prix 15 dernieres s',
  since_last_trade_s: 'Secondes depuis le dernier trade', first_trade_delay_s: 'Delai avant le 1er trade',
  name_len: 'Longueur du nom', symbol_len: 'Longueur du ticker', symbol_upper: 'Ticker en majuscules',
  name_has_digit: 'Chiffres dans le nom', name_emoji: 'Emojis dans le nom', name_eq_symbol: 'Nom = ticker',
  initial_buy_sol: 'Achat initial', meta_ok: 'Metadonnees trouvees', desc_len: 'Longueur de la description',
  desc_words: 'Mots dans la description', has_twitter: 'A un lien X', has_telegram: 'A un Telegram',
  has_website: 'A un site', has_image: 'A une image', x_tweet: 'Lien X = un tweet', x_profile: 'Lien X = un profil',
  x_community: 'Lien X = une communaute', x_other: 'Autre lien', x_followers: 'Abonnes du compte X',
  x_following: 'Abonnements du compte X', x_verified: 'Compte X certifie', x_verified_business: 'Compte X entreprise',
  x_account_age_d: 'Age du compte X (jours)', x_statuses: 'Tweets du compte', x_likes: 'Likes du tweet',
  x_retweets: 'Retweets', x_replies: 'Reponses', x_quotes: 'Citations', x_views: 'Vues du tweet',
  x_tweet_age_min: 'Age du tweet (min)', x_is_reply: 'Le tweet est une reponse', x_mentions: 'Mentions dans le tweet',
  x_handle_in_name: 'Handle X dans le nom', hour_sin: 'Heure de la journee (a)', hour_cos: 'Heure de la journee (b)',
  dow: 'Jour de la semaine', creator_prev_tokens: 'Coins deja lances par ce createur',
  creator_prev_pnl: 'Resultat passe du createur', creator_prev_migrated: 'Migrations passees du createur',
  img_text_sim: 'L\'image colle au nom', sim_1h: 'Coins similaires sortis dans l\'heure',
  nn_pnl: 'Resultat des narratives similaires', nn_win: 'Taux de gain des narratives similaires',
  nn_sim: 'Ressemblance avec les narratives passees', dup_symbol: 'Copies : meme ticker avant',
  dup_name: 'Copies : meme nom avant', dup_twitter: 'Copies : meme lien X avant', dup_image: 'Copies : meme image avant',
  dup_any: 'Copies : total', is_first: 'Original (aucune copie avant)', dup_first_age_min: 'Age de l\'original (min)',
  dup_best_mult: 'Meilleur x de l\'original', dup_any_migrated: 'L\'original a migre', dup_best_pnl: 'Resultat de l\'original',
  dup_strong: 'Copie certaine (image / lien X / nom+ticker)',
};
const TH = { monde: 'actu mondiale', politique: 'politique', drame: 'drame / catastrophe', celebrite: 'celebrite',
  animal: 'animaux', crypto: 'crypto', tech_ia: 'tech / IA', sport: 'sport', humour: 'humour / meme', finance: 'finance', viral: 'viral' };
FEAT.x_tweet_fresh = 'Tweet frais (moins de 2 min au lancement)';
FEAT.coin_tweet_sim = 'Le coin parle de la meme chose que le tweet';
FEAT.x_name_in_tweet = 'Nom / ticker du coin present dans le tweet';
FEAT.x_ticker_cashtag = 'Le tweet cite le $TICKER du coin';
FEAT.web_real = 'Vrai site web (pas juste un lien X)';
FEAT.web_name_match = "L'adresse du site reprend le nom du coin";
FEAT.creator_wins = 'Gros gagnants deja lances par ce createur';
FEAT.creator_last_win_d = 'Jours depuis le dernier gros gagnant du createur';
FEAT.fame_text_sim = 'Ressemblance (texte) au gros gagnant le plus proche';
FEAT.fame_img_sim = 'Ressemblance (image) au gros gagnant le plus proche';
FEAT.fame_n_close = 'Nombre de gros gagnants sur le meme sujet';
FEAT.fame_age_d = 'Age (jours) du gros gagnant le plus proche';
FEAT.fame_same_name = 'Meme nom que ce gros gagnant (copie)';
FEAT.fame_twist = "Clin d'oeil a un gros gagnant (meme idee, autre nom)";
FEAT.flop_text_sim = 'Ressemblance (texte) au flop le plus proche';
FEAT.flop_img_sim = 'Ressemblance (image) au flop le plus proche';
FEAT.flop_n_close = 'Nombre de flops sur le meme sujet';
FEAT.fame_vs_flop = 'Plus proche des gagnants que des flops';
FEAT.block0_share = 'Part de la courbe achetee au bloc 0 (lancement groupe)';
FEAT.block0_wallets = 'Wallets acheteurs au bloc 0';
FEAT.x_rel_reply = 'Le tweet du coin repond a un autre tweet';
FEAT.x_rel_quote = 'Le tweet du coin cite un autre tweet';
FEAT.x_parent_followers = "Abonnes de l'auteur du tweet d'origine";
FEAT.x_parent_verified = "Auteur du tweet d'origine certifie";
FEAT.x_parent_likes = "Likes du tweet d'origine";
FEAT.x_parent_views = "Vues du tweet d'origine";
FEAT.x_parent_age_min = "Age du tweet d'origine au lancement (min)";
FEAT.x_same_author = "Reponse de l'auteur lui-meme (suite du tweet)";
FEAT.x_follower_gap = "Ecart d'audience reponse / tweet d'origine";
FEAT.x_name_in_reply = 'Nom du coin present dans la reponse';
FEAT.x_name_in_parent = "Nom du coin present dans le tweet d'origine";
FEAT.x_parasite = "Reponse parasite (petit compte sur le tweet viral d'un autre)";
FEAT.creator_n_1h = 'Coins lances par ce createur dans l\'heure';
FEAT.creator_n_6h = 'Coins lances par ce createur en 6 h';
FEAT.proto_bon_texte = 'Ressemble a tes bons exemples (texte)';
FEAT.proto_bon_image = 'Ressemble a tes bons exemples (image)';
FEAT.proto_mauvais_texte = 'Ressemble a tes mauvais exemples (texte)';
FEAT.proto_mauvais_image = 'Ressemble a tes mauvais exemples (image)';
FEAT.proto_bon_net = 'Plus proche des bons que des mauvais exemples';
FEAT.coin_gravite = 'Gravite du theme du coin';
FEAT.tw_gravite = "Gravite de l'actu du tweet";
for (const [k, v] of Object.entries(TH)) { FEAT['coin_th_' + k] = `Coin : theme ${v}`; FEAT['tw_th_' + k] = `Tweet : theme ${v}`; }
const featName = f => FEAT[f] || (f.startsWith('cat_') ? `Ta categorie "${f.slice(4).replace(/_/g, ' ')}"` : '') || (/^txt_(\d+)$/.test(f) ? `Narrative (texte), axe ${+f.slice(4) + 1}` : /^img_(\d+)$/.test(f) ? `Image, axe ${+f.slice(4) + 1}` : f);

// ---------- navigation ----------
function setView(v) {
  view = v;
  document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  $('#v-coins').hidden = v !== 'coins'; $('#v-ai').hidden = v !== 'ai'; $('#v-models').hidden = v !== 'models'; $('#v-ex').hidden = v !== 'ex'; $('#v-amm').hidden = v !== 'amm';
  $('#tabs').style.display = v === 'coins' ? '' : 'none';
  if (v === 'ai') loadAI();
  if (v === 'models') loadModels();
  if (v === 'ex') loadExamples();
  if (v === 'amm') loadAmm();
}

// ---------- vue Apres migration ----------
const RUG = {migre_trop_vite: "Courbe remplie en moins d'1 min", achat_groupe_bloc0: 'Achat groupe au bloc 0',
  detention_concentree: 'Detention concentree (top 10)', historique_inconnu: 'Historique de courbe illisible'};
async function loadAmm() {
  const r = await (await fetch('/api/amm')).json();
  const g = Object.fromEntries(r.summary.map(s => [s.grp, s]));
  const box = (k, t) => { const s = g[k] || {}; return `<div class="card" style="padding:12px 14px"><b>${t}</b><div class="feat" style="margin-top:6px">
    <span>Coins</span><span>${s.n || 0}</span><span>Resultat connu</span><span>${s.resolus || 0}</span>
    <span>PnL simule moyen / achat</span><span class="${cls(s.pnl_moy)}">${sol(s.pnl_moy, 4)}</span>
    <span>Gagnants</span><span>${pct(s.gagnants)}</span><span>Plus haut moyen apres achat</span><span>${s.max_mult_moy ? 'x' + s.max_mult_moy.toFixed(2) : '-'}</span>
    ${k === 'eligible' ? `<span>Routables par Jupiter</span><span>${pct(s.routables)}</span><span>Jupiter / simulation (tokens)</span><span>${s.jup_vs_sim ? s.jup_vs_sim.toFixed(3) : '-'}</span>` : ''}
    </div></div>`; };
  $('#v-amm').innerHTML = `<p style="color:var(--muted);max-width:80ch">Coins qui ont fini leur courbe et sont passes sur PumpSwap.
    <b>Phase d'observation : aucun achat.</b> Chaque migration est classee <b>eligible</b> ou <b>exclue</b> (lancement groupe / rug),
    puis on simule un achat de 0,1 SOL ${15} s apres la migration avec la strategie actuelle. Si les exclus perdent nettement plus
    que les eligibles, le filtre fait son travail.</p>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:12px 0">${box('eligible', 'Eligibles')}${box('exclu', 'Exclus (rug probable)')}</div>
    <div class="card" style="padding:8px;overflow-x:auto"><table><tr><th>Coin</th><th>Migre</th><th>Verdict</th><th>Courbe</th><th>Bloc 0</th><th>Top 10</th><th>Jupiter</th><th>PnL simule</th><th>Plus haut</th><th>Sortie</th></tr>
    ${r.rows.map(d => `<tr><td><a href="https://pump.fun/coin/${esc(d.mint)}" target="_blank">${esc(d.name || d.mint.slice(0, 8))}</a> <span style="color:var(--muted)">$${esc(d.symbol || '')}</span></td>
      <td>${ago(d.migrated_ms)}</td>
      <td style="text-align:left">${d.reasons ? d.reasons.split(',').map(x => `<span class="neg">${esc(RUG[x] || x)}</span>`).join('<br>') : '<span class="pos">eligible</span>'}</td>
      <td>${d.curve_s == null ? '-' : d.curve_s < 120 ? d.curve_s.toFixed(0) + ' s' : (d.curve_s / 60).toFixed(0) + ' min'}</td>
      <td>${pct(d.block0_share)}${d.block0_wallets ? ' / ' + d.block0_wallets + ' w' : ''}</td><td>${pct(d.top10_share)}</td>
      <td>${d.jup_route ? esc(d.jup_route) : d.reasons ? '' : '-'}</td>
      <td class="${cls(d.pnl_sim)}">${d.resolved ? sol(d.pnl_sim, 4) : 'en cours'}</td>
      <td>${d.max_mult ? 'x' + d.max_mult.toFixed(2) : '-'}</td><td>${esc(d.exit_reason_sim || '')}</td></tr>`).join('')}</table></div>`;
}
$('#nav').onclick = e => e.target.dataset.v && setView(e.target.dataset.v);

// ---------- recherche par CA ----------
$('#search').onsubmit = async e => {
  e.preventDefault();
  const q = $('#q').value.trim(); if (!q) return;
  const box = $('#searchres'); box.hidden = false;
  const r = await api('/api/search?q=' + encodeURIComponent(q));
  if (!r.mint) { box.innerHTML = '<div class="box">Aucune adresse Solana reconnue dans ce texte.</div>'; return; }
  const short = r.mint.slice(0, 6) + '…' + r.mint.slice(-6);
  let html = `<div class="box"><b>${short}</b>`;
  if (r.seen) html += ` vu en direct par l'IA <button class="btn" onclick="openToken('${r.mint}')">Ouvrir et noter</button>`;
  else if (r.example) html += ` deja importe (${esc(r.example.status)}) <button class="btn" onclick="setView('ex')">Voir les exemples</button>`;
  else if (r.collecting) html += ` ${esc(r.collecting.name)} : en cours d'observation, l'IA le notera a 60 s.`;
  else html += ` pas dans nos donnees. <button class="btn" id="doImport">Importer depuis la blockchain</button>
    <span class="note">On relit sa creation et ses premieres minutes de trades, puis tu pourras le noter comme exemple.</span>`;
  box.innerHTML = html + ` <button class="btn ghost" onclick="$('#searchres').hidden=true">×</button></div>`;
  const b = $('#doImport');
  if (b) b.onclick = async () => {
    await api('/api/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q: r.mint }) });
    box.hidden = true; $('#q').value = ''; setView('ex');
  };
};

// ---------- graphiques (Chart.js) ----------
if (window.Chart) {
  Chart.defaults.color = C.muted;
  Chart.defaults.borderColor = C.grid;
  Chart.defaults.font.family = 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
  Chart.defaults.plugins.tooltip.backgroundColor = '#0b0d12ee';
  Chart.defaults.plugins.tooltip.borderColor = '#39405a';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
}
const ZOOM = {
  zoom: { wheel: { enabled: true }, pinch: { enabled: true }, drag: { enabled: true, modifierKey: 'shift', backgroundColor: '#a57bff22' }, mode: 'x' },
  pan: { enabled: true, mode: 'x' },
};
function chart(id, cfg) {
  const el = document.getElementById(id); if (!el || !window.Chart) return;
  if (charts[id]) charts[id].destroy();
  cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false } }, cfg.options || {});
  cfg.options.plugins = Object.assign({ zoom: ZOOM, legend: { labels: { boxWidth: 12, usePointStyle: true } } }, cfg.options.plugins || {});
  charts[id] = new Chart(el, cfg);
  el.ondblclick = () => charts[id].resetZoom();
}
const timeAxis = { type: 'time', time: { tooltipFormat: 'dd/MM HH:mm', displayFormats: { minute: 'HH:mm', hour: 'dd/MM HH:mm' } }, ticks: { maxRotation: 0 } };
const panel = (id, title, what, how, h = 260) => `
  <div class="panel"><div class="ph"><h2>${title}</h2><button class="btn ghost small" onclick="charts['${id}']?.resetZoom()" title="Revenir a la vue complete">↺ vue complete</button></div>
    <p class="what">${what}</p>
    <div style="height:${h}px"><canvas id="${id}"></canvas></div>
    <p class="how">💡 ${how}</p></div>`;

// ---------- vue IA ----------
function verdict(st, m) {
  const h = m.history, last = h[h.length - 1], res = st.total.resolved || 0, need = st.min_train_rows;
  if (!h.length) return { cls: 'wait', title: 'L\'IA apprend encore ses bases',
    text: `Elle a besoin de ${need} coins dont on connait le resultat pour s'entrainer une premiere fois. Il y en a ${res}. Chaque coin met environ 37 minutes a avoir tous ses resultats (7 min d'analyse + 30 min de detention).` };
  const champ = h.find(r => r.id === m.champion);
  const wins = Math.round((last.base_win_rate || 0) * last.valid_tokens);
  if (!champ) return { cls: 'wait', title: 'Des modeles existent, mais aucun n\'est encore fiable',
    text: `Le dernier modele a ete teste sur ${last.valid_tokens} coins recents, dont seulement ${wins} gagnant(s). Il n'a pas encore prouve qu'il gagne de l'argent : il note les coins en mode "non promu" en attendant. Il faut beaucoup plus de gagnants pour juger (idealement plusieurs dizaines).` };
  return { cls: champ.valid_pnl > 0 ? 'ok' : 'ko', title: `Modele en service : ${hhmm(idToMs(champ.id))}`,
    text: `Sur les heures les plus recentes (jamais vues a l'entrainement), il aurait fait ${sol(champ.valid_pnl, 2)} SOL en ${champ.valid_trades} achats, contre ${sol(champ.valid_buy_all_pnl, 2)} SOL en achetant tous les coins.` };
}

async function loadAI() {
  const el = $('#v-ai');
  if (!el.innerHTML) el.innerHTML = `
    <div id="aiVerdict"></div>
    <details class="panel glossary"><summary>📖 Comment lire cette page (glossaire)</summary>
      <dl>
        <dt>Points d'analyse</dt><dd>L'IA analyse chaque coin des 5 s d'existence, puis a 10, 15, 20, 30, 45 s, 1, 1,5, 2, 3, 5 et 7 min. Elle achete au premier point ou son score depasse le seuil ; apres 7 min, plus aucun achat sur ce coin.</dd>
        <dt>Coin "resolu"</dt><dd>Un coin dont on connait le resultat : pour chaque point d'analyse, on simule un achat a ce moment-la et on regarde les 30 min suivantes (vente a +100 %, a -40 % ou au bout de 30 min).</dd>
        <dt>PnL</dt><dd>Gain ou perte en SOL, frais compris. Les graphiques montrent le PnL qu'aurait fait l'IA avec une mise de 0,1 SOL par coin achete.</dd>
        <dt>Validation</dt><dd>Les heures les plus recentes, que l'IA n'a jamais vues pendant son entrainement. C'est le seul test honnete : on lui demande de choisir comme en vrai.</dd>
        <dt>"Acheter tout"</dt><dd>La reference : ce qu'on gagnerait (ou perdrait) en achetant tous les coins sans reflechir. Une IA utile fait mieux que ca.</dd>
        <dt>AUC</dt><dd>Est-ce que l'IA classe bien les coins, du pire au meilleur ? 0,5 = pas mieux que le hasard, 1 = parfait. Sur pump.fun, au-dessus de 0,6 c'est deja un vrai signal. Avec tres peu de gagnants, l'AUC n'est pas fiable.</dd>
        <dt>Champion / promu</dt><dd>Le modele en service. Un nouveau modele ne le remplace que s'il gagne plus d'argent sur la meme validation. "Non promu" = modele utilise faute de mieux, pas encore prouve.</dd>
        <dt>Seuil</dt><dd>Le score minimum pour acheter. Il est choisi automatiquement pour maximiser le gain sur la fin de l'entrainement.</dd>
        <dt>Copie (vamp)</dt><dd>Un coin qui reprend le ticker, le nom, l'image ou le lien X d'un coin sorti avant lui.</dd>
        <dt>Zoom</dt><dd>Molette pour zoomer, cliquer-glisser pour se deplacer, Shift + selection pour zoomer sur une zone, double-clic pour revenir. Passe la souris sur un point pour le detail.</dd>
      </dl></details>
    <div class="two" style="margin-top:16px">
      <div class="panel"><h2>Etat</h2><p class="what">Les chiffres cles de l'apprentissage et de la collecte.</p><div id="aiStatus"></div></div>
      ${panel('chHours', 'Activite par heure', 'Combien de coins l\'IA a notes chaque heure, combien etaient des copies, et quelle part aurait ete gagnante si on les avait achetes.', 'Les barres sont des nombres de coins (axe de gauche). La ligne verte est un pourcentage (axe de droite). Une heure avec beaucoup de copies est souvent une heure "a la mode".')}
    </div>
    <div class="two" style="margin-top:16px">
      ${panel('chPnl', 'Combien chaque modele aurait gagne', 'Chaque point est un entrainement. On le teste sur les heures recentes qu\'il n\'a jamais vues et on mesure ce qu\'il aurait gagne en SOL.', 'Violet = le nouveau modele, orange = le champion sur le meme test, gris = acheter tous les coins. Plus c\'est haut, mieux c\'est. Au-dessus de 0 = gagnant.')}
      ${panel('chAuc', 'Qualite du classement (AUC)', 'Est-ce que l\'IA met les meilleurs coins en haut de sa liste ?', 'La ligne pointillee a 0,5 = le hasard. Au-dessus = l\'IA voit quelque chose. Peu fiable tant qu\'il y a peu de gagnants dans le test.')}
    </div>
    <div class="panel" style="margin-top:16px"><h2>Tous les entrainements</h2>
      <p class="what">Clique sur une ligne pour voir le detail de ce modele (ce qu'il regarde, ses erreurs, ses reglages). Les colonnes "En direct" montrent ce qu'il a fait sur les vrais coins quand il etait en service.</p>
      <div id="aiTable" style="overflow-x:auto"></div></div>
    <div id="aiDetail" style="margin-top:16px"></div>`;
  const [m, st] = await Promise.all([api('/api/models'), api('/api/datastats')]);
  models = m;
  const v = verdict(st, m);
  $('#aiVerdict').innerHTML = `<div class="verdict ${v.cls}"><div class="vt">${v.title}</div><div>${v.text}</div></div>`;

  // ---- etat ----
  const res = st.total.resolved || 0, need = st.min_train_rows, h = m.history, last = h[h.length - 1];
  const pp = st.paper || {}, co = st.collector || {};
  const exclu = (pp.exclu_courbe || 0) + (pp.exclu_coupure || 0) + (pp.exclu_migre || 0);
  $('#aiStatus').innerHTML = `
    <div class="feat">
      <span title="Coins dont le resultat de chaque point d'analyse est connu">Coins avec resultat connu</span><span><b>${res}</b> ${res < need ? `/ ${need} pour le 1er modele` : ''}</span>
      ${res < need ? `<div class="bar" style="grid-column:1/3;margin:4px 0 8px"><i style="width:${Math.min(100, 100 * res / need)}%"></i></div>` : ''}
      <span>Coins notes au total</span><span>${st.total.n || 0}</span>
      <span>Entrainements</span><span>${h.length} (un toutes les ${st.retrain_every_hours} h)</span>
      <span>Dernier entrainement</span><span>${last ? hhmm(idToMs(last.id)) + (last.promoted ? ' · promu ✅' : ' · non promu') : '-'}</span>
      <span title="Coins pas notes dans la derniere heure : courbe non standard, trou de collecte ou deja migre">Coins ecartes (derniere heure)</span><span>${exclu} (courbe ${pp.exclu_courbe || 0}, trou de donnees ${pp.exclu_coupure || 0}, migre ${pp.exclu_migre || 0}) · notes ${pp.notes || 0}</span>
      ${co.budget_jour ? `<span title="Credits Helius estimes depuis minuit (flux coin par coin + rattrapages)">Credits Helius aujourd'hui</span><span>${co.credits_jour ?? '-'} / ${co.budget_jour} · ${co.suivis_actifs ?? '-'} coins suivis en ce moment</span>` : ''}
      <span title="Ecart moyen entre le gain reel au prix Jupiter et le gain simule par la courbe">Jupiter vs simulation</span><span>${st.calib.n ? `${st.calib.n} positions : ecart moyen ${sol(st.calib.gap, 4)} SOL` : 'pas encore de position fermee'}</span>
    </div>`;

  // Rafraichissement : on ne redessine les graphiques que s'il y a du nouveau (sinon ton zoom serait perdu)
  const sig = JSON.stringify([h.length, st.hours.length, st.hours.at(-1)?.n]);
  if (sig === aiSig && charts.chPnl) return;
  aiSig = sig;

  // ---- activite par heure ----
  const hourLbl = ms => new Date(ms).toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit' }) + 'h';
  chart('chHours', { data: { labels: st.hours.map(r => hourLbl(r.h)), datasets: [
    { type: 'bar', label: 'Coins notes', data: st.hours.map(r => r.n), backgroundColor: C.violet + 'aa', yAxisID: 'y', order: 2 },
    { type: 'bar', label: 'dont copies', data: st.hours.map(r => r.copies), backgroundColor: C.red + 'aa', yAxisID: 'y', order: 2 },
    { type: 'line', label: '% gagnants si achetes', data: st.hours.map(r => r.win == null ? null : 100 * r.win), borderColor: C.green, backgroundColor: C.green, yAxisID: 'y1', tension: .2, order: 1 },
  ] }, options: { scales: { x: { title: { display: true, text: 'heure' } }, y: { title: { display: true, text: 'coins' }, beginAtZero: true },
    y1: { position: 'right', title: { display: true, text: '% gagnants' }, beginAtZero: true, grid: { drawOnChartArea: false } } },
    plugins: { tooltip: { callbacks: { label: c => c.dataset.yAxisID === 'y1' ? `${c.dataset.label} : ${c.parsed.y?.toFixed(1)} %` : `${c.dataset.label} : ${c.parsed.y}` } } } } });

  // ---- evolution des modeles ----
  const pts = key => h.map(r => ({ x: idToMs(r.id), y: r[key], r }));
  const tipModel = c => {
    const r = c.raw.r; if (!r) return '';
    if (c.datasetIndex === 0) return [`Nouveau modele : ${sol(r.valid_pnl, 3)} SOL`, `  ${r.valid_trades} achats sur ${r.valid_tokens} coins testes`, `  gagnants parmi ses achats : ${pct(r.valid_win_rate_selected)}`, `  ${r.promoted ? 'PROMU champion' : 'non promu'}`];
    if (c.datasetIndex === 1) return r.champion_valid_pnl == null ? 'Pas de champion a ce moment' : `Champion sur le meme test : ${sol(r.champion_valid_pnl, 3)} SOL (${r.champion_valid_trades} achats)`;
    return `Acheter tout : ${sol(r.valid_buy_all_pnl, 3)} SOL`;
  };
  chart('chPnl', { type: 'line', data: { datasets: [
    { label: 'Nouveau modele', data: pts('valid_pnl'), borderColor: C.violet, backgroundColor: C.violet, pointRadius: 4 },
    { label: 'Champion', data: pts('champion_valid_pnl'), borderColor: C.amber, backgroundColor: C.amber, pointRadius: 3 },
    { label: 'Acheter tout', data: pts('valid_buy_all_pnl'), borderColor: C.muted, borderDash: [5, 4], pointRadius: 2 },
  ] }, options: { scales: { x: timeAxis, y: { title: { display: true, text: 'SOL gagnes sur le test' } } },
    plugins: { tooltip: { callbacks: { title: it => 'Entrainement du ' + hhmm(it[0].parsed.x), label: tipModel } } } } });
  chart('chAuc', { type: 'line', data: { datasets: [
    { label: 'Nouveau modele', data: pts('valid_auc'), borderColor: C.blue, backgroundColor: C.blue, pointRadius: 4 },
    { label: 'Champion', data: pts('champion_valid_auc'), borderColor: C.amber, backgroundColor: C.amber, pointRadius: 3 },
    { label: 'Hasard (0,5)', data: h.length ? [{ x: idToMs(h[0].id), y: .5 }, { x: idToMs(h[h.length - 1].id), y: .5 }] : [], borderColor: C.muted, borderDash: [4, 4], pointRadius: 0 },
  ] }, options: { scales: { x: timeAxis, y: { min: 0, max: 1, title: { display: true, text: 'AUC' } } },
    plugins: { tooltip: { callbacks: { title: it => 'Entrainement du ' + hhmm(it[0].parsed.x),
      label: c => c.datasetIndex === 2 ? 'Hasard : 0,5' : `${c.dataset.label} : ${fmt(c.parsed.y)}${c.raw.r ? ` (teste sur ${c.raw.r.valid_tokens} coins, ${Math.round((c.raw.r.base_win_rate || 0) * c.raw.r.valid_tokens)} gagnants)` : ''}` } } } } });

  // ---- tableau ----
  const rows = h.slice().reverse();
  const th = (t, tip) => `<th title="${tip}">${t}</th>`;
  $('#aiTable').innerHTML = rows.length ? `<table><thead><tr>${th('Entrainement', 'Date et heure')}${th('Statut', 'Champion = en service')}
    ${th('Teste sur', 'Coins de la validation (heures recentes jamais vues)')}${th('Gagnants', 'Coins gagnants dans le test')}
    ${th('Achats', 'Coins que le modele aurait achetes dans le test')}${th('Gain', 'PnL en SOL sur le test')}${th('Acheter tout', 'PnL si on achete tout')}
    ${th('AUC', 'Qualite du classement')}${th('Appris sur', 'Coins utilises pour apprendre')}${th('Tes avis', 'Avis utilises')}
    ${th('En direct : notes', 'Coins notes en vrai par ce modele')}${th('En direct : achats', 'Achats en vrai')}${th('En direct : gain', 'Gain simule des achats en vrai (si resolu)')}</tr></thead><tbody>` +
    rows.map(r => { const L = m.live[r.id] || {}; const wins = Math.round((r.base_win_rate || 0) * r.valid_tokens); return `<tr data-id="${r.id}" class="${r.id === selModel ? 'sel' : ''}">
      <td>${hhmm(idToMs(r.id))}</td><td>${r.id === m.champion ? '👑 champion' : r.promoted ? 'promu' : 'non promu'}</td>
      <td>${r.valid_tokens}</td><td>${wins}</td><td>${r.valid_trades}</td>
      <td class="${cls(r.valid_pnl)}">${sol(r.valid_pnl, 3)}</td><td class="${cls(r.valid_buy_all_pnl)}">${sol(r.valid_buy_all_pnl, 2)}</td>
      <td>${fmt(r.valid_auc, 2)}</td><td>${r.train_tokens}${r.backfill_used ? ` (+${r.backfill_used} ex.)` : ''}</td><td>${r.feedback_used}</td>
      <td>${L.n ?? '-'}</td><td>${L.buys ?? '-'}</td><td class="${cls(L.buy_pnl)}">${L.buy_resolved ? sol(L.buy_pnl, 3) : '-'}</td></tr>`; }).join('') + '</tbody></table>'
    : `<div class="empty">Aucun entrainement pour l'instant : il faut ${need} coins avec un resultat connu (${res} aujourd'hui). Le premier se lance tout seul.</div>`;
  $('#aiTable tbody')?.addEventListener('click', e => { const tr = e.target.closest('tr'); if (tr) showModel(tr.dataset.id); });
  if (rows.length && (!selModel || !h.some(r => r.id === selModel))) showModel(rows[0].id); else if (selModel) showModel(selModel);
}

async function showModel(id, target = 'aiDetail') {
  selModel = id;
  const other = target === 'aiDetail' ? 'mdDetail' : 'aiDetail';
  if ($('#' + other)) $('#' + other).innerHTML = '';
  document.querySelectorAll('#aiTable tr').forEach(tr => tr.classList.toggle('sel', tr.dataset.id === id));
  const d = await api('/api/model/' + id), M = d.metrics;
  const h = models.history, i = h.findIndex(r => r.id === id), prev = i > 0 ? h[i - 1] : null;
  let prevRank = {};
  if (prev) try { (await api('/api/model/' + prev.id)).importance.forEach(([f], k) => prevRank[f] = k); } catch (e) {}
  const wins = Math.round((M.base_win_rate || 0) * M.valid_tokens);
  const coinList = (arr, empty) => arr.length ? `<table><thead><tr><th>Coin</th><th title="Score donne par l'IA (0 a 1)">Score IA</th><th title="Ce qu'aurait rapporte l'achat">Resultat</th><th title="Plus haut atteint apres l'achat">Pic</th></tr></thead><tbody>` +
    arr.map(c => `<tr onclick="openToken('${c.mint}')"><td>${esc(c.name)} <span class="sym">$${esc(c.symbol)}</span></td><td>${fmt(c.score)}</td>
      <td class="${cls(c.pnl)}">${sol(c.pnl)}</td><td>x${fmt(c.max_mult_after, 1)}</td></tr>`).join('') + '</tbody></table>' : `<div class="note">${empty}</div>`;
  $('#' + target).innerHTML = `
    <div class="panel"><h2>Detail du modele du ${hhmm(idToMs(id))} ${id === models.champion ? '👑' : ''}</h2>
      <p class="what">Teste sur <b>${M.valid_tokens} coins</b> des ${M.valid_hours < 1 ? Math.round(M.valid_hours * 60) + ' dernieres minutes' : M.valid_hours + ' dernieres heures'} (dont <b>${wins} gagnant${wins > 1 ? 's' : ''}</b>).
      Il en aurait achete <b>${M.valid_trades}</b>${M.valid_buy_age_s != null ? ` (en moyenne a ${Math.round(M.valid_buy_age_s)} s d'existence)` : ''} pour un resultat de <b class="${cls(M.valid_pnl)}">${sol(M.valid_pnl)} SOL</b>, contre
      <b class="${cls(M.valid_buy_all_pnl)}">${sol(M.valid_buy_all_pnl)} SOL</b> en achetant tout.
      Il a appris sur ${M.train_tokens} coins (${M.data_hours} h de donnees)${M.backfill_used ? `, plus ${M.backfill_used} exemple(s) importe(s)` : ''}${M.feedback_used ? ` et ${M.feedback_used} de tes avis` : ''}.
      ${wins < 10 ? '<br>⚠️ Moins de 10 gagnants dans le test : ces chiffres sont encore du bruit, ne pas s\'y fier.' : ''}</p>
    </div>
    <div class="two" style="margin-top:16px">
      ${panel('chCurve', 'Gains cumules sur le test', 'Si on avait suivi ce modele pendant la periode de test, voici comment aurait evolue le gain total, achat apres achat.', 'Une courbe qui monte regulierement = un modele qui gagne souvent. Des marches brusques = quelques gros gains (ou pertes).')}
      ${panel('chBins', 'Est-ce que les meilleurs scores rapportent le plus ?', 'Les coins du test sont ranges en 10 groupes, du score le plus bas (1) au plus haut (10). Pour chaque groupe : gain moyen si on les avait tous achetes.', 'Un bon modele a des barres qui montent vers la droite : plus il est confiant, plus ca rapporte.')}
    </div>
    <div class="two" style="margin-top:16px">
      ${panel('chImp', 'Ce que le modele regarde le plus', 'Les indicateurs qui ont le plus servi a ses decisions (importance). En dessous de la barre : evolution par rapport au modele precedent.', '▲ = devenu plus important qu\'au modele d\'avant, ▼ = moins, 🆕 = nouveau. Ca montre ce que l\'IA a "appris" entre deux entrainements.', 520)}
      <div class="panel"><h2>Reglages essayes</h2>
        <p class="what">A chaque entrainement, ${(d.trials || []).length} facons de construire le modele sont testees (plus ou moins simple, plus ou moins prudent). La meilleure (✓) est gardee.</p>
        <table><thead><tr><th>#</th><th title="Complexite des arbres de decision">Complexite</th><th title="Vitesse d'apprentissage">Vitesse</th><th title="Exemples minimum par regle (plus = plus prudent)">Prudence</th><th title="Qualite du classement sur la fin de l'entrainement">AUC</th><th title="Gain sur la fin de l'entrainement">Gain</th><th>Achats</th></tr></thead>
        <tbody>${(d.trials || []).map(t => `<tr class="${t.i === M.best_trial ? 'sel' : ''}"><td>${t.i + 1}${t.i === M.best_trial ? ' ✓' : ''}</td><td>${t.params.num_leaves}</td>
          <td>${t.params.learning_rate}</td><td>${t.params.min_data_in_leaf}</td><td>${fmt(t.es_auc, 2)}</td><td class="${cls(t.es_pnl)}">${sol(t.es_pnl, 3)}</td><td>${t.es_trades}</td></tr>`).join('')}</tbody></table>
      </div>
    </div>
    <div class="two" style="margin-top:16px">
      <div class="panel"><h2>Ses erreurs : achats perdants</h2><p class="what">Les coins qu'il aurait achetes avec le plus de confiance et qui ont perdu. Clique pour ouvrir la fiche.</p>${coinList(d.bad_buys || [], 'Aucun achat perdant sur le test.')}</div>
      <div class="panel"><h2>Ses erreurs : gagnants rates</h2><p class="what">Les meilleurs coins qu'il n'aurait PAS achetes. C'est la que ton 👍 l'aide le plus : ouvre-les et note-les.</p>${coinList(d.missed || [], 'Aucun gagnant rate.')}</div>
    </div>`;

  chart('chCurve', { type: 'line', data: { datasets: [{ label: 'Gain cumule (SOL)', data: (d.valid_curve || []).map(([x, y]) => ({ x, y })), borderColor: C.green, backgroundColor: C.green + '22', fill: true, pointRadius: 0, stepped: true }] },
    options: { scales: { x: timeAxis, y: { title: { display: true, text: 'SOL' } } }, plugins: { tooltip: { callbacks: { label: c => `Gain cumule : ${sol(c.parsed.y)} SOL` } } } } });
  const bins = d.bins || [];
  chart('chBins', { type: 'bar', data: { labels: bins.map((b, k) => `${k + 1}`), datasets: [{ label: 'Gain moyen par coin (SOL)', data: bins.map(b => b.pnl_mean), backgroundColor: bins.map(b => b.pnl_mean >= 0 ? C.green + 'cc' : C.red + 'cc') }] },
    options: { plugins: { zoom: { zoom: { wheel: { enabled: false } }, pan: { enabled: false } }, tooltip: { callbacks: {
      title: it => `Groupe ${it[0].label} (${it[0].label === '1' ? 'scores les plus bas' : it[0].label === String(bins.length) ? 'scores les plus hauts' : 'scores intermediaires'})`,
      label: c => { const b = bins[c.dataIndex]; return [`${b.n} coins, ${pct(b.win)} gagnants`, `gain moyen ${sol(b.pnl_mean, 4)} SOL`, `score moyen ${fmt(b.pred)}`]; } } } },
      scales: { x: { title: { display: true, text: 'groupe de score (1 = bas, 10 = haut)' } }, y: { title: { display: true, text: 'SOL par coin' } } } } });
  const imp = d.importance.slice(0, 20), maxG = Math.max(...imp.map(x => x[1]), 1);
  chart('chImp', { type: 'bar', data: { labels: imp.map(([f], k) => { const p = prevRank[f]; const a = p == null ? (prev ? ' 🆕' : '') : p > k ? ` ▲${p - k}` : p < k ? ` ▼${k - p}` : ''; return featName(f) + a; }),
    datasets: [{ label: 'Importance (% du maximum)', data: imp.map(x => 100 * x[1] / maxG), backgroundColor: C.violet + 'cc' }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false }, zoom: { zoom: { wheel: { enabled: false } }, pan: { enabled: false } },
      tooltip: { callbacks: { title: it => featName(imp[it[0].dataIndex][0]), label: c => `importance : ${c.parsed.x.toFixed(0)} % de l'indicateur le plus utile` } } },
      scales: { x: { title: { display: true, text: '% de l\'indicateur le plus utile' }, max: 100 }, y: { ticks: { autoSkip: false, font: { size: 11 } } } } } });
}

// ---------- vue Exemples ----------
async function loadExamples() {
  const el = $('#v-ex');
  if (!el.innerHTML) el.innerHTML = `
    <div class="panel"><h2>Apprendre a l'IA avec des coins passes</h2>
      <form class="search" id="exForm" style="max-width:none;margin:0"><input id="exQ" placeholder="Colle le CA (ou le lien) d'un coin qui a bien marche... ou d'un rug"><button>Importer</button></form>
      <div class="note">L'IA relit sur la blockchain sa creation et ses premieres minutes de trades, calcule les memes indicateurs qu'en direct,
        puis le coin sert d'exemple a chaque entrainement. Les coins tres actifs peuvent prendre quelques minutes a importer.</div>
      <div class="what" style="margin-top:12px"><b>Mode d'emploi pour chaque exemple</b> (clique sur la carte une fois importee) :
        <ol style="margin:6px 0 0 18px;padding:0;line-height:1.7">
          <li><b>👍 ou 👎</b> : coin a reproduire, ou a eviter. Mets aussi des mauvais exemples (rugs, copies) : l'IA apprend par contraste.</li>
          <li><b>Categorie(s) de narrative</b> : "news du jour", "animal", "celebrite"... ou cree la tienne. Chaque categorie devient un filtre :
            l'IA mesure a quel point un nouveau coin ressemble a tes bons exemples de cette categorie. Vise au moins 3 a 5 exemples par categorie.</li>
          <li><b>Une phrase d'explication</b> : pourquoi ca marche (le tweet, le timing, l'image, le premier sur la narrative...). Elle est gardee
            avec l'exemple pour toi et pour les prochaines ameliorations.</li>
        </ol>
        Ce qui compte le plus : des coins vraiment differents, et pas que des gagnants. Les coins deja sortis depuis plusieurs jours fonctionnent aussi.</div>
    </div>
    <div class="grid" id="exGrid" style="margin-top:16px"></div>`;
  $('#exForm').onsubmit = async e => {
    e.preventDefault(); const q = $('#exQ').value.trim(); if (!q) return;
    try { await api('/api/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q }) }); $('#exQ').value = ''; }
    catch (err) { alert('CA invalide'); }
    loadExamples();
  };
  const rows = await api('/api/examples');
  $('#exGrid').innerHTML = rows.length ? rows.map(r => {
    const img = r.image_path ? `<img class="img" src="/img/${encodeURI(r.image_path.replace(/\\/g, '/'))}">` : '<div class="img"></div>';
    const done = r.status === 'ok' || r.status === 'narratif';
    return `<div class="card ${done ? 'buy' : 'observe'}" data-m="${r.mint}">
      <div class="row">${img}<div style="flex:1;min-width:0">
        <div class="row" style="justify-content:space-between"><div class="name">${esc(r.name || r.mint.slice(0, 8) + '…')}</div>
        <span class="badge ${r.status === 'erreur' ? 'b-skip' : done ? 'b-buy' : 'b-observe'}">${esc(r.status)}</span></div>
        <div class="sym">${r.symbol ? '$' + esc(r.symbol) + ' · ' : ''}${r.created_ms ? 'cree ' + hhmm(r.created_ms) : 'ajoute ' + ago(r.added_ms)}</div></div></div>
      <div class="desc">${esc(r.msg || r.description || '')}</div>
      ${r.status === 'narratif' ? `<div class="meta"><span>narratif seul (pas de trades rejouables) : sert a la bibliotheque de narratives</span></div>
      <div class="fb"><button class="up ${r.verdict === 1 ? 'on' : ''}" data-v="1">👍 a reproduire</button><button class="down ${r.verdict === -1 ? 'on' : ''}" data-v="-1">👎 a eviter</button></div>` : ''}
      ${r.status === 'ok' ? `<div class="meta"><span>${r.n_trades} trades lus · pic x${fmt((r.entry_mult || 1) * (r.max_mult_after || 1), 1)}${r.migrated ? ' · migre' : ''}</span>
        <span>si achete a 5 s <b class="${cls(r.pnl_sim)}">${sol(r.pnl_sim)}</b></span></div>
      <div class="fb"><button class="up ${r.verdict === 1 ? 'on' : ''}" data-v="1">👍 a reproduire</button><button class="down ${r.verdict === -1 ? 'on' : ''}" data-v="-1">👎 a eviter</button></div>` : ''}
    </div>`; }).join('') : '<div class="empty">Aucun exemple pour l\'instant.</div>';
  clearTimeout(exTimer);
  if (rows.some(r => !['ok', 'erreur'].includes(r.status)) && view === 'ex') exTimer = setTimeout(loadExamples, 3000);
}
document.addEventListener('click', async e => {
  const c = e.target.closest('#exGrid .card'); if (!c) return;
  const v = e.target.closest('button')?.dataset.v;
  if (v) { const on = e.target.closest('button').classList.contains('on'); await sendFb(c.dataset.m, on ? 0 : +v, '', ''); loadExamples(); return; }
  if (c.classList.contains('buy')) openToken(c.dataset.m);
});
setInterval(() => { if (view === 'ai' && !document.querySelector('#v-ai details[open].glossary:hover')) loadAI(); }, 60000);
