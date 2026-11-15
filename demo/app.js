// Mock Canvas renderer plus the Study Assistant panel.
// Every page is rendered from /api/corpus, i.e. the same chunk files that went into the
// vector store, so clicking a citation lands on the exact page the answer came from.

const SECTION_ROUTE = {
  'Assignments': 'assignments',
  'Modules': 'modules',
  'Announcements': 'announcements',
  'Reading List': 'reading',
};
const ROUTE_LABEL = {
  home: 'Home', announcements: 'Announcements', assignments: 'Assignments',
  modules: 'Modules', reading: 'Reading List', outline: 'Subject Outline',
};
const CARD_COLOR = { '32555': '#0374B5', '32557': '#9E58B4', '32601': '#0B874B', '32606': '#C23B22' };
const TERM = 'Spring 2026 (City campus)';

let DATA = { subjects: [], top_k: 8 };
const INDEX = new Map();   // chunk_id -> { code, route, chunk }

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const subject = (code) => DATA.subjects.find((s) => s.code === code);

function routeOf(chunk) {
  return chunk.source_type === 'handbook' ? 'outline' : (SECTION_ROUTE[chunk.section] || 'home');
}

function navItems(s) {
  const routes = ['home', 'announcements', 'assignments', 'modules', 'reading'];
  const present = new Set(s.canvas.map(routeOf));
  return routes.filter((r) => r === 'home' || present.has(r)).concat(['outline']);
}

function fmtDue(due) {
  if (!due) return null;
  const [d, t] = due.split(' ');
  const date = new Date(`${d}T${t || '00:00'}`);
  if (isNaN(date)) return due;
  const mon = date.toLocaleString('en-AU', { month: 'short' });
  let h = date.getHours();
  const m = String(date.getMinutes()).padStart(2, '0');
  const ap = h >= 12 ? 'pm' : 'am';
  h = h % 12 || 12;
  return `${mon} ${date.getDate()} at ${h}:${m}${ap}`;
}

// ---------- routing ----------

function parseHash() {
  const [path, query] = (location.hash.replace(/^#\/?/, '') || '').split('?');
  const parts = path.split('/').filter(Boolean);
  const focus = new URLSearchParams(query || '').get('c');
  if (parts[0] === 'c' && parts[1]) return { view: 'course', code: parts[1], route: parts[2] || 'home', focus };
  return { view: 'dashboard' };
}

function go(code, route, focus) {
  const q = focus ? `?c=${encodeURIComponent(focus)}` : '';
  location.hash = `#/c/${code}/${route}${q}`;
}

function openChunk(chunkId) {
  const hit = INDEX.get(chunkId);
  if (hit) go(hit.code, hit.route, chunkId);
}

// ---------- courses tray ----------

function initTray() {
  const list = $('#traylist');
  for (const s of DATA.subjects) {
    const a = el('a', 'item');
    a.href = `#/c/${s.code}/home`;
    a.append(el('span', 't', `${s.code} ${s.name} - Spring 2026`));
    a.append(el('span', 'm', `${s.code} 2026 Spring All Students`));
    a.append(el('span', 'm', `Term: ${TERM}`));
    a.onclick = () => toggleTray(false);
    list.append(a);
  }
  $('#trayclose').onclick = () => toggleTray(false);
  $('#tray').querySelector('a[href="#/"]').onclick = () => toggleTray(false);
  addEventListener('keydown', (e) => { if (e.key === 'Escape') toggleTray(false); });
}

function toggleTray(force) {
  const tray = $('#tray');
  const open = force ?? tray.hidden;
  tray.hidden = !open;
  document.querySelector('#rail button[data-nav="courses"]').setAttribute('aria-current', String(open));
}

// ---------- views ----------

function render() {
  const st = parseHash();
  const dash = document.querySelector('#rail button[data-nav="dashboard"]');
  dash.setAttribute('aria-current', String(st.view === 'dashboard'));
  st.view === 'course' ? renderCourse(st) : renderDashboard();
}

function renderDashboard() {
  $('#coursenav').hidden = true;
  const c = $('#content');
  c.replaceChildren();
  c.append(el('h1', null, 'Dashboard'), el('div', 'rule'));

  const cards = el('div', 'cards');
  for (const s of DATA.subjects) {
    const card = el('div', 'card');
    const top = el('div', 'top');
    top.style.background = CARD_COLOR[s.code] || '#0374B5';
    const body = el('div', 'body');
    const t = el('a', 't', `${s.code} ${s.name}`);
    t.href = `#/c/${s.code}/home`;
    const next = s.canvas.filter((x) => x.due_date).sort((a, b) => a.due_date.localeCompare(b.due_date))[0];
    body.append(t, el('span', 's', next ? `Next due: ${next.title} — ${fmtDue(next.due_date)}` : 'Spring 2026'));
    card.append(top, body);
    cards.append(card);
  }
  c.append(cards, note());
}

function renderCourse(st) {
  const s = subject(st.code);
  if (!s) return renderDashboard();

  const nav = $('#coursenav');
  nav.hidden = false;
  nav.replaceChildren(el('div', 'code', s.code));
  for (const r of navItems(s)) {
    const a = el('a', st.route === r ? 'on' : null, ROUTE_LABEL[r]);
    a.href = `#/c/${s.code}/${r}`;
    nav.append(a);
  }

  const c = $('#content');
  c.replaceChildren();
  c.append(el('div', 'crumbs', `${s.code} / ${ROUTE_LABEL[st.route] || 'Home'}`));
  c.append(el('h1', null, st.route === 'home' ? `${s.code} ${s.name}` : ROUTE_LABEL[st.route]), el('div', 'rule'));

  if (st.route === 'home') renderHome(c, s);
  else if (st.route === 'outline') renderGroups(c, s.handbook, 'handbook');
  else renderGroups(c, s.canvas.filter((x) => routeOf(x) === st.route), 'canvas');

  c.append(note());
  if (st.focus) focusChunk(st.focus);
}

function renderHome(c, s) {
  const desc = s.handbook.find((x) => x.section === 'Subject description');
  if (desc) {
    c.append(el('h2', null, 'Subject description'));
    c.append(el('div', 'bodytext', desc.body));
  }
  const ann = s.canvas.filter((x) => x.page_type === 'announcement');
  if (ann.length) {
    c.append(el('h2', null, 'Recent announcements'));
    renderGroups(c, ann, 'canvas');
  }
}

// One chunk is one row; clicking it expands the body, like the list-to-detail flow in Canvas.
function renderGroups(c, chunks, kind) {
  if (!chunks.length) { c.append(el('p', 'empty', 'No content.')); return; }
  // In Handbook chunks section == title, so grouping by section would show the heading
  // twice. Keep them in one group instead.
  const bySection = new Map();
  for (const ch of chunks) {
    const key = kind === 'handbook' ? 'UTS Handbook 2026' : ch.section;
    if (!bySection.has(key)) bySection.set(key, []);
    bySection.get(key).push(ch);
  }
  for (const [section, list] of bySection) {
    const g = el('div', 'group');
    g.append(el('div', 'gh', section));
    for (const ch of list) {
      const row = el('div', 'row');
      row.id = `chunk-${ch.chunk_id}`;
      row.append(el('div', 'ri', kind === 'handbook' ? '▤' : ch.page_type === 'announcement' ? '⚑' : '✓'));
      const rm = el('div', 'rm');
      const title = el('button', 'rt', ch.title);
      const sub = el('div', 'rs');
      sub.append(pill(ch.source_type, ch.source_type), pill(ch.page_type));
      if (ch.week) sub.append(pill(`week ${ch.week}`));
      if (ch.due_date) sub.append(pill(`Due ${fmtDue(ch.due_date)}`, 'due'));
      sub.append(document.createTextNode(ch.chunk_id));
      const body = el('div', 'bodytext', ch.body);
      body.hidden = true;
      title.onclick = () => { body.hidden = !body.hidden; row.classList.toggle('open', !body.hidden); };
      rm.append(title, sub, body);
      row.append(rm);
      g.append(row);
    }
    c.append(g);
  }
}

// Turn **bold** markers in the answer into real bold; otherwise the ** shows up on screen.
function renderRich(text) {
  const box = el('div');
  text.split(/\*\*/).forEach((part, i) => {
    if (!part) return;
    box.append(i % 2 ? el('strong', null, part) : document.createTextNode(part));
  });
  return box;
}

function pill(text, cls) {
  return el('span', `pill ${cls || ''}`.trim(), text);
}

function note() {
  return el('div', 'synthnote',
    'Mock Canvas environment built for a research prototype. Canvas pages are synthetic data authored by the project; ' +
    'Subject Outline pages are derived from the public UTS Handbook. Not affiliated with Instructure or UTS.');
}

function focusChunk(chunkId) {
  const row = document.getElementById(`chunk-${chunkId}`);
  if (!row) return;
  row.querySelector('.rt')?.click();
  row.classList.add('hl');
  row.scrollIntoView({ block: 'center', behavior: 'smooth' });
  setTimeout(() => row.classList.remove('hl'), 4000);
}

// ---------- assistant panel ----------

function initPanel() {
  $('#go').onclick = ask;
  $('#q').onkeydown = (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask(); };
  $('#assistbtn').onclick = () => {
    const open = document.body.classList.toggle('panel-open');
    $('#assistbtn').setAttribute('aria-current', String(open));
  };
}

async function ask() {
  const question = $('#q').value.trim();
  if (!question) return;
  const out = $('#result');
  const btn = $('#go');
  btn.disabled = true;
  const spin = el('div', 'spin');
  spin.append(el('div', 'dots'), el('div', 'msg', 'Searching your Handbook and Canvas pages…'));
  spin.firstChild.append(el('i'), el('i'), el('i'));
  out.classList.add('loading');
  out.replaceChildren(spin);
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    data.error ? out.replaceChildren(el('div', 'err', data.error)) : renderAnswer(out, question, data);
  } catch (e) {
    out.replaceChildren(el('div', 'err', String(e)));
  } finally {
    out.classList.remove('loading');
    btn.disabled = false;
  }
}

function renderAnswer(out, question, data) {
  out.replaceChildren();
  out.append(el('div', 'sechead', 'Question'), el('div', 'qtext', question));

  const badge = data.abstained
    ? el('div', 'badge abstain', '⚠ Partly not covered by the course material')
    : el('div', 'badge grounded', '✓ Answered from the course material');
  out.append(el('div', 'sechead', 'Answer'), badge);
  const answer = el('div', `answer${data.abstained ? ' abstain' : ''}`);
  // The trailing "Sources:" list is for terminal output and duplicates the source cards
  // below, so it is hidden here.
  answer.append(renderRich(data.answer.replace(/\n+\s*\**Sources\**:[\s\S]*$/i, '')));
  out.append(answer);

  if (data.sub_queries?.length > 1) {
    out.append(el('div', 'sechead', 'Question split into'));
    for (const sq of data.sub_queries) out.append(el('div', 'empty', `• ${sq}`));
  }

  // Retrieval ranks (1, 2, 3...) follow the original order, before grouping by subject.
  const rank = new Map(data.sources.map((s, i) => [s.chunk_id, i + 1]));
  // Only the sources the model actually listed under "Sources:" get a card; chunks that
  // were retrieved but never cited are left out.
  const lines = (data.answer.match(/\n\s*\**Sources\**:([\s\S]*)$/i)?.[1] || '').toLowerCase().split('\n');
  const cited = (s) => !!s.source_url && lines.some((l) =>
    l.includes(s.source_url.toLowerCase()) && (s.source_type !== 'handbook' || l.includes(s.section.toLowerCase())));
  const shown = data.sources.filter(cited);
  const byCode = new Map();
  for (const s of shown) {
    if (!byCode.has(s.subject_code)) byCode.set(s.subject_code, []);
    byCode.get(s.subject_code).push(s);
  }

  out.append(el('div', 'sechead', 'Where this answer came from'));
  out.append(el('div', 'lead', shown.length
    ? 'Course material the assistant cited, most relevant first. Click one to open it.'
    : 'The assistant did not cite any course material for this answer.'));

  for (const [code, list] of byCode) {
    out.append(el('div', 'srcgroup', `${code} ${list[0].subject_name}`));
    for (const s of list) {
      // A Handbook citation opens the real UTS Handbook page in a new tab, not the mock site.
      const ext = s.source_type === 'handbook' && s.source_url;
      const linked = ext || INDEX.has(s.chunk_id);
      const card = el('div', `src${linked ? '' : ' nolink'}`);
      card.append(el('div', 'rank', String(rank.get(s.chunk_id))));
      const sb = el('div', 'sb');
      sb.append(el('div', 'st', s.title || s.section));
      const meta = el('div', 'sm');
      meta.append(pill(s.source_type, s.source_type), pill(s.page_type));
      meta.append(document.createTextNode(s.section));
      sb.append(meta);
      sb.append(el('div', 'cid', ext ? `${s.chunk_id} · opens uts.edu.au` : linked ? s.chunk_id : `${s.chunk_id} · Handbook only — not one of your courses`));
      card.append(sb);
      if (ext) card.onclick = () => window.open(s.source_url, '_blank', 'noopener');
      else if (linked) card.onclick = () => openChunk(s.chunk_id);
      out.append(card);
    }
  }
}

// ---------- boot ----------

(async function main() {
  DATA = await (await fetch('/api/corpus')).json();
  for (const s of DATA.subjects) {
    for (const ch of [...s.canvas, ...s.handbook]) {
      INDEX.set(ch.chunk_id, { code: s.code, route: routeOf(ch), chunk: ch });
    }
  }
  initTray();
  initPanel();
  for (const b of document.querySelectorAll('#rail button[data-nav]')) {
    const nav = b.dataset.nav;
    if (nav === 'dashboard') b.onclick = () => { toggleTray(false); location.hash = '#/'; render(); };
    if (nav === 'courses') b.onclick = () => toggleTray();
  }
  addEventListener('hashchange', render);
  render();
})();
