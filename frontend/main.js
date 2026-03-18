// ─── State ────────────────────────────────────────────────────────────────
const S = {
  jobs:          [],
  pipeline:      null,
  stats:         { total_docs:0, processed:0, failed:0, active:0, queued:0,
                   total_nodes:0, total_relations:0, queue_paused:false, current_job:null },
  uploadQueue:   [],   // File objects
  activeJobId:   null,
  sseSource:     null,
  sseJobId:      null,
  stages:        {},
  blockTypes:    {},
  entityProg:    { current:0, total:0 },
  multimodalProg:{ current:0, total:0 },
  logLines:      [],   // all lines for current SSE job
  logFilter:     '',
  autoScroll:    true,
  logJobId:      null, // job shown in log tab
  logJobLines:   [],   // log lines for log-tab job
  uploads:       [],   // /uploads file list
  uploadJobIds:  new Set(), // job ids we created this session
};

const STAGE_NAMES = [
  'Layout Predict','MFD Predict','MFR Predict',
  'Table-ocr det','Table-ocr rec ch','Table-wireless Predict',
  'Table-wired Predict','OCR-det Predict','Processing pages','OCR-rec Predict'
];
const STAGE_RE = /(\d+)\s*\/\s*(\d+)/;

// ─── Tab navigation ────────────────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'documents') refreshDocuments();
  });
});

// ─── API helpers ───────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ─── Bootstrap ─────────────────────────────────────────────────────────────
async function init() {
  try {
    const h = await api('GET', '/health');
    document.getElementById('status-dot').classList.add('online');
    document.getElementById('model-tag').textContent = h.llm_model || '—';
  } catch {
    document.getElementById('status-dot').classList.remove('online');
  }
  await refreshAll();
  setInterval(refreshAll, 4000);
}

async function refreshAll() {
  try {
    const [jobs, stats] = await Promise.all([
      api('GET', '/jobs'),
      api('GET', '/stats'),
    ]);
    S.jobs  = jobs;
    S.stats = stats;
    updateDashboard();
    updateDocTabStats();
    updateLogJobSelect();
    updateQueueBadge();
  } catch {}
}

// ─── Dashboard ─────────────────────────────────────────────────────────────
function updateDashboard() {
  const s = S.stats;
  document.getElementById('stat-processed').textContent  = s.processed;
  document.getElementById('stat-nodes').textContent      = s.total_nodes.toLocaleString();
  document.getElementById('stat-relations').textContent  = s.total_relations.toLocaleString();
  document.getElementById('stat-queued').textContent     = s.queued + (s.active ? ' (+1 active)' : '');

  // Paused banner
  const pb = document.getElementById('paused-banner');
  if (s.queue_paused) { pb.classList.add('visible'); }
  else                { pb.classList.remove('visible'); }

  // Pause button label
  const pauseBtn = document.getElementById('pause-btn');
  if (pauseBtn) {
    pauseBtn.textContent = s.queue_paused ? 'Resume Queue' : 'Pause Queue';
    pauseBtn.className   = s.queue_paused ? 'btn btn-secondary btn-sm' : 'btn btn-warn btn-sm';
  }

  // Active job
  const cur = s.current_job;
  if (cur) {
    document.getElementById('idle-state').style.display     = 'none';
    document.getElementById('active-job-panel').style.display = '';
    document.getElementById('active-filename').textContent  = cur.filename;
    const elapsed = cur.started_at ? fmtDuration((Date.now()/1000) - cur.started_at) : '—';
    const chunkEta = getETA(cur);
    document.getElementById('active-meta').textContent =
      `Running for ${elapsed}${chunkEta ? ' · ETA ' + chunkEta : ''}`;
    document.getElementById('active-status-badge').textContent = cur.status;
    document.getElementById('active-status-badge').className =
      'status-badge badge-' + cur.status;

    // Connect SSE if not already connected to this job
    if (S.sseJobId !== cur.id) connectSSE(cur.id);
  } else {
    document.getElementById('idle-state').style.display     = '';
    document.getElementById('active-job-panel').style.display = 'none';
    if (S.sseSource) { S.sseSource.close(); S.sseSource = null; S.sseJobId = null; }
  }
}

function getETA(job) {
  if (!job.chunks || !job.started_at) return null;
  // rough: if we've processed N chunks in T seconds, total ≈ T * (total_known/N)
  const elapsed = (Date.now()/1000) - job.started_at;
  const rate = job.chunks / elapsed; // chunks per second
  if (rate <= 0) return null;
  // Can't know total upfront, so just show rate
  return null; // disabled until we can infer total
}

// ─── SSE connection ─────────────────────────────────────────────────────────
function connectSSE(jobId) {
  if (S.sseSource) S.sseSource.close();
  S.sseJobId = jobId;
  S.stages = {}; S.blockTypes = {};
  S.entityProg = {current:0,total:0};
  S.multimodalProg = {current:0,total:0};
  S.logLines = [];
  renderStages();

  S.sseSource = new EventSource(`/progress/${jobId}`);
  S.sseSource.onmessage = e => {
    try {
      const ev = JSON.parse(e.data);
      processEvent(ev);
    } catch {}
  };
  S.sseSource.onerror = () => {};
}

function processEvent(ev) {
  const msg = ev.message || '';

  // ── Stage detection ───────────────────────────────────────────────────
  for (const stage of STAGE_NAMES) {
    if (msg.includes(stage)) {
      const m = STAGE_RE.exec(msg);
      if (m) {
        const cur = parseInt(m[1]), tot = parseInt(m[2]);
        S.stages[stage] = { current:cur, total:tot, pct: tot>0 ? Math.round(100*cur/tot) : 0 };
      } else if (msg.toLowerCase().includes('100%') || msg.includes(' done') || msg.includes('complete')) {
        S.stages[stage] = { current:1, total:1, pct:100 };
      }
    }
  }

  // ── Entity extraction ─────────────────────────────────────────────────
  if (ev.kind === 'chunk') {
    const m = /Chunk\s+(\d+)\s+of\s+(\d+)/i.exec(msg);
    if (m) {
      S.entityProg = { current: parseInt(m[1]), total: parseInt(m[2]) };
      updateEntityBar();
    }
  }

  // ── Block types ────────────────────────────────────────────────────────
  if (ev.kind === 'block_type') {
    S.blockTypes[ev.btype] = ev.count;
    updateBlockTypes();
  }

  // ── Multimodal progress ────────────────────────────────────────────────
  if (ev.kind === 'multimodal_progress') {
    S.multimodalProg = { current: ev.current, total: ev.total };
    updateMultimodalBar();
  }

  // ── Log line ───────────────────────────────────────────────────────────
  const ts = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString('en',{hour12:false}) : '';
  S.logLines.push({ ts, kind: ev.kind, msg });

  // If this job is also the one shown in Log tab, append there too
  if (S.logJobId === S.sseJobId) {
    S.logJobLines = S.logLines;
    appendLogLine({ ts, kind: ev.kind, msg });
  }

  renderStages();
}

// ─── Stage rendering ──────────────────────────────────────────────────────
function renderStages() {
  const grid = document.getElementById('stages-grid');
  grid.innerHTML = STAGE_NAMES.map(name => {
    const st = S.stages[name];
    const pct    = st ? st.pct  : 0;
    const sub    = st ? `${st.current} / ${st.total}` : '—';
    const active = st && pct > 0 && pct < 100;
    const done   = st && pct === 100;
    return `
      <div class="stage-box ${active?'active':''} ${done?'done':''}">
        <div class="stage-header">
          <span class="stage-name">${name}</span>
          <span class="stage-pct ${active?'active-pct':''}">${done?'100%': pct>0?pct+'%':'—'}</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill ${active?'orange':''}" style="width:${pct}%"></div>
        </div>
        <div class="stage-sub">${sub}</div>
      </div>`;
  }).join('');
}

function updateEntityBar() {
  const p = S.entityProg;
  if (!p.total) { document.getElementById('entity-bar').style.display = 'none'; return; }
  document.getElementById('entity-bar').style.display = '';
  const pct = Math.round(100 * p.current / p.total);
  document.getElementById('entity-pct').textContent  = pct + '%';
  document.getElementById('entity-fill').style.width = pct + '%';
  document.getElementById('entity-sub').textContent  = `Chunk ${p.current} of ${p.total} — entity & relation extraction`;
}

function updateMultimodalBar() {
  const p = S.multimodalProg;
  if (!p.total) { document.getElementById('multimodal-bar').style.display = 'none'; return; }
  document.getElementById('multimodal-bar').style.display = '';
  const pct = Math.round(100 * p.current / p.total);
  document.getElementById('multimodal-pct').textContent  = pct + '%';
  document.getElementById('multimodal-fill').style.width = pct + '%';
  document.getElementById('multimodal-sub').textContent  = `${p.current} / ${p.total} multimodal items processed`;
}

function updateBlockTypes() {
  const bt = S.blockTypes;
  if (!Object.keys(bt).length) { document.getElementById('block-types-panel').style.display = 'none'; return; }
  document.getElementById('block-types-panel').style.display = '';
  const total = Object.values(bt).reduce((a,b)=>a+b,0) || 1;
  const order = ['text','image','table','equation','discarded'];
  const extra = Object.keys(bt).filter(k => !order.includes(k));
  const keys  = [...order.filter(k=>bt[k]), ...extra.filter(k=>bt[k])];
  document.getElementById('block-types-bars').innerHTML = keys.map(k => `
    <div class="block-type-row bt-${k}">
      <span class="bt-label">${k}</span>
      <div class="bt-bar-wrap"><div class="bt-bar" style="width:${Math.max(1,Math.round(100*bt[k]/total))}%"></div></div>
      <span class="bt-count">${bt[k].toLocaleString()}</span>
    </div>`).join('');
}

// ─── Upload tab ──────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  addFilesToQueue([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => {
  addFilesToQueue([...fileInput.files]);
  fileInput.value = '';
});

function addFilesToQueue(files) {
  files.forEach(f => {
    if (!S.uploadQueue.find(q => q.name === f.name && q.size === f.size)) {
      S.uploadQueue.push(f);
    }
  });
  renderUploadQueue();
}

function renderUploadQueue() {
  const list  = document.getElementById('upload-queue-list');
  const count = document.getElementById('queue-count');
  count.textContent = `${S.uploadQueue.length} file${S.uploadQueue.length!==1?'s':''} in queue`;
  if (!S.uploadQueue.length) {
    list.innerHTML = '<div class="queue-empty">No files in queue — drop some above</div>';
    return;
  }
  list.innerHTML = S.uploadQueue.map((f,i) => `
    <div class="queue-item" id="qi-${i}">
      <span class="queue-item-icon">📄</span>
      <div class="queue-item-info">
        <div class="queue-item-name">${escHtml(f.name)}</div>
        <div class="queue-item-size">${fmtSize(f.size)}</div>
      </div>
      <button class="queue-item-remove" onclick="removeFromQueue(${i})">✕</button>
    </div>`).join('');
}

function removeFromQueue(idx) { S.uploadQueue.splice(idx,1); renderUploadQueue(); }
function clearUploadQueue()   { S.uploadQueue = []; renderUploadQueue(); }

async function processQueue() {
  if (!S.uploadQueue.length) return;
  const btn = document.getElementById('process-btn');
  btn.disabled = true; btn.textContent = 'Uploading…';
  const files = [...S.uploadQueue];
  S.uploadQueue = [];
  renderUploadQueue();

  for (const file of files) {
    try {
      const fd = new FormData(); fd.append('file', file);
      const r  = await fetch('/upload', { method:'POST', body:fd });
      if (r.ok) {
        const j = await r.json();
        S.uploadJobIds.add(j.job_id);
      }
    } catch (e) {
      console.error('Upload failed:', file.name, e);
    }
  }
  btn.disabled = false; btn.textContent = 'Process Queue ▶';
  // Switch to dashboard
  document.querySelector('[data-tab="dashboard"]').click();
  await refreshAll();
}

function updateQueueBadge() {
  const badge = document.getElementById('nav-upload').querySelector('.badge');
  if (S.uploadQueue.length > 0) {
    if (!badge) {
      const b = document.createElement('span');
      b.className = 'badge';
      document.getElementById('nav-upload').appendChild(b);
    }
    document.getElementById('nav-upload').querySelector('.badge').textContent = S.uploadQueue.length;
  } else {
    if (badge) badge.remove();
  }
}

// ─── Documents tab ────────────────────────────────────────────────────────
async function refreshDocuments() {
  updateDocTabStats();
  await renderDocTable();
  await refreshFailedUploads();
}

function updateDocTabStats() {
  const s = S.stats;
  document.getElementById('kb-docs').textContent      = s.processed;
  document.getElementById('kb-nodes').textContent     = s.total_nodes.toLocaleString();
  document.getElementById('kb-relations').textContent = s.total_relations.toLocaleString();
}

async function renderDocTable() {
  const tbody = document.getElementById('docs-tbody');

  // Use in-memory jobs — reliable across restarts since we have completed_docs.json
  // and active jobs are always in memory
  const allJobs = [...S.jobs];

  // Also merge in completed docs from persistent log that may not be in current session
  try {
    const completed = await api('GET', '/processed-filenames');
    const inMemoryNames = new Set(S.jobs.map(j => j.filename));
    completed.forEach(fname => {
      if (!inMemoryNames.has(fname)) {
        allJobs.push({ filename: fname, status: 'done', chunks: 0, nodes: 0, relations: 0,
                       started_at: 0, finished_at: 0, error: '' });
      }
    });
  } catch {}

  if (!allJobs.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No jobs recorded yet</td></tr>';
    return;
  }

  // Sort: active first, then done, then error
  const order = { parsing:0, processing:1, queued:2, done:3, error:4 };
  allJobs.sort((a,b) => (order[a.status]??9) - (order[b.status]??9));

  tbody.innerHTML = allJobs.map(j => {
    const dur = j.started_at && j.finished_at
      ? fmtDuration(j.finished_at - j.started_at)
      : j.started_at ? fmtDuration((Date.now()/1000) - j.started_at) + ' (running…)' : '—';
    return `
      <tr>
        <td>
          <div class="doc-filename" title="${escHtml(j.filename)}">${escHtml(j.filename)}</div>
          ${j.error ? `<div class="doc-error-msg">${escHtml(j.error.slice(0,80))}</div>` : ''}
        </td>
        <td><span class="status-badge badge-${j.status}">${j.status}</span></td>
        <td>${j.chunks || '—'}</td>
        <td>${j.nodes  || '—'}</td>
        <td class="doc-duration">${dur}</td>
      </tr>`;
  }).join('');
}

async function refreshFailedUploads() {
  try {
    // Use LightRAG as primary source of truth; fall back to our persistent log
    const [uploads, processedList] = await Promise.all([
      api('GET', '/uploads'),
      api('GET', '/processed-filenames').catch(() => []),
    ]);
    S.uploads = uploads;
    const processedNames = new Set(processedList);
    // Also include any in-memory jobs regardless of status — active jobs are not orphans
    S.jobs.forEach(j => processedNames.add(j.filename));
    const failed = S.uploads.filter(u => !processedNames.has(u.filename));
    const sec = document.getElementById('failed-section');
    if (!failed.length) { sec.style.display = 'none'; return; }
    sec.style.display = '';
    document.getElementById('failed-title').textContent =
      `Orphaned / Failed Uploads (${failed.length})`;
    document.getElementById('failed-list').innerHTML = failed.map(f => `
      <div class="failed-item">
        <span class="failed-name">${escHtml(f.filename)}</span>
        <span class="failed-size">${fmtSize(f.size)}</span>
        <button class="btn btn-danger btn-sm" onclick="deleteUpload('${escHtml(f.filename)}')">Delete</button>
      </div>`).join('');
  } catch {}
}

async function deleteUpload(filename) {
  try {
    await api('DELETE', '/uploads/' + encodeURIComponent(filename));
    await refreshDocuments();
  } catch (e) { alert('Delete failed: ' + e.message); }
}

async function clearAllFailed() {
  const processedList  = await api('GET', '/processed-filenames').catch(() => []);
  const processedNames = new Set(processedList);
  // Exclude all jobs from any session — active, done, or error
  S.jobs.forEach(j => processedNames.add(j.filename));
  const failed = S.uploads.filter(u => !processedNames.has(u.filename));
  for (const f of failed) {
    try { await api('DELETE', '/uploads/' + encodeURIComponent(f.filename)); } catch {}
  }
  await refreshDocuments();
}

// ─── Live Log tab ─────────────────────────────────────────────────────────
function updateLogJobSelect() {
  const sel = document.getElementById('log-job-select');
  const cur = sel.value;
  sel.innerHTML = '<option value="">— Select a job —</option>' +
    S.jobs.map(j => `<option value="${j.id}">${escHtml(j.filename.slice(0,50))} [${j.status}]</option>`).join('');
  if (cur) sel.value = cur;
}

async function selectLogJob(jobId) {
  S.logJobId    = jobId;
  S.logJobLines = [];
  clearDisplayedLogs();
  if (!jobId) return;

  // If this is the active SSE job, just show the buffered lines
  if (jobId === S.sseJobId) {
    S.logJobLines = [...S.logLines];
    renderFullLog();
    return;
  }

  // Otherwise fetch all events at once via SSE or just poll the job
  // We replay from /progress with a short stream read
  try {
    const job = S.jobs.find(j => j.id === jobId);
    if (!job) return;
    // For completed jobs, fetch progress events up to completion
    const src = new EventSource(`/progress/${jobId}`);
    src.onmessage = e => {
      try {
        const ev = JSON.parse(e.data);
        const ts = ev.ts ? new Date(ev.ts*1000).toLocaleTimeString('en',{hour12:false}) : '';
        S.logJobLines.push({ ts, kind:ev.kind, msg:ev.message||'' });
        appendLogLine({ ts, kind:ev.kind, msg:ev.message||'' });
      } catch {}
    };
    src.onerror = () => src.close();
    // Close after job is done or 5s
    if (job.status === 'done' || job.status === 'error') {
      setTimeout(() => src.close(), 3000);
    }
  } catch {}
}

function renderFullLog() {
  const stream = document.getElementById('log-stream');
  stream.innerHTML = '';
  const filtered = filterLines(S.logJobLines);
  if (!filtered.length) {
    stream.innerHTML = '<div class="empty-state">No log lines</div>';
    return;
  }
  filtered.forEach(l => {
    stream.appendChild(buildLogLine(l));
  });
  if (S.autoScroll) stream.scrollTop = stream.scrollHeight;
}

function appendLogLine(line) {
  if (S.logJobId !== S.sseJobId && S.logJobId !== null) return;
  if (S.logJobId === null) return;
  const stream = document.getElementById('log-stream');
  if (stream.querySelector('.empty-state')) stream.innerHTML = '';
  if (S.logFilter && !line.msg.toLowerCase().includes(S.logFilter)) return;
  stream.appendChild(buildLogLine(line));
  if (S.autoScroll) stream.scrollTop = stream.scrollHeight;
}

function buildLogLine(line) {
  const div = document.createElement('div');
  div.className = `log-line log-${line.kind}`;
  div.innerHTML = `<span class="log-ts">${line.ts}</span><span class="log-msg">${escHtml(line.msg)}</span>`;
  return div;
}

function filterLines(lines) {
  if (!S.logFilter) return lines;
  return lines.filter(l => l.msg.toLowerCase().includes(S.logFilter));
}

function filterLogs() {
  S.logFilter = document.getElementById('log-search').value.toLowerCase();
  if (S.logJobId) renderFullLog();
}

function clearDisplayedLogs() {
  document.getElementById('log-stream').innerHTML =
    '<div class="empty-state">Select a job above to view its log</div>';
}

function toggleAutoScroll() {
  S.autoScroll = !S.autoScroll;
  const btn = document.getElementById('autoscroll-btn');
  btn.textContent = 'Auto-scroll';
  btn.className   = S.autoScroll ? 'toggle-btn on' : 'toggle-btn';
}

// ─── Query ─────────────────────────────────────────────────────────────────
function handleQueryKey(e) {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); submitQuery(); }
}

function autoResize(el) {
  el.style.height = ''; el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

async function submitQuery() {
  const q    = document.getElementById('query-input').value.trim();
  const mode = document.getElementById('mode-select').value;
  if (!q) return;

  const ansDiv  = document.getElementById('query-answer');
  const ansText = document.getElementById('answer-text');
  const askBtn  = document.getElementById('ask-btn');

  ansDiv.classList.add('visible');
  ansText.innerHTML = '<span class="thinking-indicator">Thinking…</span>';
  askBtn.disabled   = true;
  askBtn.textContent = '…';

  try {
    const r = await api('POST', '/query', { question: q, mode, return_nodes: true });
    ansText.textContent = r.answer || '(no answer)';
  } catch (e) {
    ansText.innerHTML = `<span style="color:var(--red)">Error: ${escHtml(e.message)}</span>`;
  } finally {
    askBtn.disabled   = false;
    askBtn.textContent = 'Ask';
  }
}

function copyAnswer() {
  const text = document.getElementById('answer-text').textContent;
  navigator.clipboard.writeText(text).catch(() => {});
}

// ─── Queue pause / resume ─────────────────────────────────────────────────
async function toggleQueuePause() {
  const paused = S.stats.queue_paused;
  try {
    await api('POST', paused ? '/queue/resume' : '/queue/pause');
    await refreshAll();
  } catch (e) { alert(e.message); }
}

async function resumeQueue() {
  try { await api('POST', '/queue/resume'); await refreshAll(); }
  catch (e) { alert(e.message); }
}

// ─── Utilities ────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtSize(bytes) {
  if (bytes < 1024)      return bytes + ' B';
  if (bytes < 1048576)   return (bytes/1024).toFixed(1) + ' KB';
  return (bytes/1048576).toFixed(1) + ' MB';
}
function fmtDuration(secs) {
  secs = Math.floor(secs);
  if (secs < 60)   return secs + 's';
  if (secs < 3600) return Math.floor(secs/60) + 'm ' + (secs%60) + 's';
  return Math.floor(secs/3600) + 'h ' + Math.floor((secs%3600)/60) + 'm';
}

// ─── Graph visualization ──────────────────────────────────────────────────
const TYPE_COLORS = {
  concept:      '#39ff14',
  method:       '#ff8c00',
  equation:     '#c084fc',
  image:        '#00d4ff',
  data:         '#ffe066',
  artifact:     '#888888',
  content:      '#4a6fa5',
  location:     '#20b2aa',
  person:       '#ff69b4',
  event:        '#ff6b6b',
  organization: '#4ecdc4',
  condition:    '#95e1d3',
  type:         '#f8b500',
  issue:        '#e84393',
  naturalobject:'#a8e6cf',
  discarded:    '#222222',
  unknown:      '#333333',
};

function nodeColor(n) { return TYPE_COLORS[n.type] || '#666666'; }

// ── THREE loaded explicitly from CDN before force-graph ──────────────────
function getThree() { return window.THREE; }

let grapg = null;
let graphData = { nodes:[], links:[] };
let graphNodeLimit = 1000;
let graphLinkDistance = 120;
let graphNodeRepulsion = 500;
let currentNode = null;
let searchTimer = null;
let graphInited = false;
let autoRotating = true;
let pulseInterval = null;
let idleTimer = null;
let orbitInterval = null;
let isIdle = false;
let orbitAngle = 0;

document.querySelector('[data-tab="graph"]').addEventListener('click', () => {
  if (!graphInited) { graphInited = true; initGraph(); }
});

async function initGraph() {
  setGraphLoading('Querying Neo4j…');
  await loadGraph('');
}

function updateGraphLimit(newVal) {
  graphNodeLimit = parseInt(newVal);
  // Re-load the graph immediately when the slider is released
  const currentSearch = document.getElementById('graph-search').value.trim();
  loadGraph(currentSearch);
}

function updateGraphDistance(newVal) {
  graphLinkDistance = parseInt(newVal);
  
  if (graph) {
    // 1. Update the physics spring distance
    graph.d3Force('link').distance(graphLinkDistance);
    
    // 2. The Magic Trick: "Reheat" the simulation so the nodes instantly move to their new positions!
    graph.d3ReheatSimulation(); 
  }
}

function updateGraphRepulsion(newVal) {
  graphNodeRepulsion = parseInt(newVal);
  
  if (graph) {
    // 1. Update the physics charge strength (Notice we make it negative here!)
    graph.d3Force('charge').strength(-graphNodeRepulsion);
    
    // 2. "Reheat" the simulation so the nodes push apart or collapse together live!
    graph.d3ReheatSimulation(); 
  }
}

async function loadGraph(search) {
  setGraphLoading(search ? `Searching "${search}"…` : 'Loading graph…');
  try {
    const url = search ? `/graph?search=${encodeURIComponent(search)}` : `/graph?limit=${graphNodeLimit}`;
    const data = await api('GET', url);
    
    // --- NEW: Erase "Discarded Content" from the data ---
    const hiddenNodeId = "Discarded Content";
    
    // 1. Remove the node itself
    data.nodes = data.nodes.filter(n => n.id !== hiddenNodeId);
    
    // 2. Remove any links where this node was the source or target
    data.links = data.links.filter(l => 
      l.source !== hiddenNodeId && l.target !== hiddenNodeId
    );

    graphData = data;
    renderGraph(data);
  } catch(e) {
    document.getElementById('graph-loading-msg').textContent = 'Error: ' + e.message;
  }
}

function linkSrcColor(l, nodes) {
  const srcId   = l.source?.id || l.source;
  const srcNode = nodes.find(n => n.id === srcId);
  return srcNode ? nodeColor(srcNode) : '#ffffff';
}

function renderGraph(data) {
  const el = document.getElementById('3d-graph');
  el.innerHTML = '';

  const degreeMap = {};
  data.links.forEach(l => {
    degreeMap[l.source] = (degreeMap[l.source]||0) + 1;
    degreeMap[l.target] = (degreeMap[l.target]||0) + 1;
  });
  data.nodes.forEach(n => { n.degree = degreeMap[n.id] || 1; });
  const maxDeg = Math.max(...data.nodes.map(n => n.degree), 1);

  graph = ForceGraph3D({ antialias: true, alpha: false })(el)
    .backgroundColor('#060608')
    .showNavInfo(false)

    // --- NODES ---
    .nodeColor(n => TYPE_COLORS[n.type] || '#666666')
    .nodeVal(n => 0.2 + (n.degree / maxDeg) * 3) 
    .nodeOpacity(0.9)
    .nodeResolution(16)
    
    // --- RESTORED: Hover Labels ---
    .nodeLabel(n => `<div style="
      background:rgba(6,6,8,0.92);border:1px solid ${TYPE_COLORS[n.type] || '#666'}55;
      border-radius:3px;padding:6px 10px;
      font-family:JetBrains Mono,monospace;font-size:11px;
      color:${TYPE_COLORS[n.type] || '#666'};max-width:220px;line-height:1.5">
      <b>${n.id}</b><br>
      <span style="color:#555;font-size:10px">${n.type} · ${n.degree} links</span>
      ${n.desc ? '<br><span style="color:#777;font-size:10px">' + n.desc.slice(0,80) + '…</span>' : ''}
    </div>`)

    // --- LINKS ---
    .linkCurvature(0.15)
    .linkCurveRotation(0.5)
    .linkColor(() => '#ffffff10')
    .linkOpacity(1)

    // --- LANDMARK ---
    .nodeThreeObjectExtend(false) // Keeps the default spheres, but lets us add to them
    .nodeThreeObject(node => {
      const group = new THREE.Group();
      const color = TYPE_COLORS[node.type] || '#666666';
      
      // --- SIZED DOWN ---
      // Much smaller base size, and smaller growth per connection
      const size = 1.2 + (node.degree * 0.15); 

      // --- GLOSSY MATERIAL ---
      const geo = new THREE.SphereGeometry(size, 24, 24); // Slightly higher res for smooth reflections
      const mat = new THREE.MeshPhongMaterial({ 
        color: color,
        emissive: color,        // Gives it a very faint internal glow
        emissiveIntensity: 0.1, 
        shininess: 80          // The gloss factor!
      });
      const mesh = new THREE.Mesh(geo, mat);
      group.add(mesh);

      // --- FLOATING TEXT (Also scaled down slightly) ---
      if (node.degree > 50) {
        const sprite = new SpriteText(node.id);
        sprite.color = 'rgba(255, 255, 255, 0.6)';
        sprite.textHeight = 3 + (node.degree * 0.05); 
        sprite.position.y = size + 4; // Hovers just above the new smaller sphere
        group.add(sprite);
      }

      return group; 
    })

    // --- PARTICLES ---
    .linkDirectionalParticles(0) 
    .linkDirectionalParticleWidth(2.5)
    .linkDirectionalParticleSpeed(0.008) // Nice and smooth speed
    .linkDirectionalParticleColor(l => {
      // Find the source node of this specific link
      const srcId = l.source?.id || l.source;
      const srcNode = graphData.nodes.find(n => n.id === srcId);
      // Return the node's specific type color, or fallback to green
      return srcNode ? (TYPE_COLORS[srcNode.type] || '#39ff14') : '#39ff14';
    })

    // --- RESTORED: Interactions ---
    .onNodeClick(node => {
      showNodeInfo(node);
      const dist = 80;
      graph.cameraPosition(
        { x: node.x + dist, y: node.y + dist * 0.4, z: node.z + dist },
        node, 800
      );

      // --- THE SHOCKWAVE EFFECT ---
      const geometry = new THREE.SphereGeometry(1, 32, 32);
      const material = new THREE.MeshBasicMaterial({ 
        color: TYPE_COLORS[node.type] || '#ffffff', 
        transparent: true, 
        opacity: 0.5,
        wireframe: true // Makes it look techy
      });
      const wave = new THREE.Mesh(geometry, material);
      
      // Place it exactly on the clicked node
      wave.position.copy(node);
      graph.scene().add(wave);

      // Animate the expansion and fade out
      let scale = 1;
      let opacity = 0.5;
      const expand = setInterval(() => {
        scale += 2;
        opacity -= 0.02;
        wave.scale.set(scale, scale, scale);
        material.opacity = opacity;
        
        if (opacity <= 0) {
          clearInterval(expand);
          graph.scene().remove(wave); // Clean it up
          geometry.dispose();
          material.dispose();
        }
      }, 16);

      // --- LOCK-ON TARGETING RING ---
     if (window.activeRing) {
        graph.scene().remove(window.activeRing);
      }

      // Changed to a 3D Torus (radius: 8, tube thickness: 0.4)
      const ringGeo = new THREE.TorusGeometry(8, 0.4, 16, 64);
      const ringMat = new THREE.MeshBasicMaterial({ 
        color: TYPE_COLORS[node.type] || '#ffffff', 
        transparent: true,
        opacity: 0.8
      });
      
      window.activeRing = new THREE.Mesh(ringGeo, ringMat);
      
      // Lay it flat like a planetary ring instead of facing the camera
      window.activeRing.rotation.x = Math.PI / 2;
      
      graph.scene().add(window.activeRing); 
    })
    .onNodeHover(node => { el.style.cursor = node ? 'pointer' : 'default'; })
    .onBackgroundClick(() => restoreColors())
    .graphData(data);

  // --- WIDER SPACING ---
  graph.d3Force('charge').strength(-graphNodeRepulsion);
  graph.d3Force('link').distance(graphLinkDistance).strength(0.15);

  // --- ADD BLOOM GLOW EFFECT ---
  const bloomPass = new THREE.UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.1, 1.0, 0.1
  );
  bloomPass.strength = 1.1;  // How intensely it glows
  bloomPass.radius = 1.0;    // How wide the glow spreads
  bloomPass.threshold = 0.1; // Lower = more things glow
  graph.postProcessingComposer().addPass(bloomPass);

  // --- AMBIENT DATA DUST ---
  const dustGeometry = new THREE.BufferGeometry();
  const dustCount = 800;
  const positions = new Float32Array(dustCount * 3);

  for (let i = 0; i < dustCount * 3; i++) {
    // Spread the dust widely across the scene
    positions[i] = (Math.random() - 0.5) * 1500; 
  }

  dustGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const dustMaterial = new THREE.PointsMaterial({
    color: '#888888',
    size: 1.5,
    transparent: true,
    opacity: 0.4
  });

  const dustMesh = new THREE.Points(dustGeometry, dustMaterial);
  graph.scene().add(dustMesh);

  // --- NEW: THE RANDOM PULSE GENERATOR ---
  // Clear any existing timer first
  if (pulseInterval) clearInterval(pulseInterval);
  
  // Start a loop that runs every 800 milliseconds
  pulseInterval = setInterval(() => {
    if (!graph || !graphData || !graphData.links.length) return;
    
    // Pick 3 random links to fire a pulse down
    for (let i = 0; i < 3; i++) {
      const randomIndex = Math.floor(Math.random() * graphData.links.length);
      const randomLink = graphData.links[randomIndex];
      
      // Tell the graph to shoot a single particle down this specific link
      graph.emitParticle(randomLink);
    }
  }, 800);

  // --- NEW: SCREENSAVER IDLE TIMER ---
  const IDLE_TIME = 30000; // Change to 3000 to test it quickly!

  function resetIdleTimer() {
    if (!graph) return;
    
    // 1. User interacted! Turn off screensaver mode
    isIdle = false;
    
    // 2. Restart the countdown
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      if (graph) {
        isIdle = true; 
        
        // Capture the current camera angle so it doesn't snap/jump when it starts
        const camPos = graph.cameraPosition();
        orbitAngle = Math.atan2(camPos.x, camPos.z);
      }
    }, IDLE_TIME);
  }

  // Listen on the WINDOW so the canvas doesn't block the events
  window.addEventListener('mousemove', resetIdleTimer);
  window.addEventListener('mousedown', resetIdleTimer);
  window.addEventListener('wheel', resetIdleTimer);
  window.addEventListener('keydown', resetIdleTimer);

  // Start the first countdown
  resetIdleTimer();

  // The Orbit Loop: Manually nudge the camera (this prevents the renderer from sleeping)
  if (orbitInterval) clearInterval(orbitInterval);
  orbitInterval = setInterval(() => {
    if (isIdle && graph) {
      const camPos = graph.cameraPosition();
      
      // Find how far zoomed in we currently are
      const distance = Math.hypot(camPos.x, camPos.z);
      
      // Slowly increase the angle
      orbitAngle += 0.001; 
      
      // Manually orbit the camera
      graph.cameraPosition({
        x: distance * Math.sin(orbitAngle),
        z: distance * Math.cos(orbitAngle),
        y: camPos.y // Keep the current vertical height
      });
    }

    // --- SPIN AND TRACK THE TARGETING RING ---
    if (window.activeRing && currentNode) {
      // 1. Lock its position to the node, even if the node moves!
      window.activeRing.position.copy(currentNode);
      
      // 2. Spin it on the Z axis
      window.activeRing.rotation.z += 0.02; 
    }
  }, 16); // Runs at roughly 60 FPS
 
  updateLegend(data.nodes);
  updateGraphStats(data);
  hideGraphLoading();
}



function restoreColors() {
  if (!graph) return;
  currentNode = null;
  document.getElementById('node-info').style.display = 'none'; 
  
  graph.linkDirectionalParticles(0);
  
  // Reset all nodes back to solid colors
  graph.nodeColor(n => TYPE_COLORS[n.type] || '#666666');
  
  // NEW: Reset all lines back to the faint default web
  graph.linkColor(() => '#ffffff10'); 
}

function showNodeInfo(node) {
  if (!node) return;
  currentNode = node;
  const color = TYPE_COLORS[node.type] || '#666666';
  
  // UI Updates
  document.getElementById('node-info-type').textContent = node.type || 'unknown';
  document.getElementById('node-info-type').style.color = color;
  document.getElementById('node-info-name').textContent = node.id;
  document.getElementById('node-info-desc').textContent = node.desc || '—';
  document.getElementById('node-info-degree').textContent = `${node.degree} connections`;
  document.getElementById('node-info').style.display = ''; 

  if (!graph) return;
  const neighborIds = new Set([node.id]);
  graphData.links.forEach(l => {
    const s = l.source?.id || l.source;
    const t = l.target?.id || l.target;
    if (s === node.id) neighborIds.add(t);
    if (t === node.id) neighborIds.add(s);
  });

  // Dim non-neighbor nodes
  graph.nodeColor(n => {
    const baseColor = TYPE_COLORS[n.type] || '#666666';
    return neighborIds.has(n.id) ? baseColor : baseColor + '10';
  });

  // NEW: Make connected lines glow with the node's color, dim the rest
  graph.linkColor(l => {
    const s = l.source?.id || l.source;
    const t = l.target?.id || l.target;
    if (s === node.id || t === node.id) {
      return color + 'cc'; // Bright glow matching the clicked node
    }
    return '#ffffff05'; // Super faint background line
  });
}

function focusNode(node) {
  if (!node) return;
  // 1. Populate and show your new left-side panel
  showCurrentViewContext(node);
  
  // 2. Hide the right-side popup panel
  document.getElementById('node-info').style.display = 'none';
  
  // 3. Clean up the targeting ring before the new graph loads!
  if (window.activeRing && graph) {
    graph.scene().remove(window.activeRing);
    window.activeRing = null;
  }
  
  // 4. Load the new neighborhood graph
  loadGraph(node.id); 
}

function resetGraph() {
  document.getElementById('graph-search').value = '';
  document.getElementById('node-info').style.display = 'none';
  currentNode = null;
  loadGraph('');
}

function debounceSearch(val) {
  clearTimeout(searchTimer);
  if (!val.trim()) { loadGraph(''); return; }
  searchTimer = setTimeout(() => loadGraph(val.trim()), 600);
}

function updateLegend(nodes) {
  const typeCounts = {};
  nodes.forEach(n => { if (n.type) typeCounts[n.type] = (typeCounts[n.type]||0) + 1; });
  const types = Object.entries(typeCounts)
    .filter(([t]) => t !== 'discarded' && t !== 'unknown')
    .sort((a,b) => b[1]-a[1]);
  document.getElementById('legend-items').innerHTML = types.map(([t, count]) => {
    const color = TYPE_COLORS[t] || '#666';
    return `<div style="display:flex;align-items:center;gap:5px;font-size:10px;
                        font-family:'JetBrains Mono',monospace">
      <span style="width:8px;height:8px;border-radius:50%;background:${color};
                   flex-shrink:0;box-shadow:0 0 6px ${color}88"></span>
      <span style="color:${color}">${t}</span>
      <span style="color:#333">${count}</span>
    </div>`;
  }).join('');
}

function updateGraphStats(data) {
  document.getElementById('graph-stats').innerHTML =
    `<span style="color:#39ff14;font-weight:700">${data.nodes.length}</span>` +
    `<span style="color:#444"> nodes · </span>` +
    `<span style="color:#ff8c00;font-weight:700">${data.links.length}</span>` +
    `<span style="color:#444"> edges</span>`;
}

function setGraphLoading(msg) {
  document.getElementById('graph-loading').style.display = 'flex';
  document.getElementById('graph-loading-msg').textContent = msg;
}

function hideGraphLoading() {
  document.getElementById('graph-loading').style.display = 'none';
}


// --- RESPONSIVE GRAPH RESIZER ---
function resizeGraph() {
  if (!graph) return;
  
  const container = document.getElementById('tab-graph');
  
  // Only resize if the graph tab is currently active/visible!
  // This prevents the canvas from collapsing to 0x0 when hidden.
  if (container.classList.contains('active')) {
    graph.width(container.clientWidth);
    graph.height(container.clientHeight);
  }
}

// 1. Watch for the browser window changing size
window.addEventListener('resize', resizeGraph);

// 2. Watch for when you click back to the "Graph" tab. 
// If you resized the window while looking at the Dashboard, 
// this ensures the graph snaps to the correct size the second you return.
document.querySelector('[data-tab="graph"]').addEventListener('click', () => {
  // A tiny 50ms delay gives the browser time to apply the CSS 'display: block' 
  // before we try to measure the container's new width.
  setTimeout(resizeGraph, 50); 
});

// --- VIEW CONTEXT UI FUNCTIONS ---
function showCurrentViewContext(node) {
  if (!node) return;
  
  const panel = document.getElementById('current-view-panel');
  const nameEl = document.getElementById('current-view-name');
  const descEl = document.getElementById('current-view-desc');

  // Populate data and match the node's color
  nameEl.textContent = node.id;
  nameEl.style.color = TYPE_COLORS[node.type] || '#39ff14'; 
  descEl.textContent = node.desc || 'No description available.';
  
  // Show the panel
  panel.style.display = 'block';
}

function hideCurrentViewContext() {
  const panel = document.getElementById('current-view-panel');
  if (panel) panel.style.display = 'none';
}

// Triggers when you click "Back to Main Graph" inside the new panel
function resetToMainGraph() {
  hideCurrentViewContext();
  
  // Call your existing graph load function here. 
  // It's usually loadGraph() or loadGraph(null) to fetch the default 1000 nodes.
  loadGraph(); 
}

// ─── Start ────────────────────────────────────────────────────────────────
init();
