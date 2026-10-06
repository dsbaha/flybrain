// flyviz front end: renders the world canvas + telemetry from /events (SSE).
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const SCALE = 6.2, OX = 10, OY = 10; // world 100x100 -> canvas
const sensNames = ['antenna L', 'antenna R'];
const motorNames = ['motor pop L', 'motor pop R'];

let state = null;
let sensScale = 15;

function wx(x) { return OX + x * SCALE; }
function wy(y) { return OY + y * SCALE; }

function bar(container, id, label, cls) {
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.className = 'bar'; el.id = id;
    el.innerHTML = '<div class="fill ' + (cls || '') + '"></div><span></span><span class="lbl">' + label + '</span>';
    container.appendChild(el);
  }
  return el;
}
function setBar(id, frac, text) {
  const el = document.getElementById(id);
  el.querySelector('.fill').style.width = (Math.max(0, Math.min(1, frac)) * 100).toFixed(1) + '%';
  el.querySelector('span').textContent = text;
}

function draw() {
  if (!state) return;
  ctx.clearRect(0, 0, cv.width, cv.height);
  // grid
  ctx.strokeStyle = '#10151d'; ctx.lineWidth = 1;
  for (let i = 0; i <= 100; i += 10) {
    ctx.beginPath(); ctx.moveTo(wx(i), wy(0)); ctx.lineTo(wx(i), wy(100)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(wx(0), wy(i)); ctx.lineTo(wx(100), wy(i)); ctx.stroke();
  }
  const s = state;
  // trail
  if (s.trail && s.trail.length > 1) {
    ctx.strokeStyle = 'rgba(63,169,245,0.55)'; ctx.lineWidth = 2;
    ctx.beginPath();
    s.trail.forEach((p, i) => { i ? ctx.lineTo(wx(p[0]), wy(p[1])) : ctx.moveTo(wx(p[0]), wy(p[1])); });
    ctx.stroke();
  }
  // odor plume: soft hazy cloud with flicker
  const bx = wx(s.beacon.x), by = wy(s.beacon.y);
  const fl = 0.9 + 0.1 * Math.sin(s.tick * 0.7);
  for (const [r, a] of [[95, 0.10], [62, 0.15], [36, 0.20]]) {
    const g = ctx.createRadialGradient(bx, by, 2, bx, by, r);
    g.addColorStop(0, 'rgba(214,178,106,' + (a * fl).toFixed(3) + ')');
    g.addColorStop(1, 'rgba(214,178,106,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(bx, by, r, 0, 7); ctx.fill();
  }
  ctx.fillStyle = '#8a6f3c';
  ctx.beginPath(); ctx.arc(bx, by, 5, 0, 7); ctx.fill();
  // agent triangle
  const ax = wx(s.agent.x), ay = wy(s.agent.y), h = s.agent.h;
  ctx.save(); ctx.translate(ax, ay); ctx.rotate(h);
  ctx.fillStyle = s.battery > 0.25 ? '#8fd0ff' : '#ff7a7a';
  ctx.beginPath();
  ctx.moveTo(12, 0); ctx.lineTo(-8, -8); ctx.lineTo(-4, 0); ctx.lineTo(-8, 8);
  ctx.closePath(); ctx.fill();
  ctx.restore();
  // heading tick
  ctx.strokeStyle = '#8fd0ff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(ax, ay);
  ctx.lineTo(ax + Math.cos(h) * 22, ay + Math.sin(h) * 22); ctx.stroke();
}

function render() {
  if (!state) return;
  const s = state;
  draw();
  const sensBox = document.getElementById('sens');
  s.sens.forEach((v, i) => {
    bar(sensBox, 'sens' + i, sensNames[i] || ('ch' + i));
    setBar('sens' + i, v / sensScale, v.toFixed(2) + ' mV');
  });
  const motBox = document.getElementById('motor');
  s.motor.forEach((v, i) => {
    bar(motBox, 'mot' + i, motorNames[i] || ('m' + i), 'motor');
    setBar('mot' + i, v, v.toFixed(2) + '  (' + s.motorV[i].toFixed(1) + ' mV)');
  });
  setBar('battBar', s.battery, (s.battery * 100).toFixed(0) + '%');
  const st = document.getElementById('stats');
  const b = s.brain;
  st.innerHTML =
    '<dt>state</dt><dd>' + s.state + (s.paused ? ' (held)' : '') + '</dd>' +
    '<dt>tick</dt><dd>' + s.tick + '</dd>' +
    '<dt>brain steps</dt><dd>' + b.steps + '</dd>' +
    '<dt>sim time</dt><dd>' + (b.simMs / 1000).toFixed(1) + ' s</dd>' +
    '<dt>total spikes</dt><dd>' + b.spikes + '</dd>' +
    '<dt>mean rate</dt><dd>' + b.hz.toFixed(1) + ' Hz</dd>';
}

const es = new EventSource('/events');
es.onmessage = e => { state = JSON.parse(e.data); render(); };
es.onerror = () => {
  document.getElementById('subtitle').textContent = 'reconnecting…';
};
es.onopen = () => {
  document.getElementById('subtitle').textContent = 'live';
};

cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) * (cv.width / r.width);
  const y = (e.clientY - r.top) * (cv.height / r.height);
  fetch('/api/beacon', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x: (x - OX) / SCALE, y: (y - OY) / SCALE })
  });
});
document.getElementById('bPause').onclick = async () => {
  const r = await fetch('/api/pause', { method: 'POST' });
  const j = await r.json();
  document.getElementById('bPause').textContent = j.paused ? 'resume' : 'pause';
};
document.getElementById('bReset').onclick = () =>
  fetch('/api/reset', { method: 'POST' });
document.getElementById('bBeacon').onclick = () =>
  fetch('/api/beacon', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x: 5 + Math.random() * 90, y: 5 + Math.random() * 90 })
  });

fetch('/api/info').then(r => r.json()).then(j => {
  sensScale = j.sensScale || 15;
  document.getElementById('mapnote').textContent =
    'brain N=' + j.brainN + '. ' + j.note;
  document.getElementById('subtitle').textContent =
    'brain N=' + j.brainN + ' · live';
});
