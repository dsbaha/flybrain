// flyworld 3D view: Three.js scene fed by SSE snapshots.

const ARENA = 50; // half-size
let scene, camera, renderer, raycaster;
let flies = {};   // name -> {group, trailLine, trailPts}
let foods = [];   // {mesh, plume}
let addFoodMode = false;
let plumeTex = null;

// --- minimal orbit control: drag rotate, wheel zoom, right-drag pan ---
const orbit = { theta: 0.7, phi: 1.02, r: 150, tx: 0, ty: 0, tz: 0 };
function applyOrbit() {
  const sp = Math.sin(orbit.phi), cp = Math.cos(orbit.phi);
  camera.position.set(
    orbit.tx + orbit.r * sp * Math.cos(orbit.theta),
    orbit.ty + orbit.r * cp,
    orbit.tz + orbit.r * sp * Math.sin(orbit.theta));
  camera.lookAt(orbit.tx, orbit.ty, orbit.tz);
}
function bindOrbit(canvas) {
  let drag = null;
  canvas.addEventListener('mousedown', e => {
    drag = { x: e.clientX, y: e.clientY, btn: e.button };
  });
  window.addEventListener('mouseup', () => drag = null);
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.btn === 2) { // pan
      const s = orbit.r / 600;
      const cx = Math.cos(orbit.theta), sx = Math.sin(orbit.theta);
      orbit.tx -= (dx * cx) * s; orbit.tz -= (dx * sx) * s;
      orbit.ty += dy * s;
    } else {
      orbit.theta -= dx * 0.005;
      orbit.phi = Math.min(1.5, Math.max(0.15, orbit.phi - dy * 0.005));
    }
    applyOrbit();
  });
  canvas.addEventListener('wheel', e => {
    orbit.r = Math.min(400, Math.max(30, orbit.r * (1 + e.deltaY * 0.001)));
    applyOrbit();
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
}

function makePlumeTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 4, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,170,60,0.55)');
  grad.addColorStop(0.5, 'rgba(255,150,50,0.18)');
  grad.addColorStop(1, 'rgba(255,140,40,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  return t;
}

// --- scenery: procedural trees + rocks (layout from /api/scene) ---
// Deterministic pseudo-random from coordinates: reloads must not
// reshuffle rock/tree orientation (a timelapse of fresh page loads
// would otherwise look like the rocks are spinning).
function hash2(x, z) {
  const h = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
  return h - Math.floor(h);
}
const trunkMat = new THREE.MeshStandardMaterial({ color: 0x6b4a2f, roughness: 0.95 });
const leafMats = [
  new THREE.MeshStandardMaterial({ color: 0x2d6a4f, roughness: 0.9 }),
  new THREE.MeshStandardMaterial({ color: 0x40916c, roughness: 0.9 }),
  new THREE.MeshStandardMaterial({ color: 0x1b4332, roughness: 0.9 }),
];
const rockMat = new THREE.MeshStandardMaterial({ color: 0x6b7280, roughness: 1, flatShading: true });

function makeTree(h) {
  const g = new THREE.Group();
  const trunkH = h * 0.35;
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.7, trunkH, 8), trunkMat);
  trunk.position.y = trunkH / 2;
  g.add(trunk);
  // stacked foliage cones
  let y = trunkH * 0.8, r = h * 0.32;
  for (let i = 0; i < 3; i++) {
    const ch = h * 0.30;
    const cone = new THREE.Mesh(new THREE.ConeGeometry(r, ch, 10), leafMats[i % 3]);
    cone.position.y = y + ch / 2;
    g.add(cone);
    y += ch * 0.55;
    r *= 0.72;
  }
  return g;
}

function makeRock(rk) {
  const m = new THREE.Mesh(new THREE.DodecahedronGeometry(rk.r, 0), rockMat);
  m.position.y = rk.r * 0.45;
  m.rotation.set(hash2(rk.x, rk.z) * 3, hash2(rk.z, rk.x) * 3, hash2(rk.x + 7, rk.z - 3) * 3);
  m.scale.y = 0.7;
  return m;
}

function buildScenery(trees, rocks) {
  for (const t of trees) {
    const m = makeTree(t.h);
    m.position.set(t.x, 0, t.z);
    m.rotation.y = hash2(t.x, t.z) * Math.PI * 2;
    scene.add(m);
  }
  for (const rk of rocks) {
    const m = makeRock(rk);
    m.position.x = rk.x; m.position.z = rk.z;
    scene.add(m);
  }
}

function makeFly(color) {
  const grp = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(color), roughness: 0.6 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(1.1, 16, 12), mat);
  body.scale.set(1.6, 0.9, 1.0);
  body.position.y = 1.2;
  grp.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.55, 12, 10), mat);
  head.position.set(1.7, 1.25, 0);
  grp.add(head);
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0x881111, roughness: 0.3 });
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.28, 10, 8), eyeMat);
    eye.position.set(1.95, 1.45, 0.32 * s);
    grp.add(eye);
  }
  const wingMat = new THREE.MeshBasicMaterial({ color: 0xcfe8ff, transparent: true, opacity: 0.35, side: THREE.DoubleSide });
  const wings = [];
  for (const s of [-1, 1]) {
    const wgeo = new THREE.PlaneGeometry(1.6, 0.8);
    const w = new THREE.Mesh(wgeo, wingMat);
    w.position.set(-0.3, 1.9, 0.55 * s);
    w.rotation.x = -Math.PI / 2 + 0.35 * s;
    grp.add(w);
    wings.push(w);
  }
  // heading nose
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.35, 1.2, 8),
    new THREE.MeshBasicMaterial({ color: 0xffffff }));
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(2.9, 1.25, 0);
  grp.add(nose);
  // 6 legs: 3 per side, tripod-gait phase offsets
  const legMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.8 });
  const legs = [];
  for (const s of [-1, 1]) {
    for (let i = 0; i < 3; i++) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.06, 2.4, 6), legMat);
      leg.position.set(0.7 - i * 0.75, 0.5, 0.95 * s);
      leg.rotation.z = 0.85 * s; // splay outward
      leg.userData.ph = i * 2.1 + (s > 0 ? Math.PI : 0); // tripod phase
      grp.add(leg);
      legs.push(leg);
    }
  }
  return { grp, wings, legs, phase: Math.random() * 6.28, walkAmt: 0 };
}

function init() {
  const canvas = document.getElementById('scene');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e14);
  scene.fog = new THREE.Fog(0x0b0e14, 220, 420);
  camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
  bindOrbit(canvas);
  applyOrbit();

  scene.add(new THREE.AmbientLight(0x8899bb, 0.7));
  const hemi = new THREE.HemisphereLight(0x9db8d6, 0x2a2118, 0.35);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff2dd, 1.0);
  sun.position.set(60, 100, 40);
  scene.add(sun);

  // floor + grid (forest-floor tones)
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(ARENA * 2, ARENA * 2),
    new THREE.MeshStandardMaterial({ color: 0x101a13, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  const grid = new THREE.GridHelper(ARENA * 2, 20, 0x2e4a38, 0x1a2b20);
  grid.position.y = 0.02;
  scene.add(grid);

  // arena walls (translucent)
  const wallMat = new THREE.MeshBasicMaterial({ color: 0x2a3a5e, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
  for (const [w, h, x, z, ry] of [
    [ARENA*2, 8, 0, -ARENA, 0], [ARENA*2, 8, 0, ARENA, 0],
    [ARENA*2, 8, -ARENA, 0, Math.PI/2], [ARENA*2, 8, ARENA, 0, Math.PI/2]]) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), wallMat);
    m.position.set(x, h/2, z); m.rotation.y = ry;
    scene.add(m);
  }

  plumeTex = makePlumeTexture();
  raycaster = new THREE.Raycaster();

  // scenery: trees + rocks from the server (deterministic layout)
  fetch('/api/scene').then(r => r.json()).then(s => {
    buildScenery(s.trees || [], s.rocks || []);
  }).catch(() => {});

  canvas.addEventListener('click', e => {
    if (!addFoodMode) return;
    const r = canvas.getBoundingClientRect();
    const nd = new THREE.Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(nd, camera);
    // intersect y=0 plane
    const t = -raycaster.ray.origin.y / raycaster.ray.direction.y;
    if (t > 0) {
      const p = raycaster.ray.origin.clone().add(raycaster.ray.direction.clone().multiplyScalar(t));
      fetch(`/api/addfood?x=${p.x.toFixed(1)}&z=${p.z.toFixed(1)}`);
    }
  });

  window.addEventListener('resize', resize);
  resize();
  requestAnimationFrame(tick3d);
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function tick3d(t) {
  requestAnimationFrame(tick3d);
  const s = t / 1000;
  for (const k in flies) {
    const f = flies[k];
    const flap = Math.sin(s * 40 + f.phase) * 0.5;
    f.wings[0].rotation.z = flap;
    f.wings[1].rotation.z = -flap;
    // tripod gait: legs swing while the fly moves (forward or reverse)
    for (const leg of f.legs) {
      leg.rotation.x = Math.sin(s * 26 + leg.userData.ph) * 0.45 * f.walkAmt;
    }
  }
  renderer.render(scene, camera);
}

function ensureFly(snap) {
  let f = flies[snap.name];
  if (!f) {
    const built = makeFly(snap.color);
    scene.add(built.grp);
    const tgeo = new THREE.BufferGeometry();
    const tpos = new Float32Array(200 * 3);
    tgeo.setAttribute('position', new THREE.BufferAttribute(tpos, 3));
    const tline = new THREE.Line(tgeo, new THREE.LineBasicMaterial({ color: new THREE.Color(snap.color), transparent: true, opacity: 0.7 }));
    tline.frustumCulled = false;
    scene.add(tline);
    f = { ...built, trailLine: tline, trailPts: [] };
    flies[snap.name] = f;
  }
  return f;
}

function updateFoods(list) {
  // remove extras
  while (foods.length > list.length) {
    const f = foods.pop();
    scene.remove(f.mesh); scene.remove(f.plume);
  }
  list.forEach((fd, i) => {
    let f = foods[i];
    if (!f) {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(2.2, 20, 16),
        new THREE.MeshStandardMaterial({ color: 0xff8c1a, emissive: 0x903c00, roughness: 0.5 }));
      mesh.position.y = 2.2;
      const plume = new THREE.Sprite(new THREE.SpriteMaterial({
        map: plumeTex, transparent: true, opacity: 0.8, depthWrite: false }));
      plume.scale.set(56, 56, 1);
      plume.position.y = 3;
      scene.add(mesh); scene.add(plume);
      f = { mesh, plume };
      foods[i] = f;
    }
    f.mesh.position.x = fd.x; f.mesh.position.z = fd.z;
    f.plume.position.x = fd.x; f.plume.position.z = fd.z;
  });
}

function onSnapshot(s) {
  document.getElementById('status').textContent =
    `tick ${s.tick} · ${s.flies.length} ${s.flies.length === 1 ? 'fly' : 'flies'} · ${s.foods.length} food`;
  updateFoods(s.foods);
  const hud = document.getElementById('flies');
  hud.innerHTML = '';
  for (const fs of s.flies) {
    const f = ensureFly(fs);
    f.grp.position.set(fs.x, 0, fs.z);
    f.grp.rotation.y = -fs.h;
    // leg swing follows actual movement (0 when still)
    const target = Math.min(1, Math.abs(fs.speed) / 4);
    f.walkAmt += (target - f.walkAmt) * 0.15;
    // trail
    const pts = fs.trail;
    const pos = f.trailLine.geometry.attributes.position;
    const n = Math.min(pts.length, 200);
    for (let i = 0; i < n; i++) {
      const p = pts[pts.length - n + i];
      pos.setXYZ(i, p[0], 0.15, p[1]);
    }
    pos.needsUpdate = true;
    f.trailLine.geometry.setDrawRange(0, n);
    // feeding bounce
    f.grp.position.y = fs.state === 'feeding' ? Math.abs(Math.sin(Date.now() / 90)) * 0.8 : 0;

    const el = document.createElement('div');
    el.className = 'fly';
    el.style.borderLeftColor = fs.color;
    el.innerHTML =
      `<span class="nm">${fs.name}</span><span class="st">${fs.state}</span>` +
      `<div class="bar"><div style="width:${(fs.battery * 100).toFixed(0)}%"></div></div>` +
      `<div class="meta">DNp09 ${fs.p9L.toFixed(0)}/${fs.p9R.toFixed(0)} Hz` +
      ` · DNb05 ${fs.dnb05.toFixed(1)} · PN ${fs.pn.toFixed(0)} · MDN ${fs.mdn.toFixed(0)}` +
      ` · ant ${fs.antL.toFixed(2)}/${fs.antR.toFixed(2)}` +
      ` · brain ${fs.brain.hz.toFixed(2)} Hz</div>`;
    hud.appendChild(el);
  }
  const pb = document.getElementById('btn-pause');
  pb.textContent = s.paused ? 'resume' : 'pause';
}

function initUI() {
  document.getElementById('btn-pause').onclick = () => fetch('/api/pause', { method: 'POST' });
  document.getElementById('btn-reset').onclick = () => fetch('/api/reset', { method: 'POST' });
  document.getElementById('btn-clear').onclick = () => fetch('/api/clearfoods', { method: 'POST' });
  const bf = document.getElementById('btn-food');
  bf.onclick = () => {
    addFoodMode = !addFoodMode;
    bf.textContent = addFoodMode ? '+ food: ON' : '+ food: off';
    bf.classList.toggle('on', addFoodMode);
  };
  const es = new EventSource('/events');
  es.onmessage = e => onSnapshot(JSON.parse(e.data));
  es.onerror = () => {
    document.getElementById('status').textContent = 'disconnected — retrying…';
  };
}

init();
initUI();
