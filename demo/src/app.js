(() => {
'use strict';
THREE.ColorManagement.enabled = false;
const D = window.DEMO;
const M = ['fixed_low', 'fixed_mid', 'fixed_high', 'rule', 'rl_qoe', 'rl_task'];
const NAME = {fixed_low: 'Fixed-Low', fixed_mid: 'Fixed-Mid', fixed_high: 'Fixed-High', rule: 'Rule-ABR',
  rl_qoe: 'DQN · QoE-only', rl_task: 'DQN · task-aware'};
const COL = {fixed_low: '#8d99ae', fixed_mid: '#4cc9f0', fixed_high: '#4361ee', rule: '#f9c74f',
  rl_qoe: '#5eff7a', rl_task: '#ff4d6d'};
const PNAME = {stable: 'Stable', congested: 'Congested', highjitter: 'High-jitter',
  intermittent: 'Intermittent*', square: 'Square-wave*', ramp: 'Ramp-down*'};
const $ = (id) => document.getElementById(id);
const st = {prof: 'congested', seed: D.seeds[0], t: 0, playing: true, speed: 1, op: 'rl_qoe',
  vis: Object.fromEntries(M.map(m => [m, true])), lab: false, labR: 1, labP: 1, labB: 2, labLat: 120,
  tour: -1, tourWait: 0, ep: null, N: 1};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sig = (x) => 1 / (1 + Math.exp(-clamp(x, -30, 30)));

// ------------------------------------------------------------------ data helpers
function at(arr, t) {
  const n = arr.length;
  if (t <= 0) return arr[0];
  if (t >= n - 1) return arr[n - 1];
  const i = Math.floor(t), f = t - i;
  return arr[i] * (1 - f) + arr[i + 1] * f;
}
function atAng(arr, t) {
  const n = arr.length;
  if (t <= 0) return arr[0];
  if (t >= n - 1) return arr[n - 1];
  const i = Math.floor(t), f = t - i;
  let d = arr[i + 1] - arr[i];
  d = Math.atan2(Math.sin(d), Math.cos(d));
  return arr[i] + d * f;
}
const decodeAct = (a) => [Math.floor(a / 9), Math.floor(a / 3) % 3, a % 3];
function sentRate(r, p, b) { const af = Math.min(1, Math.PI * r * r / 4); return b * (af + (1 - af) * p * p); }
function quality(r, p, b, e) {
  const qf = (1 - Math.exp(-b / 3)) / (1 - Math.exp(-4));
  const qp = qf * Math.pow(p, 0.6);
  const w = sig((r - e) / 0.04);
  return w * qf + (1 - w) * qp;
}

// ------------------------------------------------------------------ UI scaffolding
function mkBtn(parent, label, on, cb) {
  const b = document.createElement('button'); b.textContent = label; if (on) b.classList.add('on');
  b.onclick = cb; parent.appendChild(b); return b;
}
const profBtns = {}, seedBtns = {}, spdBtns = {}, opBtns = {};
D.profiles.forEach(p => profBtns[p] = mkBtn($('profbtns'), PNAME[p], p === st.prof, () => { tourStop(); setEpisode(p, st.seed); }));
D.seeds.forEach((s, i) => seedBtns[s] = mkBtn($('seedbtns'), String(i + 1), s === st.seed, () => { tourStop(); setEpisode(st.prof, s); }));
[0.5, 1, 2, 4].forEach(v => spdBtns[v] = mkBtn($('spd'), v + '×', v === 1, () => { st.speed = v; Object.entries(spdBtns).forEach(([k, b]) => b.classList.toggle('on', +k === v)); }));
M.forEach(m => { const b = mkBtn($('oprow'), NAME[m], m === st.op, () => selectOp(m)); b.style.borderColor = COL[m] + '66'; opBtns[m] = b; });
const labBtn = mkBtn($('oprow'), '🎮 Foveation lab', false, () => { st.lab = !st.lab; labBtn.classList.toggle('on', st.lab); $('lab').style.display = st.lab ? 'block' : 'none'; });
labBtn.style.marginLeft = 'auto';
function seg(id, labels, key) {
  const el = $(id);
  labels.forEach((l, i) => { const b = mkBtn(el, l, i === st[key], () => { st[key] = i; [...el.children].forEach((c, j) => c.classList.toggle('on', j === i)); }); });
}
seg('labr', ['0.15', '0.30', '0.50'], 'labR'); seg('labp', ['25%', '50%', '100%'], 'labP'); seg('labb', ['2 Mbps', '6 Mbps', '12 Mbps'], 'labB');
$('lablat').oninput = (e) => { st.labLat = +e.target.value; $('lablatv').textContent = st.labLat + ' ms'; };
$('play').onclick = () => { st.playing = !st.playing; $('play').textContent = st.playing ? '❚❚ Pause' : '▶ Play'; $('play').classList.toggle('on', st.playing); };
$('restart').onclick = () => { st.t = 0; resetFx(); };
$('scrub').oninput = (e) => { st.t = (+e.target.value / 1000) * (st.N - 1); resetFx(); };
function selectOp(m) { st.op = m; Object.entries(opBtns).forEach(([k, b]) => b.classList.toggle('on', k === m)); }

// legend chips (visibility toggles)
const chips = {};
M.forEach(m => {
  const c = document.createElement('span'); c.className = 'chip'; c.innerHTML = `<i style="background:${COL[m]};box-shadow:0 0 8px ${COL[m]}"></i>${NAME[m]}`;
  c.onclick = () => { st.vis[m] = !st.vis[m]; c.classList.toggle('off', !st.vis[m]); };
  $('legend').appendChild(c); chips[m] = c;
});
// scoreboard
const rows = {};
M.forEach(m => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><span class="dot" style="color:${COL[m]};background:${COL[m]}"></span>${NAME[m]}</td><td><span class="bar"><i></i></span> <span class="pg"></span></td><td class="co"></td><td class="ag"></td><td class="qq"></td><td class="ss"></td><td><span class="st run">driving</span></td>`;
  tr.onclick = () => selectOp(m);
  document.querySelector('#score tbody').appendChild(tr);
  rows[m] = {tr, bar: tr.querySelector('.bar i'), pg: tr.querySelector('.pg'), co: tr.querySelector('.co'),
    ag: tr.querySelector('.ag'), qq: tr.querySelector('.qq'), ss: tr.querySelector('.ss'), stt: tr.querySelector('.st')};
});

// ------------------------------------------------------------------ THREE: shared world
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050914);
scene.fog = new THREE.FogExp2(0x050914, 0.022);
scene.add(new THREE.HemisphereLight(0x9bb8ff, 0x1a1230, 0.9));
const sun = new THREE.DirectionalLight(0xbfd4ff, 0.9); sun.position.set(6, 14, 4); scene.add(sun);
const world = new THREE.Group(); scene.add(world);      // course + city (visible in both views)
const overlay = new THREE.Group(); scene.add(overlay);  // robots, trails, labels (arena view only)
const P3 = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

function groundTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 512; const g = c.getContext('2d');
  g.fillStyle = '#0e1a38'; g.fillRect(0, 0, 512, 512);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 1400; i++) { g.fillStyle = `rgba(${70 + rnd() * 80},${100 + rnd() * 90},${170 + rnd() * 85},${0.08 + rnd() * 0.2})`; g.fillRect(rnd() * 512, rnd() * 512, 2 + rnd() * 12, 2 + rnd() * 12); }
  g.strokeStyle = 'rgba(49,225,255,0.95)'; g.lineWidth = 3;
  for (let i = 0; i <= 512; i += 128) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 512); g.stroke(); g.beginPath(); g.moveTo(0, i); g.lineTo(512, i); g.stroke(); }
  g.strokeStyle = 'rgba(139,123,255,0.55)'; g.lineWidth = 1.5;
  for (let i = 0; i <= 512; i += 16) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 512); g.stroke(); g.beginPath(); g.moveTo(0, i); g.lineTo(512, i); g.stroke(); }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(30, 30); t.anisotropy = 8; return t;
}
const ground = new THREE.Mesh(new THREE.PlaneGeometry(160, 160), new THREE.MeshStandardMaterial({map: groundTexture(), roughness: 0.8, metalness: 0.05, emissive: 0x0b1a3a, emissiveIntensity: 0.9}));
ground.rotation.x = -Math.PI / 2; ground.position.set(3.5, -0.001, -6.5); world.add(ground);
// sky dome with a glowing horizon
world.add(new THREE.Mesh(new THREE.SphereGeometry(110, 32, 16), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  vertexShader: 'varying float vy; void main(){ vy=normalize(position).y; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }',
  fragmentShader: 'varying float vy; void main(){ float h=clamp(vy,0.,1.); vec3 hor=vec3(0.30,0.16,0.62); vec3 mid=vec3(0.05,0.10,0.28); vec3 top=vec3(0.01,0.02,0.06); vec3 c=mix(hor,mid,smoothstep(0.0,0.22,h)); c=mix(c,top,smoothstep(0.15,0.8,h)); if(vy<0.) c=hor*0.25; gl_FragColor=vec4(c,1.); }'
})));
// neon city ring (deterministic)
(() => {
  let s = 12345; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const mats = [0x31e1ff, 0x8b7bff, 0xff4d6d, 0x5eff7a];
  for (let i = 0; i < 110; i++) {
    const a = rnd() * Math.PI * 2, r = 19 + rnd() * 30, w = 0.8 + rnd() * 2.4, h = 0.6 + Math.pow(rnd(), 2) * 7, d = 0.8 + rnd() * 2.4;
    const cx = 3.5 + Math.cos(a) * r, cz = -6.5 + Math.sin(a) * r;
    const geo = new THREE.BoxGeometry(w, h, d);
    const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color: 0x0d1530, roughness: 0.6, metalness: 0.3}));
    m.position.set(cx, h / 2, cz); world.add(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({color: mats[i % 4], transparent: true, opacity: 0.8}));
    e.position.copy(m.position); world.add(e);
  }
})();
let courseGroup = null;
function textSprite(txt, color, size = 0.9) {
  const c = document.createElement('canvas'); c.width = 256; c.height = 64; const g = c.getContext('2d');
  g.font = '600 30px Segoe UI,Arial'; g.textAlign = 'center'; g.fillStyle = 'rgba(5,10,24,.75)';
  g.fillRect(8, 8, 240, 48); g.strokeStyle = color; g.lineWidth = 2; g.strokeRect(8, 8, 240, 48);
  g.fillStyle = color; g.fillText(txt, 128, 43);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false}));
  sp.scale.set(size * 2, size * 0.5, 1); return sp;
}
function buildCourse(ep) {
  if (courseGroup) { world.remove(courseGroup); courseGroup.traverse(o => { if (o.geometry) o.geometry.dispose(); }); }
  courseGroup = new THREE.Group(); world.add(courseGroup);
  // path
  const pts = [P3(0, 0, 0.03), ...ep.wps.map(w => P3(w[0], w[1], 0.03))];
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({color: 0x31e1ff, dashSize: 0.3, gapSize: 0.18, transparent: true, opacity: 0.75}));
  line.computeLineDistances(); courseGroup.add(line);
  // start pad
  const pad = new THREE.Mesh(new THREE.RingGeometry(0.4, 0.62, 40), new THREE.MeshBasicMaterial({color: 0x5eff7a, side: THREE.DoubleSide}));
  pad.rotation.x = -Math.PI / 2; pad.position.copy(P3(0, 0, 0.02)); courseGroup.add(pad);
  // waypoints
  ep.wps.forEach((w, i) => {
    const last = i === ep.wps.length - 1, col = last ? 0xffc857 : 0x31e1ff;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.045, 10, 40), new THREE.MeshBasicMaterial({color: col}));
    ring.rotation.x = Math.PI / 2; ring.position.copy(P3(w[0], w[1], 0.05)); courseGroup.add(ring);
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 4, 8, 1, true), new THREE.MeshBasicMaterial({color: col, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false}));
    beam.position.copy(P3(w[0], w[1], 2)); courseGroup.add(beam);
    const sp = textSprite(last ? 'FINISH' : String(i + 1), last ? '#ffc857' : '#31e1ff', 0.55); sp.position.copy(P3(w[0], w[1], 1.0)); courseGroup.add(sp);
  });
  // landmark crates beside the route (give the operator texture to perceive; never on the path)
  (() => {
    let s = 99; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    const segs = [[0, 0], ...ep.wps]; const cols = [0x31e1ff, 0x8b7bff, 0xffc857, 0x5eff7a];
    const dseg = (px, py, a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy; const t = Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / L2)); return Math.hypot(px - a[0] - t * dx, py - a[1] - t * dy); };
    let placed = 0, tries = 0;
    while (placed < 36 && tries++ < 600) {
      const px = -6 + rnd() * 20, py = -4 + rnd() * 21;
      let ok = true; for (let q = 0; q < segs.length - 1; q++) if (dseg(px, py, segs[q], segs[q + 1]) < 2.0) ok = false;
      for (const o of ep.obst) if (Math.hypot(px - o[0], py - o[1]) < 1.6) ok = false;
      if (!ok) continue;
      const w = 0.4 + rnd() * 0.7, h = 0.3 + rnd() * 1.4, geo = new THREE.BoxGeometry(w, h, w * (0.7 + rnd() * 0.6)), col = cols[placed % 4];
      const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color: 0x101a3a, emissive: col, emissiveIntensity: 0.12, roughness: 0.5}));
      m.position.copy(P3(px, py, h / 2)); m.rotation.y = rnd() * 3; courseGroup.add(m);
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({color: col})); e.position.copy(m.position); e.rotation.copy(m.rotation); courseGroup.add(e);
      placed++;
    }
  })();
  // obstacles
  ep.obst.forEach(o => {
    const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.95, 24), new THREE.MeshStandardMaterial({color: 0x7a1020, emissive: 0xff2a4a, emissiveIntensity: 0.55, roughness: 0.4}));
    cyl.position.copy(P3(o[0], o[1], 0.475)); courseGroup.add(cyl);
    const rg = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.8, 32), new THREE.MeshBasicMaterial({color: 0xff4d6d, transparent: true, opacity: 0.35, side: THREE.DoubleSide}));
    rg.rotation.x = -Math.PI / 2; rg.position.copy(P3(o[0], o[1], 0.02)); courseGroup.add(rg);
    const top = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.52, 0.08, 24), new THREE.MeshBasicMaterial({color: 0xff4d6d}));
    top.position.copy(P3(o[0], o[1], 0.95)); courseGroup.add(top);
  });
}
// robots
const robots = {};
function makeRobot(col) {
  const g = new THREE.Group(), c = new THREE.Color(col);
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.2, 0.42), new THREE.MeshStandardMaterial({color: c, emissive: c, emissiveIntensity: 0.35, metalness: 0.4, roughness: 0.35}));
  body.position.y = 0.2; g.add(body);
  const top = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.12, 0.26), new THREE.MeshStandardMaterial({color: 0x0a0f20, metalness: 0.6, roughness: 0.3}));
  top.position.set(-0.04, 0.36, 0); g.add(top);
  const eye = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12), new THREE.MeshBasicMaterial({color: 0xffffff}));
  eye.position.set(0.2, 0.36, 0); g.add(eye);
  const wg = new THREE.CylinderGeometry(0.11, 0.11, 0.07, 14); wg.rotateX(Math.PI / 2);
  [[0.2, 0.22], [0.2, -0.22], [-0.2, 0.22], [-0.2, -0.22]].forEach(p => { const w = new THREE.Mesh(wg, new THREE.MeshStandardMaterial({color: 0x111827})); w.position.set(p[0], 0.11, p[1]); g.add(w); });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.58, 36), new THREE.MeshBasicMaterial({color: c, transparent: true, opacity: 0.9, side: THREE.DoubleSide}));
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.015; g.add(ring);
  const beam = new THREE.Mesh(new THREE.ConeGeometry(0.9, 2.4, 20, 1, true), new THREE.MeshBasicMaterial({color: c, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending}));
  beam.rotation.z = -Math.PI / 2; beam.position.set(1.3, 0.25, 0); g.add(beam);
  return g;
}
M.forEach((m, mi) => {
  const g = makeRobot(COL[m]); g.scale.setScalar(1.7); overlay.add(g);
  const lab = textSprite(NAME[m], COL[m], 0.9); lab.position.y = 0.8 + mi * 0.34; lab.scale.multiplyScalar(1 / 1.7 * 1.5); g.add(lab);
  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: new THREE.Color(COL[m]), transparent: true, opacity: 0.85}));
  trail.frustumCulled = false; overlay.add(trail);
  robots[m] = {g, trail, lab};
});
// ghost = what the operator currently sees (delayed pose) for the selected controller
const ghost = makeRobot('#ffffff'); ghost.traverse(o => { if (o.material) { o.material = o.material.clone(); o.material.transparent = true; o.material.opacity = 0.28; } });
overlay.add(ghost);
const ghostLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.55}));
ghostLine.frustumCulled = false; overlay.add(ghostLine);
// collision ripples
const ripples = [];
function spawnRipple(x, y, col) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.4, 0.5, 40), new THREE.MeshBasicMaterial({color: col, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false}));
  m.rotation.x = -Math.PI / 2; m.position.copy(P3(x, y, 0.06)); m.userData.age = 0; overlay.add(m); ripples.push(m);
}

// ------------------------------------------------------------------ arena renderer + orbit
const arenaCanvas = $('arena');
const arR = new THREE.WebGLRenderer({canvas: arenaCanvas, antialias: true});
arR.outputColorSpace = THREE.LinearSRGBColorSpace;
const arCam = new THREE.PerspectiveCamera(45, 1, 0.1, 300);
const orb = {az: -0.3, el: 1.0, dist: 27, tx: 3.6, tz: -4.6, last: -1e9, drag: false, px: 0, py: 0};
arenaCanvas.addEventListener('pointerdown', e => { orb.drag = true; orb.px = e.clientX; orb.py = e.clientY; orb.last = performance.now(); arenaCanvas.setPointerCapture(e.pointerId); });
arenaCanvas.addEventListener('pointerup', () => orb.drag = false);
arenaCanvas.addEventListener('pointermove', e => { if (!orb.drag) return; orb.az -= (e.clientX - orb.px) * 0.006; orb.el = clamp(orb.el + (e.clientY - orb.py) * 0.005, 0.15, 1.5); orb.px = e.clientX; orb.py = e.clientY; orb.last = performance.now(); });
arenaCanvas.addEventListener('wheel', e => { e.preventDefault(); orb.dist = clamp(orb.dist * (1 + e.deltaY * 0.001), 7, 45); orb.last = performance.now(); }, {passive: false});
function updateArenaCam(now) {
  if (now - orb.last > 4000) orb.az += 0.0009;      // gentle idle sway
  arCam.position.set(orb.tx + Math.sin(orb.az) * Math.cos(orb.el) * orb.dist, Math.sin(orb.el) * orb.dist, orb.tz + Math.cos(orb.az) * Math.cos(orb.el) * orb.dist);
  arCam.lookAt(orb.tx, 0, orb.tz);
}

// ------------------------------------------------------------------ operator view (foveated rendering shader)
const opCanvas = $('opcanvas');
const opR = new THREE.WebGLRenderer({canvas: opCanvas, antialias: false, preserveDrawingBuffer: true});
opR.outputColorSpace = THREE.LinearSRGBColorSpace;
opR.setPixelRatio(1); opR.setSize(512, 512, false);
const opCam = new THREE.PerspectiveCamera(78, 1, 0.05, 150);
const RT_OPT = {minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: true};
const rtHi = new THREE.WebGLRenderTarget(512, 512, RT_OPT);
const rtLo = [new THREE.WebGLRenderTarget(128, 128, RT_OPT), new THREE.WebGLRenderTarget(256, 256, RT_OPT), rtHi];
const compScene = new THREE.Scene(), compCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const compMat = new THREE.ShaderMaterial({
  uniforms: {tHi: {value: rtHi.texture}, tLo: {value: rtLo[0].texture}, fov: {value: new THREE.Vector2(0.5, 0.5)},
    gaze: {value: new THREE.Vector2(0.5, 0.5)}, rad: {value: 0.3}, blkHi: {value: 1}, blkLo: {value: 6}, levHi: {value: 64},
    levLo: {value: 16}, grain: {value: 0}, stale: {value: 0}, time: {value: 0}},
  vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.,1.); }',
  fragmentShader: `
    varying vec2 vUv; uniform sampler2D tHi,tLo; uniform vec2 fov,gaze; uniform float rad,blkHi,blkLo,levHi,levLo,grain,stale,time;
    float h(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
    vec3 q(vec2 uv,float blk,float lev,sampler2D t){
      vec2 res=vec2(512.);
      vec2 u = blk>1.5 ? (floor(uv*res/blk)+0.5)*blk/res : uv;
      vec3 c=texture2D(t,u).rgb; return floor(c*lev+0.5)/lev; }
    void main(){
      vec2 d=(vUv-fov)*2.0; float dist=length(d);
      float m=smoothstep(rad+0.035,rad-0.035,dist);
      vec3 hi=q(vUv,blkHi,levHi,tHi);
      vec3 lo=q(vUv,blkLo,levLo,tLo);
      lo += (h(floor(vUv*512./2.)+floor(time*12.))-0.5)*grain;
      vec3 c=mix(lo,hi,m);
      float ring=smoothstep(0.012,0.0,abs(dist-rad)); c=mix(c,vec3(0.19,0.88,1.0),ring*0.9);
      vec2 g=abs((vUv-gaze)*512.); float cross=(g.x<1.5&&g.y<9.)||(g.y<1.5&&g.x<9.)?1.:0.;
      float cgap=(g.x<2.5&&g.y<2.5)?0.:1.; c=mix(c,vec3(1.),cross*cgap*0.95);
      float l=dot(c,vec3(0.299,0.587,0.114));
      vec3 gray=vec3(l)*vec3(1.0,0.85,0.55);
      float scan=0.5+0.5*sin(vUv.y*900.+time*20.);
      c=mix(c,gray*(0.75+0.25*scan),stale*0.85);
      gl_FragColor=vec4(c,1.);
    }`
});
compScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), compMat));
const mouse = {u: 0.5, v: 0.5, inside: false, hist: []};
opCanvas.addEventListener('pointermove', e => {
  const r = opCanvas.getBoundingClientRect(); mouse.u = clamp((e.clientX - r.left) / r.width, 0, 1); mouse.v = clamp(1 - (e.clientY - r.top) / r.height, 0, 1); mouse.inside = true;
});
opCanvas.addEventListener('pointerleave', () => mouse.inside = false);
const BLK = {lo: [14, 6, 2], hi: [5, 1, 1], levLo: [5, 12, 48], levHi: [12, 32, 64], grain: [0.07, 0.03, 0]};
function renderOperator(pose, fovUV, gazeUV, i, j, k, stale, time) {
  opCam.position.set(pose[0], 0.5, -pose[1]);
  opCam.lookAt(pose[0] + Math.cos(pose[2]), 0.42, -(pose[1] + Math.sin(pose[2])));
  overlay.visible = false;
  opR.setRenderTarget(rtHi); opR.render(scene, opCam);
  if (j < 2) { opR.setRenderTarget(rtLo[j]); opR.render(scene, opCam); }
  overlay.visible = true;
  const u = compMat.uniforms;
  u.tLo.value = rtLo[j].texture; u.fov.value.set(fovUV[0], fovUV[1]); u.gaze.value.set(gazeUV[0], gazeUV[1]);
  u.rad.value = D.radii[i]; u.blkHi.value = BLK.hi[k]; u.blkLo.value = BLK.lo[k]; u.levHi.value = BLK.levHi[k]; u.levLo.value = BLK.levLo[k];
  u.grain.value = BLK.grain[k]; u.stale.value = stale; u.time.value = time;
  opR.setRenderTarget(null); opR.render(compScene, compCam);
}

// ------------------------------------------------------------------ episode management
function resetFx() { st.lastCol = {}; M.forEach(m => st.lastCol[m] = countCol(st.ep.methods[m], st.t)); ripples.splice(0).forEach(r => overlay.remove(r)); }
function countCol(md, t) { let n = 0; for (const c of md.col) if (c <= t) n++; return n; }
function setEpisode(prof, seed) {
  st.prof = prof; st.seed = seed; st.ep = D.episodes[prof + '|' + seed];
  st.N = Math.max(...M.map(m => st.ep.methods[m].n));
  Object.entries(profBtns).forEach(([k, b]) => b.classList.toggle('on', k === prof));
  Object.entries(seedBtns).forEach(([k, b]) => b.classList.toggle('on', +k === seed));
  M.forEach(m => {  // waypoint progress per sample
    const md = st.ep.methods[m]; md.wp = new Array(md.n); let wi = 0;
    for (let s = 0; s < md.n; s++) { const w = st.ep.wps[wi]; if (w && Math.hypot(md.x[s] - w[0], md.y[s] - w[1]) < 0.6) wi++; md.wp[s] = wi; }
  });
  M.forEach(m => { const md = st.ep.methods[m]; md.qs = md.q.map((_, s) => { let a = 0, c = 0; for (let d = -4; d <= 4; d++) { const v = md.q[s + d]; if (v !== undefined) { a += v; c++; } } return a / c; }); });
  buildCourse(st.ep); st.t = 0; resetFx();
  M.forEach(m => { const md = st.ep.methods[m]; const pos = new Float32Array(md.n * 3); for (let s = 0; s < md.n; s++) { pos[s * 3] = md.x[s]; pos[s * 3 + 1] = 0.05; pos[s * 3 + 2] = -md.y[s]; } robots[m].trail.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3)); robots[m].trail.geometry.setDrawRange(0, 1); });
}

// ------------------------------------------------------------------ charts
const chCanvas = $('charts'), chCtx = chCanvas.getContext('2d');
function drawCharts() {
  const dpr = Math.min(devicePixelRatio || 1, 2), W = chCanvas.clientWidth, H = chCanvas.clientHeight;
  if (chCanvas.width !== Math.floor(W * dpr) || chCanvas.height !== Math.floor(H * dpr)) { chCanvas.width = Math.floor(W * dpr); chCanvas.height = Math.floor(H * dpr); }
  const g = chCtx; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
const ep = st.ep, N = st.N, pad = {l: 44, r: 12, t: 40, b: 14}, gap = 22, ph = (H - pad.t - pad.b - gap * 2) / 3;
  const panels = [
    {title: 'Bandwidth vs sent bitrate (Mbps, log)', key: 's', lo: 0.1, hi: 30, log: true, unit: 'Mbps'},
    {title: 'Frame age shown to operator (ms, log)', key: 'age', lo: 20, hi: 3000, log: true, unit: 'ms'},
    {title: 'Perceived quality Q (proxy, smoothed)', key: 'qs', lo: 0, hi: 1.02, log: false, unit: ''}];
  const X = (s) => pad.l + (s / (N - 1)) * (W - pad.l - pad.r);
  panels.forEach((p, pi) => {
    const y0 = pad.t + pi * (ph + gap), Y = (v) => { const f = p.log ? (Math.log(clamp(v, p.lo, p.hi)) - Math.log(p.lo)) / (Math.log(p.hi) - Math.log(p.lo)) : (v - p.lo) / (p.hi - p.lo); return y0 + ph - clamp(f, 0, 1) * ph; };
    g.fillStyle = '#9fb4e8'; g.font = '600 11px Segoe UI'; g.textAlign = 'left'; g.fillText(p.title, pad.l, y0 - 8);
    g.fillStyle = '#070d20'; g.fillRect(pad.l, y0, W - pad.l - pad.r, ph);
    // outage shading
    g.fillStyle = 'rgba(255,77,109,0.16)';
    let s0 = -1; ep.w.forEach((w, s) => { if (w <= 0.6 && s0 < 0) s0 = s; if ((w > 0.6 || s === ep.w.length - 1) && s0 >= 0) { g.fillRect(X(s0), y0, X(s) - X(s0), ph); s0 = -1; } });
    // grid
    g.strokeStyle = '#16224a'; g.lineWidth = 1; g.fillStyle = '#6f80ad'; g.font = '10px Segoe UI'; g.textAlign = 'right';
    const ticks = p.key === 's' ? [0.1, 1, 10] : p.key === 'age' ? [100, 1000] : [0, 0.5, 1];
    ticks.forEach(v => { const y = Y(v); g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke(); g.fillText(String(v), pad.l - 4, y + 3); });
    if (p.key === 'age') { g.strokeStyle = '#2fbf71'; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(pad.l, Y(100)); g.lineTo(W - pad.r, Y(100)); g.stroke(); g.setLineDash([]); g.fillStyle = '#2fbf71'; g.textAlign = 'left'; g.fillText('100 ms target', pad.l + 4, Y(100) - 3); }
    if (p.key === 's') { // available bandwidth
      g.strokeStyle = '#ffffff'; g.setLineDash([5, 3]); g.lineWidth = 1.4; g.beginPath(); ep.w.forEach((w, s) => { s ? g.lineTo(X(s), Y(w)) : g.moveTo(X(s), Y(w)); }); g.stroke(); g.setLineDash([]);
      g.fillStyle = '#fff'; g.textAlign = 'right'; g.fillText('available', W - pad.r - 4, y0 + 10);
    }
    // lines
    const order = M.filter(m => st.vis[m] && m !== st.op).concat(st.vis[st.op] ? [st.op] : []);
    order.forEach(m => {
      const md = ep.methods[m], sel = m === st.op, arr = md[p.key];
      for (const pass of [0, 1]) {
        const lim = pass ? Math.min(Math.floor(st.t), md.n - 1) : md.n - 1;
        g.globalAlpha = pass ? (sel ? 1 : 0.5) : (sel ? 0.28 : 0.08); g.strokeStyle = COL[m]; g.lineWidth = sel ? 2.1 : 1.1;
        g.beginPath(); for (let s = 0; s <= lim; s++) { s ? g.lineTo(X(s), Y(arr[s])) : g.moveTo(X(s), Y(arr[s])); } g.stroke();
      }
      g.globalAlpha = 1;
    });
    g.strokeStyle = '#31e1ff'; g.lineWidth = 1.2; g.beginPath(); g.moveTo(X(st.t), y0); g.lineTo(X(st.t), y0 + ph); g.stroke();
  });
  g.fillStyle = '#6f80ad'; g.textAlign = 'left'; g.font = '10px Segoe UI'; g.fillText('red bands = network outage · thick line = selected controller', pad.l, H - 0);
}

// ------------------------------------------------------------------ per-frame update
const clsAge = (a) => a < 100 ? '#5eff7a' : a < 250 ? '#ffc857' : '#ff4d6d';
function update(now, dtSec) {
  if (st.playing) {
    st.t = Math.max(0, st.t + dtSec / D.dt * st.speed);
    if (st.t >= st.N - 1) { st.t = st.N - 1; if (st.tour >= 0) { st.tourWait += dtSec; if (st.tourWait > 3) tourNext(); } }
  }
  const t = st.t; $('scrub').value = Math.round(t / (st.N - 1) * 1000); $('tclock').textContent = (t * D.dt).toFixed(1) + ' s';
  const ep = st.ep;
  M.forEach(m => {
    const md = ep.methods[m], tt = Math.min(t, md.n - 1), R = robots[m];
    R.g.visible = st.vis[m]; R.trail.visible = st.vis[m]; R.lab.visible = st.vis[m] && m === st.op;
    R.g.position.set(at(md.x, tt), 0, -at(md.y, tt)); R.g.rotation.y = atAng(md.th, tt);
    R.trail.geometry.setDrawRange(0, Math.max(1, Math.floor(tt) + 1));
    const c = countCol(md, tt); if (c > (st.lastCol[m] || 0) && st.vis[m]) spawnRipple(at(md.x, tt), at(md.y, tt), new THREE.Color(COL[m])); st.lastCol[m] = c;
    // scoreboard
    const r = rows[m], fin = t >= md.n - 1, wp = md.wp[Math.min(Math.floor(tt), md.n - 1)];
    const a = Math.floor(tt), age = md.age[a], qv = md.q[a], sv = md.s[a];
    r.bar.style.width = (wp / ep.wps.length * 100) + '%'; r.pg.textContent = wp + '/' + ep.wps.length;
    r.co.textContent = c; r.co.style.color = c ? '#ff4d6d' : '';
    r.ag.textContent = age + ' ms'; r.ag.style.color = clsAge(age); r.qq.textContent = qv.toFixed(2); r.ss.textContent = sv.toFixed(1) + ' Mbps';
    let label = 'driving', cl = 'run';
    if (fin) { if (md.success) { label = '✓ success ' + md.time + 's'; cl = 'ok'; } else if (md.finished) { label = '⚠ finished, ' + md.collisions + ' hit' + (md.collisions > 1 ? 's' : ''); cl = 'warn'; } else { label = '⏱ timed out'; cl = 'bad'; } }
    else if (md.late[a]) { label = '❄ frame frozen'; cl = 'warn'; }
    r.stt.textContent = label; r.stt.className = 'st ' + cl; r.tr.classList.toggle('sel', m === st.op);
  });
  for (let i = ripples.length - 1; i >= 0; i--) { const r = ripples[i]; r.userData.age += dtSec; const f = r.userData.age / 1.2; r.scale.setScalar(1 + f * 3); r.material.opacity = 0.9 * (1 - f); if (f >= 1) { overlay.remove(r); ripples.splice(i, 1); } }
  // operator view
  const md = ep.methods[st.op], tt = Math.min(t, md.n - 1), a = Math.floor(tt);
  let ageMs, i, j, k, fov, gz, Q, S, stale = md.late[a] ? 1 : 0;
  if (st.lab) {
    ageMs = st.labLat; i = st.labR; j = st.labP; k = st.labB; stale = 0;
    const nowMs = performance.now(); mouse.hist.push([nowMs, mouse.u, mouse.v]); while (mouse.hist.length > 2 && nowMs - mouse.hist[0][0] > 700) mouse.hist.shift();
    gz = [mouse.u, mouse.v]; let fv = mouse.hist[0]; for (const h of mouse.hist) { if (nowMs - h[0] >= st.labLat) fv = h; } fov = [fv[1], fv[2]];
    const e = Math.hypot((gz[0] - fov[0]) * 2, (gz[1] - fov[1]) * 2);
    Q = quality(D.radii[i], D.pres[j], D.brates[k], e); S = sentRate(D.radii[i], D.pres[j], D.brates[k]);
  } else {
    ageMs = at(md.age, tt); [i, j, k] = decodeAct(md.act[a]); Q = md.q[a]; S = md.s[a];
    const lagS = ageMs / (D.dt * 1000);
    fov = [0.5 + at(md.gx, Math.max(0, tt - lagS)) / 2, 0.5 + at(md.gy, Math.max(0, tt - lagS)) / 2];
    gz = [0.5 + at(md.gx, tt) / 2, 0.5 + at(md.gy, tt) / 2];
  }
  const lagS2 = ageMs / (D.dt * 1000), vt = Math.max(0, tt - lagS2);
  const pose = [at(md.x, vt), at(md.y, vt), atAng(md.th, vt)];
  renderOperator(pose, fov, gz, i, j, k, stale, now / 1000);
  // ghost (operator's delayed belief) in the arena
  ghost.visible = st.vis[st.op]; ghost.position.set(pose[0], 0, -pose[1]); ghost.rotation.y = pose[2];
  const cur = [at(md.x, tt), at(md.y, tt)]; const gp = ghostLine.geometry.attributes.position;
  gp.setXYZ(0, cur[0], 0.3, -cur[1]); gp.setXYZ(1, pose[0], 0.3, -pose[1]); gp.needsUpdate = true;
  // HUD
  $('hudwho').innerHTML = (st.lab ? 'FOVEATION LAB · ' : 'LIVE FEED · ') + '<span style="color:' + COL[st.op] + '">' + NAME[st.op] + '</span>';
  $('hudage').textContent = Math.round(ageMs) + ' ms'; $('hudage').style.color = clsAge(ageMs);
  $('hudact1').textContent = 'foveal radius · periphery · bitrate';
  $('hudact2').textContent = 'r ' + D.radii[i].toFixed(2) + '  ·  ' + Math.round(D.pres[j] * 100) + '%  ·  ' + D.brates[k] + ' Mbps  →  ' + S.toFixed(1) + ' Mbps sent';
  $('hudq').textContent = Q.toFixed(2); $('hudq').style.color = Q > 0.8 ? '#5eff7a' : Q > 0.6 ? '#ffc857' : '#ff4d6d';
  const W = ep.w[a], ban = $('banner');
  if (W <= 0.6) { ban.className = 'out'; ban.textContent = 'LINK OUTAGE · ' + W.toFixed(1) + ' Mbps'; }
  else if (!st.lab && md.late[a]) { ban.className = 'freeze'; ban.textContent = 'FRAME FROZEN'; }
  else ban.className = '';
  drawCharts();
  // arena
  updateArenaCam(now);
  const aw = arenaCanvas.clientWidth, ah = arenaCanvas.clientHeight;
  if (arenaCanvas.width !== Math.floor(aw * Math.min(devicePixelRatio, 2)) || arenaCanvas.height !== Math.floor(ah * Math.min(devicePixelRatio, 2))) { arR.setPixelRatio(Math.min(devicePixelRatio, 2)); arR.setSize(aw, ah, false); arCam.aspect = aw / ah; arCam.updateProjectionMatrix(); }
  arR.render(scene, arCam);
}

// ------------------------------------------------------------------ guided tour + results
const P = D.paper;
const pct = (p, m, k = 'success') => Math.round(P[p][m][k][0] * (k === 'success' ? 100 : 1));
const TOUR = [
  {prof: 'stable', seed: D.seeds[0], op: 'rl_qoe', txt: () => `<b>The problem.</b> A VR operator drives a robot through video streamed over a network. Streaming everything in full quality wastes bandwidth; streaming too little hides obstacles. Six controllers drive the <b>same</b> obstacle course over the <b>same</b> network. Drag to orbit the arena.`},
  {prof: 'congested', seed: D.seeds[0], op: 'fixed_high', txt: () => `<b>Fixed-High</b> always streams 12 Mbps. On a congested link the frames arrive late: mean frame age ${pct('congested', 'fixed_high', 'lat_ms')} ms and success <b>${pct('congested', 'fixed_high')}%</b> over 40 evaluation episodes. Look at the operator view: the picture freezes and the robot drives blind.`},
  {prof: 'congested', seed: D.seeds[0], op: 'rl_qoe', txt: () => `<b>The learned agent</b> shrinks the stream to fit the link and keeps the sharp fovea where the operator looks: success <b>${pct('congested', 'rl_qoe')}%</b>, frame age ${pct('congested', 'rl_qoe', 'lat_ms')} ms. Note the periphery: low resolution, blocky - but the operator doesn't look there.`},
  {prof: 'intermittent', seed: D.seeds[0], op: 'rl_qoe', txt: () => `<b>Outages</b> (never seen in training). The agent drops to a minimal stream, then recovers. Honest result: ${pct('intermittent', 'rl_qoe')}% success against ${pct('intermittent', 'fixed_mid')}% for the best fixed setting on this pattern - we report that gap in the paper.`},
  {prof: 'congested', seed: D.seeds[0], op: 'rl_task', txt: () => `<b>A negative result we kept.</b> Adding a robot-task penalty to the reward made the agent <i>worse</i>: ${pct('congested', 'rl_task')}% success, and it flips its settings ${pct('congested', 'rl_task', 'switch_pm')} times a minute. Ablations traced the problem to the control-error term.`},
];
function showCaption(html) { const c = $('caption'); if (html) { c.innerHTML = html; c.style.display = 'block'; } else c.style.display = 'none'; }
function tourStart(from = 0) { st.tour = from - 1; tourNext(); $('tour').textContent = '■ Stop tour'; }
function tourStop() { if (st.tour < 0) return; st.tour = -1; showCaption(null); $('tour').textContent = '▶ Guided tour'; }
function tourNext() {
  st.tour++; st.tourWait = 0;
  if (st.tour >= TOUR.length) { tourStop(); showResults(); return; }
  const s = TOUR[st.tour]; setEpisode(s.prof, s.seed); selectOp(s.op); st.playing = true; st.speed = 2;
  Object.entries(spdBtns).forEach(([k, b]) => b.classList.toggle('on', +k === 2)); $('play').textContent = '❚❚ Pause'; $('play').classList.add('on');
  showCaption(`<span style="color:var(--dim);font-size:11px">STEP ${st.tour + 1}/${TOUR.length}</span><br>` + s.txt());
}
$('tour').onclick = () => { if (st.tour >= 0) tourStop(); else tourStart(); };

function showResults() {
  const avg = (m, k) => D.profiles.slice(0, 3).concat(['intermittent']).reduce((a, p) => a + P[p][m][k][0], 0) / 4;
  const bits = Math.round((1 - avg('rl_qoe', 'bw_mbps') / avg('fixed_high', 'bw_mbps')) * 100);
  $('mbox').innerHTML = `<button class="x" id="mx">✕ Close</button><h2>What the experiments showed</h2>
  <p class="small">Mean over the four main network profiles, 40 held-out episodes each, five independently trained agents per DQN variant (2000 training episodes). Full statistics and ablations are in the paper.</p>
  <div class="kpis">
   <div class="kpi"><b>${bits}% fewer bits</b><span>DQN QoE-only vs Fixed-High (${avg('rl_qoe', 'bw_mbps').toFixed(1)} vs ${avg('fixed_high', 'bw_mbps').toFixed(0)} Mbps)</span></div>
   <div class="kpi"><b>${(avg('rl_qoe', 'success') * 100).toFixed(1)}% success</b><span>QoE-only agent · Fixed-Mid ${(avg('fixed_mid', 'success') * 100).toFixed(1)}% · Fixed-High ${(avg('fixed_high', 'success') * 100).toFixed(1)}%</span></div>
   <div class="kpi"><b>Q ${avg('rl_qoe', 'quality').toFixed(2)}</b><span>perceived-quality proxy · Fixed-Mid ${avg('fixed_mid', 'quality').toFixed(2)}</span></div>
   <div class="kpi"><b style="color:var(--rd)">${(avg('rl_task', 'success') * 100).toFixed(1)}% success</b><span>task-aware reward: the negative result</span></div>
  </div>
  <canvas id="resCanvas"></canvas>
  <p><b>Honest scope.</b> Everything here is a simulation: perceived quality is a parametric proxy, the operator and network models are assumed, and the findings show that reward design matters in this setting - not what to expect on a real headset. Starred profiles were never used in training.</p>`;
  $('modal').classList.add('show'); $('mx').onclick = () => $('modal').classList.remove('show');
  const cv = $('resCanvas'), dpr = Math.min(devicePixelRatio || 1, 2), W = cv.clientWidth, H = 300; cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const profs = ['stable', 'congested', 'highjitter', 'intermittent'], bw = (W - 80) / profs.length, barW = (bw - 24) / M.length;
  g.font = '11px Segoe UI'; g.fillStyle = '#6f80ad'; g.textAlign = 'right';
  [0, 25, 50, 75, 100].forEach(v => { const y = 20 + (1 - v / 100) * 230; g.strokeStyle = '#16224a'; g.beginPath(); g.moveTo(40, y); g.lineTo(W, y); g.stroke(); g.fillText(v + '%', 34, y + 4); });
  profs.forEach((p, pi) => {
    M.forEach((m, mi) => { const v = P[p][m].success[0] * 100, h = v / 100 * 230, x = 50 + pi * bw + mi * barW; g.fillStyle = COL[m]; g.fillRect(x, 250 - h, barW - 3, h); });
    g.fillStyle = '#9fb4e8'; g.textAlign = 'center'; g.fillText((PNAME[p]), 50 + pi * bw + bw / 2 - 10, 270);
  });
  g.textAlign = 'left'; let lx = 50; M.forEach(m => { g.fillStyle = COL[m]; g.fillRect(lx, 284, 10, 10); g.fillStyle = '#b9c8f0'; g.fillText(NAME[m], lx + 14, 293); lx += g.measureText(NAME[m]).width + 40; });
  g.save(); g.translate(12, 140); g.rotate(-Math.PI / 2); g.fillStyle = '#6f80ad'; g.textAlign = 'center'; g.fillText('task success rate', 0, 0); g.restore();
}
$('res').onclick = showResults;
$('modal').onclick = (e) => { if (e.target === $('modal')) $('modal').classList.remove('show'); };

// ------------------------------------------------------------------ keyboard + boot
addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  else if (e.key >= '1' && e.key <= '6') selectOp(M[+e.key - 1]);
  else if (e.code === 'ArrowRight') { st.t = Math.min(st.N - 1, st.t + 10); resetFx(); }
  else if (e.code === 'ArrowLeft') { st.t = Math.max(0, st.t - 10); resetFx(); }
  else if (e.key === 'Escape') $('modal').classList.remove('show');
  else if (e.key.toLowerCase() === 't') $('tour').click();
});
const H = new URLSearchParams(location.hash.slice(1));      // test/screenshot hooks: #prof=..&seed=..&t=..&op=..&pause=1&lab=1&tour=1&results=1
setEpisode(H.get('prof') || st.prof, +(H.get('seed') || st.seed));
if (H.get('op')) selectOp(H.get('op'));
if (H.get('t')) { st.t = +H.get('t'); resetFx(); }
if (H.get('pause')) $('play').click();
if (H.get('lab')) labBtn.click();
if (H.get('tour')) tourStart(+H.get('tour') - 1);
if (H.get('results')) showResults();
const FAST = +(H.get('fast') || 1);                           // test hook: fast-forward
let prev = performance.now();
function loop(now) {
  const dt = clamp((now - prev) / 1000, 0, 0.1) * FAST; prev = now;     // never negative on the first frame
  try { update(now, dt); } catch (err) { console.error(err); }   // one bad frame must not stop the demo
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
window.__demo = {st, update, setEpisode};
})();
