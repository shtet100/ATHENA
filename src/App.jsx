import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

// ============ NOISE ============
function seededRandom(seed) {
  let s = seed | 0;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s - 1) / 2147483646; };
}

let _noiseData1, _noiseData2, _noiseData3;
const NS = 256;

function initNoise(seed) {
  const rng1 = seededRandom(seed);
  const rng2 = seededRandom(seed + 137);
  const rng3 = seededRandom(seed + 293);
  _noiseData1 = new Float32Array(NS * NS);
  _noiseData2 = new Float32Array(NS * NS);
  _noiseData3 = new Float32Array(NS * NS);
  for (let i = 0; i < NS * NS; i++) {
    _noiseData1[i] = rng1();
    _noiseData2[i] = rng2();
    _noiseData3[i] = rng3();
  }
}

function smoothNoise(noise, x, y) {
  const ix = ((Math.floor(x) % NS) + NS) % NS;
  const iy = ((Math.floor(y) % NS) + NS) % NS;
  const fx = x - Math.floor(x), fy = y - Math.floor(y);
  const nx = (ix + 1) % NS, ny = (iy + 1) % NS;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  return (noise[iy * NS + ix] * (1 - sx) + noise[iy * NS + nx] * sx) * (1 - sy) +
         (noise[ny * NS + ix] * (1 - sx) + noise[ny * NS + nx] * sx) * sy;
}

function fbm(x, y, noise, octaves) {
  let val = 0, amp = 1, freq = 1, maxAmp = 0;
  for (let i = 0; i < octaves; i++) {
    val += smoothNoise(noise, x * freq, y * freq) * amp;
    maxAmp += amp; amp *= 0.45; freq *= 2.1;
  }
  return val / maxAmp;
}

// ============ CONTINUOUS WORLD HEIGHT ============
const MAX_HEIGHT = 10;
let _roughness = 1.0;
let _craterDensity = 0.5;

// Deterministic craters: divide world into cells, hash cell coords for crater placement
function cellHash(cx, cz) {
  let h = (cx * 374761393 + cz * 668265263 + 1234567) | 0;
  h = ((h ^ (h >> 13)) * 1274126177) | 0;
  h = h ^ (h >> 16);
  return h;
}

function getCratersInCell(cx, cz) {
  const h = cellHash(cx, cz);
  const rng = seededRandom(Math.abs(h));
  const count = Math.floor(rng() * 3 * _craterDensity + rng() * _craterDensity);
  const craters = [];
  for (let i = 0; i < count; i++) {
    craters.push({
      x: cx * 50 + rng() * 50,
      z: cz * 50 + rng() * 50,
      r: 4 + rng() * 14,
      depth: 0.15 + rng() * 0.4,
    });
  }
  return craters;
}

function getWorldHeight(wx, wz) {
  if (!_noiseData1) return 0;
  const nx = wx * 0.08, nz = wz * 0.08;
  let h = fbm(nx, nz, _noiseData1, 7) * _roughness * 0.65;
  h += fbm(nx * 2.3 + 5.2, nz * 2.3 + 1.3, _noiseData2, 5) * _roughness * 0.22;
  h += fbm(nx * 5.1 + 9.1, nz * 5.1 + 3.7, _noiseData3, 4) * _roughness * 0.1;
  const ridge = Math.abs(fbm(nx * 1.5, nz * 1.5, _noiseData2, 4) - 0.5) * 2;
  h += ridge * _roughness * 0.12;

  // Apply craters from nearby cells
  const ccx = Math.floor(wx / 50), ccz = Math.floor(wz / 50);
  for (let dx = -1; dx <= 1; dx++) {
    for (let dz = -1; dz <= 1; dz++) {
      const craters = getCratersInCell(ccx + dx, ccz + dz);
      for (const c of craters) {
        const dist = Math.sqrt((wx - c.x) ** 2 + (wz - c.z) ** 2);
        if (dist < c.r * 1.15) {
          const t = dist / c.r;
          if (t < 0.85) h -= c.depth * (1 - (t / 0.85) ** 2);
          if (t > 0.75 && t < 1.15) {
            const rt = (t - 0.75) / 0.4;
            h += c.depth * 0.3 * Math.sin(rt * Math.PI) * (1 - rt * 0.3);
          }
        }
      }
    }
  }
  return h * MAX_HEIGHT;
}

function getWorldSlope(wx, wz) {
  const d = 0.5;
  const hL = getWorldHeight(wx - d, wz), hR = getWorldHeight(wx + d, wz);
  const hU = getWorldHeight(wx, wz - d), hD = getWorldHeight(wx, wz + d);
  return Math.sqrt(((hR - hL) / (2 * d)) ** 2 + ((hD - hU) / (2 * d)) ** 2);
}

function getWorldNormal(wx, wz) {
  const d = 0.4;
  const hL = getWorldHeight(wx - d, wz), hR = getWorldHeight(wx + d, wz);
  const hU = getWorldHeight(wx, wz - d), hD = getWorldHeight(wx, wz + d);
  const n = new THREE.Vector3(hL - hR, 2 * d, hU - hD);
  n.normalize();
  return n;
}

// ============ CHUNK SYSTEM ============
const CHUNK_SIZE = 80;
const CHUNK_VERTS = 64;
const VIEW_RANGE = 3; // chunks in each direction = (2*3+1)^2 = 49 chunks

function chunkKey(cx, cz) { return `${cx},${cz}`; }

function buildChunkMesh(cx, cz) {
  const ox = cx * CHUNK_SIZE, oz = cz * CHUNK_SIZE;
  const geo = new THREE.PlaneGeometry(CHUNK_SIZE, CHUNK_SIZE, CHUNK_VERTS, CHUNK_VERTS);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);

  for (let i = 0; i < pos.count; i++) {
    // keep X and Z local, only set Y from world height
    const lx = pos.getX(i);
    const lz = pos.getZ(i);
    const wx = lx + ox;
    const wz = lz + oz;
    const h = getWorldHeight(wx, wz);
    pos.setY(i, h);
    const sl = getWorldSlope(wx, wz);
    const t = (h / MAX_HEIGHT + 0.5) * 0.5;
    const dark = sl > 1.5 ? 0.82 : 1;
    colors[i * 3] = (0.6 + t * 0.3) * dark;
    colors[i * 3 + 1] = (0.26 + t * 0.16) * dark;
    colors[i * 3 + 2] = (0.11 + t * 0.06) * dark;
  }

  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.95, metalness: 0.05, flatShading: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(ox, 0, oz);
  mesh.receiveShadow = true;
  return mesh;
}

function buildChunkRocks(cx, cz) {
  const ox = cx * CHUNK_SIZE, oz = cz * CHUNK_SIZE;
  const h = cellHash(cx * 7 + 3, cz * 13 + 7);
  const rng = seededRandom(Math.abs(h) + 9999);
  const numRocks = 8 + Math.floor(rng() * 12 * (_craterDensity * 0.5 + 0.5));
  const group = new THREE.Group();

  const rockGeos = [
    new THREE.DodecahedronGeometry(1, 0),
    new THREE.OctahedronGeometry(1, 0),
    new THREE.TetrahedronGeometry(1, 0),
  ];
  const rockMats = [
    new THREE.MeshStandardMaterial({ color: 0x6b3a23, roughness: 0.95, flatShading: true }),
    new THREE.MeshStandardMaterial({ color: 0x5a3020, roughness: 0.95, flatShading: true }),
    new THREE.MeshStandardMaterial({ color: 0x7a4530, roughness: 0.95, flatShading: true }),
  ];

  for (let i = 0; i < numRocks; i++) {
    const rx = (rng() - 0.5) * CHUNK_SIZE * 0.9;
    const rz = (rng() - 0.5) * CHUNK_SIZE * 0.9;
    const rh = getWorldHeight(rx + ox, rz + oz);
    const scale = 0.15 + rng() * 0.55;
    const rock = new THREE.Mesh(rockGeos[Math.floor(rng() * 3)], rockMats[Math.floor(rng() * 3)]);
    rock.position.set(rx, rh + scale * 0.2, rz);
    rock.scale.set(scale * (0.7 + rng() * 0.6), scale * (0.4 + rng() * 0.6), scale * (0.7 + rng() * 0.6));
    rock.rotation.set(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI);
    rock.castShadow = true;
    group.add(rock);
  }
  group.position.set(ox, 0, oz);
  return group;
}

// hazard overlay chunk
function buildChunkHazard(cx, cz) {
  const ox = cx * CHUNK_SIZE, oz = cz * CHUNK_SIZE;
  const geo = new THREE.PlaneGeometry(CHUNK_SIZE, CHUNK_SIZE, CHUNK_VERTS, CHUNK_VERTS);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 4);

  for (let i = 0; i < pos.count; i++) {
    const lx = pos.getX(i);
    const lz = pos.getZ(i);
    const wx = lx + ox;
    const wz = lz + oz;
    const h = getWorldHeight(wx, wz);
    pos.setY(i, h + 0.15);
    const sl = getWorldSlope(wx, wz);
    if (sl > 3.5) { colors[i*4]=1; colors[i*4+1]=0.1; colors[i*4+2]=0.1; colors[i*4+3]=0.35; }
    else if (sl > 2) { colors[i*4]=1; colors[i*4+1]=0.65; colors[i*4+2]=0; colors[i*4+3]=0.2; }
    else { colors[i*4]=0.1; colors[i*4+1]=0.85; colors[i*4+2]=0.3; colors[i*4+3]=0.06; }
  }

  geo.setAttribute("color", new THREE.BufferAttribute(colors, 4));
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(ox, 0, oz);
  mesh.visible = false;
  return mesh;
}

// ============ A* STEPPER (VISUALIZABLE) ============
const ASTAR_STEP = 2.5;

class AStarStepper {
  constructor(startW, endW) {
    this.STEP = ASTAR_STEP;
    this.sx = Math.round(startW[0] / this.STEP);
    this.sz = Math.round(startW[1] / this.STEP);
    this.ex = Math.round(endW[0] / this.STEP);
    this.ez = Math.round(endW[1] / this.STEP);
    this.open = new Map();
    this.closed = new Map(); // key -> {x, z, cost}
    this.gScore = new Map();
    this.parent = new Map();
    this.dirs = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
    this.done = false;
    this.found = false;
    this.path = null;
    this.iters = 0;
    this.maxCostSeen = 1;

    const startKey = `${this.sx},${this.sz}`;
    this.gScore.set(startKey, 0);
    this.open.set(startKey, { x: this.sx, z: this.sz, f: 0 });
  }

  heuristic(x, z) {
    return Math.sqrt((x - this.ex) ** 2 + (z - this.ez) ** 2);
  }

  step(n = 1) {
    for (let s = 0; s < n; s++) {
      if (this.done || this.open.size === 0) { this.done = true; return; }
      this.iters++;
      if (this.iters > 50000) { this.done = true; return; }

      let bestKey = null, bestF = Infinity;
      for (const [k, v] of this.open) { if (v.f < bestF) { bestF = v.f; bestKey = k; } }
      const curr = this.open.get(bestKey);
      this.open.delete(bestKey);
      const cost = this.gScore.get(bestKey) || 0;
      this.closed.set(bestKey, { x: curr.x, z: curr.z, cost });
      if (cost > this.maxCostSeen) this.maxCostSeen = cost;

      if (curr.x === this.ex && curr.z === this.ez) {
        const path = [];
        let ck = bestKey;
        while (ck) {
          const [px, pz] = ck.split(",").map(Number);
          path.unshift([px * this.STEP, pz * this.STEP]);
          ck = this.parent.get(ck);
        }
        this.path = path;
        this.found = true;
        this.done = true;
        return;
      }

      for (const [dx, dz] of this.dirs) {
        const nx = curr.x + dx, nz = curr.z + dz;
        const nk = `${nx},${nz}`;
        if (this.closed.has(nk)) continue;
        const wx = nx * this.STEP, wz = nz * this.STEP;
        const slope = getWorldSlope(wx, wz);
        const slopeCost = slope > 3.5 ? 200 : slope > 2 ? 15 : 1 + slope * 2;
        const dist = Math.sqrt(dx * dx + dz * dz);
        const ng = (this.gScore.get(bestKey) || 0) + dist * slopeCost;
        if (!this.gScore.has(nk) || ng < this.gScore.get(nk)) {
          this.gScore.set(nk, ng);
          this.parent.set(nk, bestKey);
          this.open.set(nk, { x: nx, z: nz, f: ng + this.heuristic(nx, nz) });
        }
      }
    }
  }

  // Get positions for visualization
  getClosedPositions() {
    const pts = [];
    for (const [, v] of this.closed) {
      const wx = v.x * this.STEP, wz = v.z * this.STEP;
      pts.push({ x: wx, z: wz, cost: v.cost });
    }
    return pts;
  }

  getOpenPositions() {
    const pts = [];
    for (const [, v] of this.open) {
      const wx = v.x * this.STEP, wz = v.z * this.STEP;
      pts.push({ x: wx, z: wz });
    }
    return pts;
  }
}

// Instant path (used when viz is off)
function astarPathInstant(startW, endW) {
  const stepper = new AStarStepper(startW, endW);
  stepper.step(50000);
  return stepper.path;
}

function smoothPath(path) {
  if (!path || path.length < 3) return path;
  const s = [path[0]];
  for (let i = 1; i < path.length - 1; i++) {
    const p = path[i - 1], c = path[i], n = path[i + 1];
    s.push([c[0] * 0.5 + p[0] * 0.25 + n[0] * 0.25, c[1] * 0.5 + p[1] * 0.25 + n[1] * 0.25]);
  }
  s.push(path[path.length - 1]);
  return s;
}

// ============ PLANET ============
const PLANET = {
  name: "MARS", subtitle: "JEZERO CRATER REGION",
  gravity: "3.72 m/s²", atmosphere: "0.6 kPa CO₂",
  temp: "−60°C avg", windSpeed: "7.2 m/s", sol: "SOL 847",
};

// ============ LOADING ============
function LoadingScreen({ progress, stage }) {
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, background: "#060302", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", fontFamily: "'Courier New', monospace", color: "#e8c8a0" }}>
      <div style={{ position: "absolute", inset: 0, overflow: "hidden", opacity: 0.25 }}>
        {Array.from({ length: 45 }, (_, i) => (
          <div key={i} style={{ position: "absolute", left: `${(i * 37 + 13) % 100}%`, top: `${(i * 53 + 7) % 100}%`, width: i % 3 === 0 ? 2 : 1, height: i % 3 === 0 ? 2 : 1, background: "#fff", borderRadius: "50%", opacity: 0.3 + (i % 4) * 0.12, animation: `tw ${2 + i % 3}s ease-in-out infinite`, animationDelay: `${i * 0.1}s` }} />
        ))}
      </div>
      <style>{`@keyframes tw { 0%,100% { opacity:0.15; } 50% { opacity:0.6; } } @keyframes pl { 0%,100% { opacity:0.5; } 50% { opacity:1; } }`}</style>
      <div style={{ width: 52, height: 52, borderRadius: "50%", background: "radial-gradient(circle at 35% 35%, #cc6644, #883322 60%, #552211)", boxShadow: "0 0 35px rgba(204,68,34,0.2)", marginBottom: 26 }} />
      <div style={{ fontSize: 9, letterSpacing: 6, opacity: 0.25, marginBottom: 5 }}>ORION SUBSYSTEM</div>
      <div style={{ fontSize: 36, fontWeight: "bold", letterSpacing: 10, color: "#ffd4a0", marginBottom: 3 }}>A T H E N A</div>
      <div style={{ fontSize: 7, letterSpacing: 4, opacity: 0.18, marginBottom: 40 }}>AUTONOMOUS TERRAIN & HAZARD EXPLORATION NAVIGATION AGENT</div>
      <div style={{ width: 280, marginBottom: 10 }}>
        <div style={{ width: "100%", height: 2, background: "rgba(255,180,100,0.07)", borderRadius: 1, overflow: "hidden" }}>
          <div style={{ width: `${progress}%`, height: "100%", background: "linear-gradient(90deg, #ffaa44, #ff6622)", borderRadius: 1, transition: "width 0.3s ease", boxShadow: "0 0 6px rgba(255,170,68,0.3)" }} />
        </div>
      </div>
      <div style={{ fontSize: 8, letterSpacing: 3, opacity: 0.3, marginBottom: 4 }}>{stage}</div>
      <div style={{ fontSize: 10, letterSpacing: 2, color: "#ffaa44", animation: "pl 1.5s ease-in-out infinite" }}>{progress}%</div>
      <div style={{ position: "absolute", bottom: 32, textAlign: "center" }}>
        <div style={{ fontSize: 8, letterSpacing: 3, opacity: 0.2, marginBottom: 5 }}>CREATED BY</div>
        <div style={{ fontSize: 11, letterSpacing: 2, opacity: 0.45, marginBottom: 3 }}>Swan Yi Htet — <span style={{ opacity: 0.55 }}>Columbia University</span></div>
        <div style={{ fontSize: 11, letterSpacing: 2, opacity: 0.45, marginBottom: 12 }}>David Young — <span style={{ opacity: 0.55 }}>University of Pennsylvania</span></div>
        <div style={{ fontSize: 6, letterSpacing: 3, opacity: 0.1 }}>ORION SYSTEMS v1.0</div>
      </div>
    </div>
  );
}

function OnboardingOverlay({ onStart }) {
  const P = { background: "rgba(8,4,2,0.85)", border: "1px solid rgba(255,170,80,0.1)", borderRadius: 3 };
  return (
    <div style={{ position: "absolute", inset: 0, zIndex: 100, background: "rgba(5,2,1,0.85)", display: "flex", alignItems: "center", justifyContent: "center", backdropFilter: "blur(10px)", fontFamily: "'Courier New', monospace", color: "#e8c8a0" }}>
      <div style={{ maxWidth: 480, padding: 32, textAlign: "center" }}>
        <div style={{ fontSize: 9, letterSpacing: 5, opacity: 0.3, marginBottom: 5 }}>ORION SUBSYSTEM</div>
        <div style={{ fontSize: 28, fontWeight: "bold", letterSpacing: 8, color: "#ffd4a0", marginBottom: 3 }}>A T H E N A</div>
        <div style={{ fontSize: 7, letterSpacing: 3, opacity: 0.2, marginBottom: 22 }}>AUTONOMOUS TERRAIN & HAZARD EXPLORATION NAVIGATION AGENT</div>
        <div style={{ ...P, padding: 16, marginBottom: 12, textAlign: "left" }}>
          <div style={{ fontSize: 9, letterSpacing: 2, color: "#ffaa44", marginBottom: 7 }}>MISSION BRIEFING — MARS SURFACE OPS</div>
          <div style={{ fontSize: 10, lineHeight: 1.8, opacity: 0.55 }}>
            You are commanding an autonomous rover on the Martian surface near Jezero Crater.
            The terrain extends infinitely in every direction. Set waypoints anywhere on the
            landscape and the rover will plan a safe route, avoiding steep slopes and hazards.
            Drive as far as you want — there are no boundaries.
          </div>
        </div>
        <div style={{ ...P, padding: 16, marginBottom: 18, textAlign: "left" }}>
          <div style={{ fontSize: 9, letterSpacing: 2, color: "#ffaa44", marginBottom: 7 }}>CONTROLS</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 16px", fontSize: 9, lineHeight: 1.5, opacity: 0.55 }}>
            <div><span style={{ color: "#00ffaa" }}>LEFT CLICK</span> — Set nav target</div>
            <div><span style={{ color: "#00ffaa" }}>RIGHT DRAG</span> — Orbit camera</div>
            <div><span style={{ color: "#00ffaa" }}>SHIFT+DRAG</span> — Orbit (alt)</div>
            <div><span style={{ color: "#00ffaa" }}>SCROLL</span> — Zoom in/out</div>
            <div><span style={{ color: "#00ffaa" }}>W / S</span> — Drive forward/reverse</div>
            <div><span style={{ color: "#00ffaa" }}>A / D</span> — Steer left/right</div>
          </div>
          <div style={{ marginTop: 8, fontSize: 9, lineHeight: 1.6, opacity: 0.45 }}>
            <span style={{ color: "#00ccff" }}>PATHFINDING AI</span> — When you click a target, watch the A* search algorithm expand in real time. Green-to-red heat map shows traversal cost. Cyan dots show the frontier. Adjust speed with the STEPS/FRAME slider.
          </div>
        </div>
        <button onClick={onStart} style={{ padding: "10px 40px", fontSize: 10, letterSpacing: 4, background: "rgba(0,255,170,0.06)", border: "1px solid rgba(0,255,170,0.25)", color: "#00ffaa", borderRadius: 3, cursor: "pointer" }}
          onMouseOver={e => e.target.style.background = "rgba(0,255,170,0.12)"}
          onMouseOut={e => e.target.style.background = "rgba(0,255,170,0.06)"}
        >BEGIN MISSION</button>
      </div>
    </div>
  );
}

// ============ 3D ENGINEERING PANEL ============
function EngPanel3D({ speed, suspAngle, wheelRpm, motorPower, terrainGrade, wheelRotAnim, onHide, roverX, roverZ }) {
  const engMountRef = useRef(null);
  const engSceneRef = useRef({});
  const engAnimRef = useRef(null);

  useEffect(() => {
    const mount = engMountRef.current;
    if (!mount) return;

    const w = mount.clientWidth, h = mount.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, w / h, 0.01, 50);
    camera.position.set(3, 2.5, 3);
    camera.lookAt(0, 0.5, 0);

    // Lighting
    const light = new THREE.DirectionalLight(0xffeedd, 1.5);
    light.position.set(4, 6, 3);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x776655, 0.6));
    scene.add(new THREE.HemisphereLight(0xddaa77, 0x443322, 0.3));

    // Build detailed rover
    const rover = new THREE.Group();
    rover.name = "engRover";

    // Chassis
    const chassis = new THREE.Mesh(
      new THREE.BoxGeometry(1.6, 0.35, 1.0),
      new THREE.MeshStandardMaterial({ color: 0xd0d0d0, metalness: 0.7, roughness: 0.25 })
    );
    chassis.position.y = 0.55; rover.add(chassis);

    // Deck
    const deck = new THREE.Mesh(
      new THREE.BoxGeometry(0.9, 0.15, 0.7),
      new THREE.MeshStandardMaterial({ color: 0x999999, metalness: 0.8, roughness: 0.2 })
    );
    deck.position.set(-0.1, 0.83, 0); rover.add(deck);

    // Camera mast
    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.03, 0.7),
      new THREE.MeshStandardMaterial({ color: 0xaaaaaa })
    );
    mast.position.set(0.4, 1.2, 0); rover.add(mast);

    // Camera head
    const camH = new THREE.Mesh(
      new THREE.BoxGeometry(0.18, 0.12, 0.15),
      new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 0.5 })
    );
    camH.position.set(0.4, 1.6, 0); rover.add(camH);

    // Lenses
    [-0.04, 0.04].forEach(z => {
      const l = new THREE.Mesh(
        new THREE.CylinderGeometry(0.03, 0.03, 0.04, 8),
        new THREE.MeshStandardMaterial({ color: 0x1a1a2e, metalness: 0.9 })
      );
      l.rotation.z = Math.PI / 2; l.position.set(0.5, 1.6, z); rover.add(l);
    });

    // Antenna
    const dish = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: 0xf0e8d0, metalness: 0.3, side: THREE.DoubleSide })
    );
    dish.position.set(-0.35, 1.15, 0); dish.rotation.x = Math.PI; rover.add(dish);
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.015, 0.015, 0.4),
      new THREE.MeshStandardMaterial({ color: 0xcccccc })
    );
    pole.position.set(-0.35, 0.95, 0); rover.add(pole);

    // Solar panels
    [-0.55, 0.55].forEach(z => {
      const arm = new THREE.Mesh(
        new THREE.BoxGeometry(0.04, 0.04, 0.3),
        new THREE.MeshStandardMaterial({ color: 0x888888 })
      );
      arm.position.set(-0.1, 0.82, z * 0.7); rover.add(arm);
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.02, 0.35),
        new THREE.MeshStandardMaterial({ color: 0x1a2a55, metalness: 0.4, roughness: 0.3 })
      );
      panel.position.set(-0.1, 0.85, z); rover.add(panel);
    });

    // Wheels
    const wheels = [];
    const wg = new THREE.CylinderGeometry(0.18, 0.18, 0.1, 16);
    const wm = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.8 });
    [[-0.6,0.18,0.55],[-0.6,0.18,-0.55],[0,0.18,0.55],[0,0.18,-0.55],[0.6,0.18,0.55],[0.6,0.18,-0.55]].forEach(([x,y,z]) => {
      // Spoke group for visible rotation
      const wheelGroup = new THREE.Group();
      wheelGroup.position.set(x, y, z);

      const tire = new THREE.Mesh(wg, wm);
      tire.rotation.x = Math.PI / 2;
      wheelGroup.add(tire);

      // Spokes
      for (let s = 0; s < 6; s++) {
        const spoke = new THREE.Mesh(
          new THREE.BoxGeometry(0.01, 0.15, 0.01),
          new THREE.MeshStandardMaterial({ color: 0x555555 })
        );
        spoke.rotation.x = Math.PI / 2;
        spoke.rotation.z = (s / 6) * Math.PI;
        wheelGroup.add(spoke);
      }

      // Hub
      const hub = new THREE.Mesh(
        new THREE.CylinderGeometry(0.04, 0.04, 0.12, 8),
        new THREE.MeshStandardMaterial({ color: 0x666666, metalness: 0.7 })
      );
      hub.rotation.x = Math.PI / 2;
      wheelGroup.add(hub);

      rover.add(wheelGroup);
      wheels.push(wheelGroup);
    });

    scene.add(rover);

    // Ghost wireframe terrain patch — samples real world terrain
    const tGeo = new THREE.PlaneGeometry(6, 6, 40, 40);
    tGeo.rotateX(-Math.PI / 2);
    const tMat = new THREE.MeshBasicMaterial({
      color: 0x00ffaa,
      wireframe: true,
      transparent: true,
      opacity: 0.08,
      depthWrite: false,
    });
    const terrainPatch = new THREE.Mesh(tGeo, tMat);
    terrainPatch.name = "engTerrain";
    scene.add(terrainPatch);

    // Faint solid fill underneath the wireframe for depth
    const tGeo2 = new THREE.PlaneGeometry(6, 6, 40, 40);
    tGeo2.rotateX(-Math.PI / 2);
    const tMat2 = new THREE.MeshStandardMaterial({
      color: 0xaa7744,
      transparent: true,
      opacity: 0.06,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const terrainFill = new THREE.Mesh(tGeo2, tMat2);
    terrainFill.name = "engTerrainFill";
    scene.add(terrainFill);

    engSceneRef.current = { renderer, scene, camera, rover, wheels, terrainPatch, terrainFill };

    // Orbit controls for mini viewport
    let eTheta = Math.PI / 4, ePhi = Math.PI / 4, eDist = 4;
    let eDrag = false, eLastX = 0, eLastY = 0;

    const eWheel = (e) => { e.preventDefault(); e.stopPropagation(); eDist = Math.max(1.5, Math.min(8, eDist + e.deltaY * 0.005)); };
    const eDown = (e) => { eDrag = true; eLastX = e.clientX; eLastY = e.clientY; e.stopPropagation(); };
    const eMove = (e) => {
      if (eDrag) {
        eTheta -= (e.clientX - eLastX) * 0.008;
        ePhi = Math.max(0.1, Math.min(Math.PI / 2.1, ePhi - (e.clientY - eLastY) * 0.008));
        eLastX = e.clientX; eLastY = e.clientY;
        e.stopPropagation();
      }
    };
    const eUp = () => { eDrag = false; };

    const canvas = renderer.domElement;
    canvas.addEventListener("wheel", eWheel, { passive: false });
    canvas.addEventListener("mousedown", eDown);
    canvas.addEventListener("mousemove", eMove);
    canvas.addEventListener("mouseup", eUp);
    canvas.addEventListener("mouseleave", eUp);

    let eTime = 0;
    const engAnimate = () => {
      engAnimRef.current = requestAnimationFrame(engAnimate);
      eTime += 0.016;

      // Update camera orbit
      camera.position.set(
        eDist * Math.sin(eTheta) * Math.cos(ePhi),
        0.6 + eDist * Math.sin(ePhi),
        eDist * Math.cos(eTheta) * Math.cos(ePhi)
      );
      camera.lookAt(0, 0.6, 0);

      renderer.render(scene, camera);
    };
    engAnimate();

    return () => {
      cancelAnimationFrame(engAnimRef.current);
      canvas.removeEventListener("wheel", eWheel);
      canvas.removeEventListener("mousedown", eDown);
      canvas.removeEventListener("mousemove", eMove);
      canvas.removeEventListener("mouseup", eUp);
      canvas.removeEventListener("mouseleave", eUp);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, []);

  // Update rover tilt and wheel spin from props
  useEffect(() => {
    const { rover, wheels, terrainPatch, terrainFill } = engSceneRef.current;
    if (!rover) return;

    // Clamp tilt to realistic range
    const clampedAngle = Math.max(-25, Math.min(25, suspAngle));
    const slopeRad = (clampedAngle * Math.PI) / 180;
    rover.rotation.set(-slopeRad * 0.8, -Math.PI / 2, 0);
    rover.position.y = -0.12;

    // Spin wheels
    wheels.forEach(w => {
      w.rotation.z = wheelRotAnim * 0.5;
    });

    // Sample REAL terrain heights from the main simulation
    if (terrainPatch && roverX !== undefined) {
      const tPos = terrainPatch.geometry.attributes.position;
      const roverH = getWorldHeight(roverX, roverZ);
      for (let i = 0; i < tPos.count; i++) {
        const lx = tPos.getX(i);
        const lz = tPos.getZ(i);
        const wx = roverX + lx;
        const wz = roverZ + lz;
        const h = getWorldHeight(wx, wz) - roverH;
        tPos.setY(i, h);
      }
      tPos.needsUpdate = true;
      terrainPatch.geometry.computeVertexNormals();

      // Sync the fill mesh to same heights
      if (terrainFill) {
        const fPos = terrainFill.geometry.attributes.position;
        for (let i = 0; i < fPos.count; i++) {
          fPos.setY(i, tPos.getY(i));
        }
        fPos.needsUpdate = true;
        terrainFill.geometry.computeVertexNormals();
      }
    }
  }, [suspAngle, wheelRotAnim, speed, roverX, roverZ]);

  const P = { background: "rgba(8,4,2,0.88)", border: "1px solid rgba(255,170,80,0.1)", borderRadius: 3 };

  return (
    <div style={{
      position: "absolute", bottom: 28, right: 10, width: 340, zIndex: 10,
      ...P, padding: 0, overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 10px 0" }}>
        <div style={{ fontSize: 7, letterSpacing: 2, color: "#ffaa44" }}>ROVER ENGINEERING — 3D</div>
        <button onClick={onHide} style={{ fontSize: 7, background: "none", border: "none", color: "#554433", cursor: "pointer", letterSpacing: 1 }}>HIDE</button>
      </div>
      <div style={{ fontSize: 5, letterSpacing: 1.5, opacity: 0.2, padding: "2px 10px 4px" }}>DRAG TO ORBIT — SCROLL TO ZOOM</div>

      {/* 3D Viewport */}
      <div ref={engMountRef} style={{ width: "100%", height: 180, cursor: "grab" }} />

      {/* Telemetry Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 10px", padding: "6px 10px" }}>
        {[
          ["WHEEL RPM", wheelRpm, parseFloat(wheelRpm) > 0 ? "#00ffaa" : "#665544"],
          ["MOTOR PWR", `${motorPower} W`, parseFloat(motorPower) > 0 ? "#00ffaa" : "#665544"],
          ["SUSP ANGLE", `${suspAngle}°`, Math.abs(suspAngle) > 5 ? "#ffaa00" : "#e8c8a0"],
          ["GRADE", `${terrainGrade}%`, parseFloat(terrainGrade) > 200 ? "#ff4444" : "#e8c8a0"],
        ].map(([label, val, color]) => (
          <div key={label}>
            <div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.22, fontFamily: "monospace" }}>{label}</div>
            <div style={{ fontSize: 9, letterSpacing: 0.5, color, fontFamily: "monospace" }}>{val}</div>
          </div>
        ))}
      </div>

      <div style={{ padding: "3px 10px 7px", fontSize: 5.5, lineHeight: 1.5, opacity: 0.2, letterSpacing: 0.5, fontFamily: "monospace" }}>
        ROCKER-BOGIE — 6 wheels, no springs. Each side has two segments on a differential pivot, keeping all wheels grounded on uneven terrain.
      </div>
    </div>
  );
}

// ============ MAIN ============
export default function ATHENA() {
  const mountRef = useRef(null);
  const sceneRef = useRef({});
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [loadingStage, setLoadingStage] = useState("INITIALIZING...");
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showSensor, setShowSensor] = useState(true);
  const [showHazard, setShowHazard] = useState(false);
  const [roughness, setRoughness] = useState(1.0);
  const [craterDensity, setCraterDensity] = useState(0.5);
  const [status, setStatus] = useState("AWAITING MISSION DIRECTIVE");
  const [roverPos, setRoverPos] = useState({ x: 0, z: 0 });
  const [roverAlt, setRoverAlt] = useState(0);
  const [slopeVal, setSlopeVal] = useState(0);
  const [speed, setSpeed] = useState(0);
  const [distance, setDistance] = useState(0);
  const [wheelRpm, setWheelRpm] = useState(0);
  const [suspAngle, setSuspAngle] = useState(0);
  const [motorPower, setMotorPower] = useState(0);
  const [terrainGrade, setTerrainGrade] = useState(0);
  const [wheelRotAnim, setWheelRotAnim] = useState(0);
  const [showEngPanel, setShowEngPanel] = useState(true);
  const [showAlgoViz, setShowAlgoViz] = useState(true);
  const [vizSpeed, setVizSpeed] = useState(60);
  const [vizStats, setVizStats] = useState({ explored: 0, frontier: 0, iters: 0 });
  const astarStepperRef = useRef(null);
  const vizMeshesRef = useRef({ explored: null, frontier: null });
  const pathRef = useRef(null);
  const roverRef = useRef({ x: 0, z: 0, angle: 0, pathIdx: 0, moving: false });
  const seedRef = useRef(42);
  const keysRef = useRef({});
  const animRef = useRef(null);
  const distRef = useRef(0);
  const showSensorRef = useRef(true);
  const showHazardRef = useRef(false);
  const showAlgoVizRef = useRef(true);
  const vizSpeedRef = useRef(60);
  const chunksRef = useRef(new Map());
  const sceneRefForRebuild = useRef(null);
  const rebuildChunksRef = useRef(null);

  useEffect(() => { showSensorRef.current = showSensor; }, [showSensor]);
  useEffect(() => { showHazardRef.current = showHazard; }, [showHazard]);
  useEffect(() => { showAlgoVizRef.current = showAlgoViz; }, [showAlgoViz]);
  useEffect(() => { vizSpeedRef.current = vizSpeed; }, [vizSpeed]);
  useEffect(() => {
    _roughness = roughness;
    _craterDensity = craterDensity;
    // Rebuild all chunks with new terrain params
    const scene = sceneRefForRebuild.current;
    if (scene && rebuildChunksRef.current) {
      // Remove all existing chunks
      for (const [, data] of chunksRef.current) {
        scene.remove(data.terrain);
        scene.remove(data.rocks);
        scene.remove(data.hazard);
        if (data.terrain.geometry) data.terrain.geometry.dispose();
        if (data.terrain.material) data.terrain.material.dispose();
        if (data.hazard.geometry) data.hazard.geometry.dispose();
        if (data.hazard.material) data.hazard.material.dispose();
      }
      chunksRef.current.clear();
      // Rebuild around current rover position
      const rv = roverRef.current;
      rebuildChunksRef.current(rv.x, rv.z);
    }
  }, [roughness, craterDensity]);

  const buildRover = useCallback((scene) => {
    const old = scene.getObjectByName("roverGroup");
    if (old) scene.remove(old);
    const g = new THREE.Group();
    g.name = "roverGroup";

    const chassis = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.35, 1.0), new THREE.MeshStandardMaterial({ color: 0xd0d0d0, metalness: 0.7, roughness: 0.25 }));
    chassis.position.set(0, 0.55, 0); chassis.castShadow = true; g.add(chassis);

    const deck = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.15, 0.7), new THREE.MeshStandardMaterial({ color: 0x999999, metalness: 0.8, roughness: 0.2 }));
    deck.position.set(-0.1, 0.83, 0); g.add(deck);

    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.7), new THREE.MeshStandardMaterial({ color: 0xaaaaaa }));
    mast.position.set(0.4, 1.2, 0); g.add(mast);

    const camHead = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.12, 0.15), new THREE.MeshStandardMaterial({ color: 0x333333 }));
    camHead.position.set(0.4, 1.6, 0); g.add(camHead);

    [-0.04, 0.04].forEach(z => {
      const l = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.04, 8), new THREE.MeshStandardMaterial({ color: 0x1a1a2e, metalness: 0.9 }));
      l.rotation.z = Math.PI / 2; l.position.set(0.5, 1.6, z); g.add(l);
    });

    const dish = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xf0e8d0, metalness: 0.3, side: THREE.DoubleSide }));
    dish.position.set(-0.35, 1.15, 0); dish.rotation.x = Math.PI; g.add(dish);

    const antPole = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.4), new THREE.MeshStandardMaterial({ color: 0xcccccc }));
    antPole.position.set(-0.35, 0.95, 0); g.add(antPole);

    [-0.55, 0.55].forEach(z => {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.3), new THREE.MeshStandardMaterial({ color: 0x888888 }));
      arm.position.set(-0.1, 0.82, z * 0.7); g.add(arm);
      const panel = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.02, 0.35), new THREE.MeshStandardMaterial({ color: 0x1a2a55, metalness: 0.4 }));
      panel.position.set(-0.1, 0.85, z); g.add(panel);
    });

    const wg = new THREE.CylinderGeometry(0.18, 0.18, 0.1, 16);
    const wm = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.8 });
    [[-0.6,0.18,0.55],[-0.6,0.18,-0.55],[0,0.18,0.55],[0,0.18,-0.55],[0.6,0.18,0.55],[0.6,0.18,-0.55]].forEach(([x,y,z]) => {
      const w = new THREE.Mesh(wg, wm); w.rotation.x = Math.PI / 2; w.position.set(x, y, z); w.name = "wheel"; g.add(w);
    });

    const cone = new THREE.Mesh(new THREE.ConeGeometry(5, 8, 24, 1, true), new THREE.MeshBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.04, side: THREE.DoubleSide, depthWrite: false }));
    cone.name = "sensorCone"; cone.rotation.x = -Math.PI / 2; cone.position.set(4.5, 0.6, 0); g.add(cone);
    const sr = new THREE.Mesh(new THREE.RingGeometry(4.5, 5, 48), new THREE.MeshBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
    sr.name = "sensorRing"; sr.rotation.x = -Math.PI / 2; sr.position.y = 0.2; g.add(sr);
    const rLight = new THREE.PointLight(0xffeedd, 0.3, 5);
    rLight.position.set(0, 1.5, 0); g.add(rLight);

    scene.add(g);
    return g;
  }, []);

  // ============ INIT ============
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // Clear stale chunk refs from previous mount (React StrictMode)
    chunksRef.current.clear();

    initNoise(seedRef.current);

    let loadStep = 0;
    const stages = [
      [10, "INITIALIZING RENDERER..."], [22, "GENERATING NOISE FIELDS..."],
      [35, "BUILDING TERRAIN CHUNKS..."], [48, "DISTRIBUTING SURFACE ROCKS..."],
      [60, "CONSTRUCTING ROVER SYSTEMS..."], [72, "CALIBRATING MARTIAN ATMOSPHERE..."],
      [84, "SPAWNING DUST PARTICLES..."], [94, "RUNNING DIAGNOSTICS..."],
      [100, "MISSION SYSTEMS ONLINE"],
    ];
    const loadInterval = setInterval(() => {
      if (loadStep < stages.length) { setProgress(stages[loadStep][0]); setLoadingStage(stages[loadStep][1]); loadStep++; }
      else { clearInterval(loadInterval); setTimeout(() => { setLoading(false); setShowOnboarding(true); }, 400); }
    }, 320);

    const w = mount.clientWidth, h = mount.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setSize(w, h); renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    renderer.setClearColor(0x8a5a38);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    // Very subtle fog — just depth cueing, not obscuring
    scene.fog = new THREE.FogExp2(0x9a7050, 0.0018);

    // Mars sky dome
    const skyGeo = new THREE.SphereGeometry(480, 64, 48);
    const skyColors = new Float32Array(skyGeo.attributes.position.count * 3);
    for (let i = 0; i < skyGeo.attributes.position.count; i++) {
      const y = skyGeo.attributes.position.getY(i);
      const t = Math.max(0, Math.min(1, (y + 50) / 530));
      // Horizon: warm salmon-peach | Mid: dusty tan | Zenith: dark rust-brown
      const t2 = t * t;
      skyColors[i * 3] = 0.65 - t2 * 0.42;
      skyColors[i * 3 + 1] = 0.40 - t2 * 0.28;
      skyColors[i * 3 + 2] = 0.30 - t2 * 0.22;
    }
    skyGeo.setAttribute("color", new THREE.BufferAttribute(skyColors, 3));
    scene.add(new THREE.Mesh(skyGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false })));

    // Sun disc in sky
    const sunDisc = new THREE.Mesh(
      new THREE.CircleGeometry(8, 32),
      new THREE.MeshBasicMaterial({ color: 0xffeedd, transparent: true, opacity: 0.6, fog: false })
    );
    sunDisc.position.set(120, 140, 60);
    sunDisc.lookAt(0, 0, 0);
    scene.add(sunDisc);

    // Sun glow
    const sunGlow = new THREE.Mesh(
      new THREE.CircleGeometry(20, 32),
      new THREE.MeshBasicMaterial({ color: 0xffccaa, transparent: true, opacity: 0.12, fog: false })
    );
    sunGlow.position.copy(sunDisc.position);
    sunGlow.lookAt(0, 0, 0);
    scene.add(sunGlow);

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 800);
    camera.position.set(15, 14, 15);

    // Lighting
    const sun = new THREE.DirectionalLight(0xffeedd, 1.4);
    sun.position.set(60, 70, 30);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -60; sun.shadow.camera.right = 60;
    sun.shadow.camera.top = 60; sun.shadow.camera.bottom = -60;
    sun.shadow.camera.near = 1; sun.shadow.camera.far = 200;
    sun.shadow.bias = -0.0005;
    scene.add(sun);
    scene.add(sun.target);
    scene.add(new THREE.AmbientLight(0x775544, 0.5));
    scene.add(new THREE.HemisphereLight(0xddaa77, 0x553322, 0.3));
    const rimLight = new THREE.DirectionalLight(0xff8844, 0.2);
    rimLight.position.set(-30, 10, -30);
    scene.add(rimLight);

    // Chunk container
    // Chunks added directly to scene (no intermediate group)

    // Dust
    const dustCount = 800;
    const dustGeo = new THREE.BufferGeometry();
    const dustPos = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      dustPos[i * 3] = (Math.random() - 0.5) * 300;
      dustPos[i * 3 + 1] = Math.random() * 30;
      dustPos[i * 3 + 2] = (Math.random() - 0.5) * 300;
    }
    dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
    const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0xbb8866, size: 0.05, transparent: true, opacity: 0.18, sizeAttenuation: true, depthWrite: false }));
    scene.add(dust);

    // Path
    const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.7 }));
    pathLine.name = "pathLine"; pathLine.frustumCulled = false; scene.add(pathLine);

    // Marker
    const markerGroup = new THREE.Group(); markerGroup.name = "marker"; markerGroup.visible = false;
    markerGroup.add(new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 10, 8), new THREE.MeshBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.2 })));
    markerGroup.children[0].position.y = 5;
    [0.6, 0.85, 1.1].forEach((r, i) => {
      const ring = new THREE.Mesh(new THREE.RingGeometry(r, r + 0.08, 32), new THREE.MeshBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.3 - i * 0.08, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.15; ring.name = `mr${i}`; markerGroup.add(ring);
    });
    scene.add(markerGroup);

    // Trail
    const trailPts = [];
    const trailLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffaa44, transparent: true, opacity: 0.2 }));
    trailLine.name = "trail"; trailLine.frustumCulled = false; scene.add(trailLine);

    // Search visualization - explored nodes (red-yellow heat map)
    const VIZ_MAX = 50000;
    const exploredGeo = new THREE.BufferGeometry();
    const exploredPositions = new Float32Array(VIZ_MAX * 3);
    const exploredColors = new Float32Array(VIZ_MAX * 3);
    exploredGeo.setAttribute("position", new THREE.BufferAttribute(exploredPositions, 3));
    exploredGeo.setAttribute("color", new THREE.BufferAttribute(exploredColors, 3));
    exploredGeo.setDrawRange(0, 0);
    const exploredPts = new THREE.Points(exploredGeo, new THREE.PointsMaterial({
      size: 1.8, vertexColors: true, transparent: true, opacity: 0.6, sizeAttenuation: true, depthWrite: false,
    }));
    exploredPts.frustumCulled = false;
    exploredPts.name = "exploredViz";
    scene.add(exploredPts);

    // Frontier nodes (bright cyan)
    const frontierGeo = new THREE.BufferGeometry();
    const frontierPositions = new Float32Array(VIZ_MAX * 3);
    frontierGeo.setAttribute("position", new THREE.BufferAttribute(frontierPositions, 3));
    frontierGeo.setDrawRange(0, 0);
    const frontierPts = new THREE.Points(frontierGeo, new THREE.PointsMaterial({
      size: 2.5, color: 0x00ffff, transparent: true, opacity: 0.8, sizeAttenuation: true, depthWrite: false,
    }));
    frontierPts.frustumCulled = false;
    frontierPts.name = "frontierViz";
    scene.add(frontierPts);

    vizMeshesRef.current = { exploredPts, frontierPts, exploredGeo, frontierGeo };

    const roverGroup = buildRover(scene);
    roverGroup.position.set(0, getWorldHeight(0, 0) - 0.12, 0);

    sceneRef.current = { renderer, scene, camera, roverGroup, sun, markerGroup, pathLine, trailLine, trailPts, sunDisc, sunGlow };

    // chunk management
    function updateChunks(rx, rz) {
      const ccx = Math.round(rx / CHUNK_SIZE);
      const ccz = Math.round(rz / CHUNK_SIZE);
      const needed = new Set();
      for (let dx = -VIEW_RANGE; dx <= VIEW_RANGE; dx++) {
        for (let dz = -VIEW_RANGE; dz <= VIEW_RANGE; dz++) {
          needed.add(chunkKey(ccx + dx, ccz + dz));
        }
      }
      // remove old
      for (const [k, data] of chunksRef.current) {
        if (!needed.has(k)) {
          scene.remove(data.terrain);
          scene.remove(data.rocks);
          scene.remove(data.hazard);
          data.terrain.geometry.dispose(); data.terrain.material.dispose();
          data.hazard.geometry.dispose(); data.hazard.material.dispose();
          chunksRef.current.delete(k);
        }
      }
      // add new
      for (const k of needed) {
        if (!chunksRef.current.has(k)) {
          const [cx, cz] = k.split(",").map(Number);
          const terrain = buildChunkMesh(cx, cz);
          const rocks = buildChunkRocks(cx, cz);
          const hazard = buildChunkHazard(cx, cz);
          scene.add(terrain);
          scene.add(rocks);
          scene.add(hazard);
          chunksRef.current.set(k, { terrain, rocks, hazard });
        }
      }
      // update hazard visibility
      for (const [, data] of chunksRef.current) {
        data.hazard.visible = showHazardRef.current;
      }
    }

    updateChunks(0, 0);
    sceneRefForRebuild.current = scene;
    rebuildChunksRef.current = updateChunks;

    // camera orbit
    let camTheta = Math.PI / 4, camPhi = Math.PI / 3.5, camDist = 25;
    let isDrag = false, lastMX = 0, lastMY = 0;
    let targetLook = new THREE.Vector3(0, 0, 0);

    const onWheel = (e) => { e.preventDefault(); camDist = Math.max(4, Math.min(80, camDist + e.deltaY * 0.04)); };
    const onDown = (e) => { if (e.button === 2 || e.button === 1 || (e.button === 0 && e.shiftKey)) { isDrag = true; lastMX = e.clientX; lastMY = e.clientY; e.preventDefault(); }};
    const onMove = (e) => { if (isDrag) { camTheta -= (e.clientX - lastMX) * 0.004; camPhi = Math.max(0.08, Math.min(Math.PI / 2.2, camPhi - (e.clientY - lastMY) * 0.004)); lastMX = e.clientX; lastMY = e.clientY; }};
    const onUp = () => { isDrag = false; };
    const onCtx = (e) => e.preventDefault();

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let clickStart = 0;
    const onCDown = (e) => { if (e.button === 0 && !e.shiftKey) clickStart = Date.now(); };
    const onCUp = (e) => {
      if (e.button !== 0 || e.shiftKey || isDrag || Date.now() - clickStart > 300) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      // intersect all terrain chunks
      const terrains = [];
      for (const [, data] of chunksRef.current) terrains.push(data.terrain);
      const hits = raycaster.intersectObjects(terrains);
      if (hits.length > 0) {
        const p = hits[0].point;
        const rv = roverRef.current;
        setStatus("COMPUTING OPTIMAL PATH...");
        setTimeout(() => {
          if (showAlgoVizRef.current) {
            // Use step-by-step visualization
            astarStepperRef.current = new AStarStepper([rv.x, rv.z], [p.x, p.z]);
            // Clear viz
            const vm = vizMeshesRef.current;
            if (vm.exploredGeo) vm.exploredGeo.setDrawRange(0, 0);
            if (vm.frontierGeo) vm.frontierGeo.setDrawRange(0, 0);
            setStatus("A* SEARCH EXPANDING...");
            setVizStats({ explored: 0, frontier: 0, iters: 0 });
            markerGroup.position.set(p.x, getWorldHeight(p.x, p.z), p.z);
            markerGroup.visible = true;
          } else {
            // Instant pathfind
            const raw = astarPathInstant([rv.x, rv.z], [p.x, p.z]);
            if (raw) {
              const path = smoothPath(smoothPath(raw));
              pathRef.current = path; rv.pathIdx = 0; rv.moving = true; distRef.current = 0;
              setStatus("NAVIGATING — EN ROUTE TO TARGET");
              const pts = path.map(([px, pz]) => new THREE.Vector3(px, getWorldHeight(px, pz) + 0.3, pz));
              pathLine.geometry.dispose(); pathLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
              markerGroup.position.set(p.x, getWorldHeight(p.x, p.z), p.z);
              markerGroup.visible = true;
            } else { setStatus("PATH BLOCKED — NO VIABLE ROUTE"); }
          }
        }, 50);
      }
    };

    const canvas = renderer.domElement;
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("mousedown", (e) => { onDown(e); onCDown(e); });
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseup", (e) => { onUp(); onCUp(e); });
    canvas.addEventListener("contextmenu", onCtx);
    const onKD = (e) => { keysRef.current[e.key.toLowerCase()] = true; };
    const onKU = (e) => { keysRef.current[e.key.toLowerCase()] = false; };
    window.addEventListener("keydown", onKD);
    window.addEventListener("keyup", onKU);
    const onResize = () => { const nw = mount.clientWidth, nh = mount.clientHeight; camera.aspect = nw / nh; camera.updateProjectionMatrix(); renderer.setSize(nw, nh); };
    window.addEventListener("resize", onResize);

    let time = 0, lastTrail = 0, lastChunkUpdate = 0;

    const animate = () => {
      animRef.current = requestAnimationFrame(animate);
      time += 0.016;
      const rv = roverRef.current;
      let currentSpeed = 0;

      // auto nav
      if (rv.moving && pathRef.current) {
        const tgt = pathRef.current[rv.pathIdx];
        const dx = tgt[0] - rv.x, dz = tgt[1] - rv.z;
        const dist = Math.sqrt(dx * dx + dz * dz);
        const spd = 0.1;
        if (dist < 0.5) {
          rv.pathIdx++;
          if (rv.pathIdx >= pathRef.current.length) {
            rv.moving = false; pathRef.current = null; markerGroup.visible = false;
            pathLine.geometry.dispose(); pathLine.geometry = new THREE.BufferGeometry();
            setStatus("TARGET REACHED — AWAITING DIRECTIVE");
          }
        } else {
          const ta = Math.atan2(dx, dz);
          let ad = ta - rv.angle; while (ad > Math.PI) ad -= Math.PI * 2; while (ad < -Math.PI) ad += Math.PI * 2;
          rv.angle += ad * 0.08;
          rv.x += (dx / dist) * spd; rv.z += (dz / dist) * spd;
          distRef.current += spd; currentSpeed = spd / 0.016;
        }
      }

      // manual
      const keys = keysRef.current;
      const ms = 0.18;
      if (keys["a"] || keys["arrowleft"]) rv.angle += 0.035;
      if (keys["d"] || keys["arrowright"]) rv.angle -= 0.035;
      if (keys["w"] || keys["arrowup"]) { rv.x += Math.sin(rv.angle) * ms; rv.z += Math.cos(rv.angle) * ms; distRef.current += ms; currentSpeed = ms / 0.016; }
      if (keys["s"] || keys["arrowdown"]) { rv.x -= Math.sin(rv.angle) * ms * 0.6; rv.z -= Math.cos(rv.angle) * ms * 0.6; distRef.current += ms * 0.6; currentSpeed = ms * 0.6 / 0.016; }

      // Position rover on terrain — offset down so wheel bottoms contact ground
      const rh = getWorldHeight(rv.x, rv.z);
      roverGroup.position.set(rv.x, rh - 0.12, rv.z);

      // Compute terrain normal and clamp tilt to realistic max (~25 degrees)
      const norm = getWorldNormal(rv.x, rv.z);
      const up = new THREE.Vector3(0, 1, 0);
      // Clamp the normal so tilt doesn't exceed max rover angle
      const maxTilt = 0.42; // ~25 degrees in radians
      const tiltAngle = Math.acos(Math.min(1, norm.dot(up)));
      if (tiltAngle > maxTilt) {
        // Lerp the normal toward up to clamp it
        const t = maxTilt / tiltAngle;
        norm.lerp(up, 1 - t).normalize();
      }
      const q = new THREE.Quaternion().setFromUnitVectors(up, norm);
      const yawQ = new THREE.Quaternion().setFromAxisAngle(up, rv.angle - Math.PI / 2);
      roverGroup.quaternion.copy(yawQ).multiply(q);

      roverGroup.children.forEach(c => { if (c.name === "wheel") c.rotation.y += currentSpeed * 0.3; });

      // update chunks every ~0.5s
      if (time - lastChunkUpdate > 0.5) {
        updateChunks(rv.x, rv.z);
        lastChunkUpdate = time;
      }

      // trail
      if (time - lastTrail > 0.2 && currentSpeed > 0) {
        trailPts.push(new THREE.Vector3(rv.x, rh + 0.08, rv.z));
        if (trailPts.length > 1000) trailPts.shift();
        trailLine.geometry.dispose();
        trailLine.geometry = new THREE.BufferGeometry().setFromPoints(trailPts);
        lastTrail = time;
      }

      // sensor/hazard
      const cObj = roverGroup.getObjectByName("sensorCone");
      const rObj = roverGroup.getObjectByName("sensorRing");
      if (cObj) cObj.visible = showSensorRef.current;
      if (rObj) { rObj.visible = showSensorRef.current; rObj.rotation.z = time * 0.5; }

      // shadow follows rover
      sun.position.set(rv.x + 60, 70, rv.z + 30);
      sun.target.position.set(rv.x, rh, rv.z);
      sun.target.updateMatrixWorld();

      // move dust with rover so it's always around
      const dp = dust.geometry.attributes.position;
      for (let i = 0; i < dustCount; i++) {
        let dx2 = dp.getX(i) - rv.x;
        let dz2 = dp.getZ(i) - rv.z;
        if (Math.abs(dx2) > 150) dp.setX(i, rv.x + (Math.random() - 0.5) * 300);
        if (Math.abs(dz2) > 150) dp.setZ(i, rv.z + (Math.random() - 0.5) * 300);
        dp.setX(i, dp.getX(i) + Math.sin(time * 0.15 + i * 0.04) * 0.006 + 0.002);
        let y = dp.getY(i) + 0.002; if (y > 28) y = 0; dp.setY(i, y);
      }
      dp.needsUpdate = true;

      // sun disc follows camera direction
      sunDisc.position.set(rv.x + 120, 140, rv.z + 60);
      sunDisc.lookAt(camera.position);
      sunGlow.position.copy(sunDisc.position);
      sunGlow.lookAt(camera.position);

      // A* Search Visualization stepper
      const stepper = astarStepperRef.current;
      const vm = vizMeshesRef.current;
      if (stepper && !stepper.done && vm.exploredGeo) {
        stepper.step(vizSpeedRef.current);
        // Update explored (closed) points with cost heat map
        const closed = stepper.getClosedPositions();
        const ePos = vm.exploredGeo.attributes.position;
        const eCol = vm.exploredGeo.attributes.color;
        const maxC = Math.max(1, stepper.maxCostSeen);
        const drawE = Math.min(closed.length, 50000);
        for (let i = 0; i < drawE; i++) {
          ePos.setXYZ(i, closed[i].x, getWorldHeight(closed[i].x, closed[i].z) + 0.4, closed[i].z);
          const t = Math.min(1, closed[i].cost / maxC);
          eCol.setXYZ(i, 0.15 + t * 0.85, 0.85 - t * 0.65, 0.1);
        }
        ePos.needsUpdate = true; eCol.needsUpdate = true;
        vm.exploredGeo.setDrawRange(0, drawE);
        // Update frontier (open) points
        const openPts = stepper.getOpenPositions();
        const fPos = vm.frontierGeo.attributes.position;
        const drawF = Math.min(openPts.length, 50000);
        for (let i = 0; i < drawF; i++) {
          fPos.setXYZ(i, openPts[i].x, getWorldHeight(openPts[i].x, openPts[i].z) + 0.5, openPts[i].z);
        }
        fPos.needsUpdate = true;
        vm.frontierGeo.setDrawRange(0, drawF);
        setVizStats({ explored: closed.length, frontier: openPts.length, iters: stepper.iters });
      }
      // When search finishes, extract path and start rover
      if (stepper && stepper.done && !pathRef.current) {
        if (stepper.found) {
          const path = smoothPath(smoothPath(stepper.path));
          pathRef.current = path; rv.pathIdx = 0; rv.moving = true; distRef.current = 0;
          setStatus("PATH FOUND — NAVIGATING");
          const pts = path.map(([px, pz]) => new THREE.Vector3(px, getWorldHeight(px, pz) + 0.3, pz));
          pathLine.geometry.dispose(); pathLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
          // Fade out viz after 3 seconds
          setTimeout(() => {
            if (vm.exploredGeo) vm.exploredGeo.setDrawRange(0, 0);
            if (vm.frontierGeo) vm.frontierGeo.setDrawRange(0, 0);
          }, 3000);
        } else {
          setStatus("PATH BLOCKED — NO VIABLE ROUTE");
          if (vm.exploredGeo) vm.exploredGeo.setDrawRange(0, 0);
          if (vm.frontierGeo) vm.frontierGeo.setDrawRange(0, 0);
        }
        astarStepperRef.current = null;
      }
      // Toggle viz mesh visibility
      if (vm.exploredPts) vm.exploredPts.visible = showAlgoVizRef.current;
      if (vm.frontierPts) vm.frontierPts.visible = showAlgoVizRef.current;

      // HUD
      setRoverPos({ x: rv.x.toFixed(1), z: rv.z.toFixed(1) });
      setRoverAlt(rh.toFixed(1));
      setSlopeVal(getWorldSlope(rv.x, rv.z).toFixed(2));
      setSpeed((currentSpeed * 3.72).toFixed(1));
      setDistance(distRef.current.toFixed(1));

      // Engineering data
      const slopeNow = getWorldSlope(rv.x, rv.z);
      const wheelRadius = 0.18;
      const speedMs = currentSpeed * 3.72;
      const rpm = speedMs > 0 ? (speedMs / (2 * Math.PI * wheelRadius)) * 60 : 0;
      setWheelRpm(rpm.toFixed(0));
      setSuspAngle((Math.atan(slopeNow) * (180 / Math.PI)).toFixed(1));
      setMotorPower(speedMs > 0 ? (speedMs * 2.5 + slopeNow * 8).toFixed(1) : "0.0");
      setTerrainGrade((slopeNow * 100 / 1).toFixed(1));
      setWheelRotAnim(prev => prev + speedMs * 0.06);

      // camera
      targetLook.lerp(new THREE.Vector3(rv.x, rh, rv.z), 0.04);
      camera.position.set(
        targetLook.x + camDist * Math.sin(camTheta) * Math.cos(camPhi),
        targetLook.y + camDist * Math.sin(camPhi),
        targetLook.z + camDist * Math.cos(camTheta) * Math.cos(camPhi)
      );
      camera.lookAt(targetLook);

      if (markerGroup.visible) {
        markerGroup.children.forEach(c => { if (c.name && c.name.startsWith("mr")) c.rotation.z += 0.012; });
        markerGroup.children[0].material.opacity = 0.15 + Math.sin(time * 3) * 0.06;
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      clearInterval(loadInterval);
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("keydown", onKD);
      window.removeEventListener("keyup", onKU);
      window.removeEventListener("resize", onResize);
      chunksRef.current.clear();
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, [buildRover]);

  const regenerate = () => {
    seedRef.current = Math.floor(Math.random() * 100000);
    initNoise(seedRef.current);
    const { scene, roverGroup, trailPts } = sceneRef.current;
    if (!scene) return;
    // clear all chunks from scene
    for (const [, data] of chunksRef.current) {
      scene.remove(data.terrain);
      scene.remove(data.rocks);
      scene.remove(data.hazard);
      if (data.terrain.geometry) data.terrain.geometry.dispose();
      if (data.terrain.material) data.terrain.material.dispose();
      if (data.hazard.geometry) data.hazard.geometry.dispose();
      if (data.hazard.material) data.hazard.material.dispose();
    }
    chunksRef.current.clear();
    roverRef.current = { x: 0, z: 0, angle: 0, pathIdx: 0, moving: false };
    pathRef.current = null; distRef.current = 0; trailPts.length = 0;
    if (roverGroup) { roverGroup.position.set(0, getWorldHeight(0, 0) - 0.12, 0); roverGroup.quaternion.identity(); }
    const pl = scene.getObjectByName("pathLine"); if (pl) { pl.geometry.dispose(); pl.geometry = new THREE.BufferGeometry(); }
    const tl = scene.getObjectByName("trail"); if (tl) { tl.geometry.dispose(); tl.geometry = new THREE.BufferGeometry(); }
    const mg = scene.getObjectByName("marker"); if (mg) mg.visible = false;
    // Clear algorithm viz
    astarStepperRef.current = null;
    const vm = vizMeshesRef.current;
    if (vm.exploredGeo) vm.exploredGeo.setDrawRange(0, 0);
    if (vm.frontierGeo) vm.frontierGeo.setDrawRange(0, 0);
    setVizStats({ explored: 0, frontier: 0, iters: 0 });
    setStatus("NEW TERRAIN — AWAITING DIRECTIVE");
  };

  const P = { background: "rgba(8,4,2,0.82)", border: "1px solid rgba(255,170,80,0.08)", borderRadius: 3, backdropFilter: "blur(10px)" };

  return (
    <div style={{ width: "100%", height: "100vh", background: "#0a0604", fontFamily: "'Courier New', monospace", color: "#e8c8a0", position: "relative", overflow: "hidden", userSelect: "none" }}>
      <div ref={mountRef} style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0 }} />
      {loading && <LoadingScreen progress={progress} stage={loadingStage} />}
      {showOnboarding && !loading && <OnboardingOverlay onStart={() => setShowOnboarding(false)} />}

      <div style={{ position: "absolute", top: 0, left: 0, right: 0, background: "linear-gradient(180deg, rgba(8,4,2,0.75) 0%, transparent 100%)", padding: "10px 16px 35px", zIndex: 10, pointerEvents: "none" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: roverRef.current?.moving ? "#00ffaa" : "#ffaa00", boxShadow: `0 0 5px ${roverRef.current?.moving ? "#00ffaa" : "#ffaa00"}` }} />
              <span style={{ fontSize: 8, letterSpacing: 3.5, opacity: 0.25 }}>ORION SUBSYSTEM v1.0</span>
            </div>
            <div style={{ fontSize: 17, fontWeight: "bold", letterSpacing: 5, color: "#ffd4a0", marginTop: 1 }}>ATHENA</div>
          </div>
          <div style={{ ...P, padding: "5px 11px", display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#cc4422", boxShadow: "0 0 5px #cc4422" }} />
            <div>
              <div style={{ fontSize: 11, letterSpacing: 2.5, color: "#ff8855", fontWeight: "bold" }}>MARS</div>
              <div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.3 }}>{PLANET.subtitle}</div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ position: "absolute", top: 68, left: 10, width: 140, zIndex: 10, ...P, padding: 9 }}>
        <div style={{ fontSize: 7, letterSpacing: 2, color: "#ffaa44", marginBottom: 6 }}>TELEMETRY</div>
        {[["POS X", `${roverPos.x} m`], ["POS Z", `${roverPos.z} m`], ["ALT", `${roverAlt} m`], ["SLOPE", slopeVal], ["SPEED", `${speed} m/s`], ["DIST", `${distance} m`]].map(([l, v]) => (
          <div key={l} style={{ marginBottom: 2 }}><div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.22 }}>{l}</div><div style={{ fontSize: 10, letterSpacing: 0.5 }}>{v}</div></div>
        ))}
      </div>

      <div style={{ position: "absolute", top: 300, left: 10, width: 140, zIndex: 10, ...P, padding: 9 }}>
        <div style={{ fontSize: 7, letterSpacing: 2, color: "#ffaa44", marginBottom: 6 }}>ENVIRONMENT</div>
        {[["GRAVITY", PLANET.gravity], ["ATMO", PLANET.atmosphere], ["TEMP", PLANET.temp], ["WIND", PLANET.windSpeed], ["MISSION", PLANET.sol]].map(([l, v]) => (
          <div key={l} style={{ marginBottom: 2 }}><div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.22 }}>{l}</div><div style={{ fontSize: 10, letterSpacing: 0.5 }}>{v}</div></div>
        ))}
      </div>

      <div style={{ position: "absolute", top: 68, right: 10, width: 170, zIndex: 10, ...P, padding: 9 }}>
        <div style={{ fontSize: 7, letterSpacing: 2, color: "#ffaa44", marginBottom: 7 }}>MISSION CONTROLS</div>
        <div style={{ marginBottom: 7 }}>
          <div style={{ fontSize: 7, opacity: 0.35, marginBottom: 3 }}>OVERLAYS</div>
          <div style={{ display: "flex", gap: 3 }}>
            <button onClick={() => setShowSensor(!showSensor)} style={{ flex: 1, padding: "3px 0", fontSize: 7, letterSpacing: 1, background: showSensor ? "rgba(0,255,170,0.08)" : "rgba(255,255,255,0.015)", border: showSensor ? "1px solid rgba(0,255,170,0.22)" : "1px solid rgba(255,255,255,0.03)", color: showSensor ? "#00ffaa" : "#554433", borderRadius: 2, cursor: "pointer" }}>SENSOR</button>
            <button onClick={() => setShowHazard(!showHazard)} style={{ flex: 1, padding: "3px 0", fontSize: 7, letterSpacing: 1, background: showHazard ? "rgba(255,170,0,0.08)" : "rgba(255,255,255,0.015)", border: showHazard ? "1px solid rgba(255,170,0,0.22)" : "1px solid rgba(255,255,255,0.03)", color: showHazard ? "#ffaa00" : "#554433", borderRadius: 2, cursor: "pointer" }}>HAZARD</button>
          </div>
        </div>
        <div style={{ marginBottom: 6 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, opacity: 0.35, marginBottom: 2 }}><span>ROUGHNESS</span><span style={{ color: "#ffd4a0" }}>{roughness.toFixed(1)}</span></div>
          <input type="range" min="0.3" max="2" step="0.1" value={roughness} onChange={e => setRoughness(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#ffaa00", height: 2 }} />
        </div>
        <div style={{ marginBottom: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, opacity: 0.35, marginBottom: 2 }}><span>CRATERS</span><span style={{ color: "#ffd4a0" }}>{craterDensity.toFixed(1)}</span></div>
          <input type="range" min="0" max="1" step="0.1" value={craterDensity} onChange={e => setCraterDensity(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#ffaa00", height: 2 }} />
        </div>
        <button onClick={regenerate} style={{ width: "100%", padding: "4px 0", fontSize: 7, letterSpacing: 3, background: "rgba(255,170,0,0.04)", border: "1px solid rgba(255,170,0,0.15)", color: "#ffaa00", borderRadius: 2, cursor: "pointer", marginBottom: 4 }}>REGENERATE</button>
        <button onClick={() => setShowOnboarding(true)} style={{ width: "100%", padding: "4px 0", fontSize: 7, letterSpacing: 2, background: "rgba(255,255,255,0.01)", border: "1px solid rgba(255,255,255,0.03)", color: "#554433", borderRadius: 2, cursor: "pointer" }}>VIEW CONTROLS</button>
      </div>

      {/* ALGORITHM VISUALIZATION PANEL */}
      <div style={{ position: "absolute", top: 340, right: 10, width: 170, zIndex: 10, ...P, padding: 9 }}>
        <div style={{ fontSize: 7, letterSpacing: 2, color: "#00ccff", marginBottom: 7 }}>PATHFINDING AI</div>

        <div style={{ marginBottom: 7 }}>
          <div style={{ fontSize: 7, opacity: 0.4, marginBottom: 3 }}>SEARCH VISUALIZATION</div>
          <button onClick={() => setShowAlgoViz(!showAlgoViz)} style={{
            width: "100%", padding: "3px 0", fontSize: 7, letterSpacing: 1,
            background: showAlgoViz ? "rgba(0,204,255,0.1)" : "rgba(255,255,255,0.015)",
            border: showAlgoViz ? "1px solid rgba(0,204,255,0.25)" : "1px solid rgba(255,255,255,0.03)",
            color: showAlgoViz ? "#00ccff" : "#554433", borderRadius: 2, cursor: "pointer",
          }}>{showAlgoViz ? "VIZ ON — CLICK TO SET TARGET" : "VIZ OFF — INSTANT PATH"}</button>
        </div>

        <div style={{ marginBottom: 7 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 7, opacity: 0.35, marginBottom: 2 }}>
            <span>STEPS/FRAME</span><span style={{ color: "#00ccff" }}>{vizSpeed}</span>
          </div>
          <input type="range" min="5" max="300" step="5" value={vizSpeed} onChange={e => setVizSpeed(parseInt(e.target.value))} style={{ width: "100%", accentColor: "#00ccff", height: 2 }} />
        </div>

        {/* Live search stats */}
        <div style={{ marginBottom: 4 }}>
          {[
            ["EXPLORED", vizStats.explored, "#44dd66"],
            ["FRONTIER", vizStats.frontier, "#00ccff"],
            ["ITERATIONS", vizStats.iters, "#e8c8a0"],
          ].map(([label, val, color]) => (
            <div key={label} style={{ marginBottom: 2 }}>
              <div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.22 }}>{label}</div>
              <div style={{ fontSize: 9, letterSpacing: 0.5, color, fontFamily: "monospace" }}>{val.toLocaleString()}</div>
            </div>
          ))}
        </div>

        {/* Heat map legend */}
        <div style={{ marginBottom: 4 }}>
          <div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.22, marginBottom: 3 }}>COST HEAT MAP</div>
          <div style={{ height: 6, borderRadius: 1, background: "linear-gradient(90deg, #22dd22, #88cc11, #ddaa00, #dd4411)" }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 5, opacity: 0.25, marginTop: 1 }}>
            <span>LOW COST</span><span>HIGH COST</span>
          </div>
        </div>

        <div style={{ fontSize: 5.5, lineHeight: 1.5, opacity: 0.2, letterSpacing: 0.3, fontFamily: "monospace" }}>
          A* explores nodes by estimated total cost (g + h). Green = cheap flat terrain. Red = expensive steep slopes. Cyan frontier = nodes queued for evaluation.
        </div>
      </div>

      {showHazard && (
        <div style={{ position: "absolute", bottom: 48, left: 10, zIndex: 10, ...P, padding: 8 }}>
          <div style={{ fontSize: 7, letterSpacing: 1.5, opacity: 0.25, marginBottom: 4 }}>HAZARD MAP</div>
          {[{ c: "#22cc44", l: "TRAVERSABLE" }, { c: "#ffaa00", l: "CAUTION" }, { c: "#ff3333", l: "IMPASSABLE" }].map(({ c, l }) => (
            <div key={l} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 1 }}>
              <div style={{ width: 5, height: 5, borderRadius: 1, background: c, flexShrink: 0 }} />
              <span style={{ fontSize: 6, letterSpacing: 1, opacity: 0.45 }}>{l}</span>
            </div>
          ))}
        </div>
      )}

      {/* 3D ENGINEERING VIEWPORT */}
      {showEngPanel && (
        <EngPanel3D
          speed={parseFloat(speed)}
          suspAngle={parseFloat(suspAngle)}
          wheelRpm={wheelRpm}
          motorPower={motorPower}
          terrainGrade={terrainGrade}
          wheelRotAnim={wheelRotAnim}
          onHide={() => setShowEngPanel(false)}
          roverX={parseFloat(roverPos.x)}
          roverZ={parseFloat(roverPos.z)}
        />
      )}

      {!showEngPanel && (
        <button onClick={() => setShowEngPanel(true)} style={{
          position: "absolute", bottom: 32, right: 10, zIndex: 10,
          ...P, padding: "5px 10px", fontSize: 7, letterSpacing: 2,
          color: "#665544", cursor: "pointer", border: "1px solid rgba(255,170,80,0.08)", background: "rgba(8,4,2,0.82)",
        }}>ENGINEERING</button>
      )}

      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "linear-gradient(0deg, rgba(8,4,2,0.75) 0%, transparent 100%)", padding: "28px 16px 7px", zIndex: 10, pointerEvents: "none" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <div>
            <div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.18, marginBottom: 1 }}>STATUS</div>
            <div style={{ fontSize: 9, letterSpacing: 2, color: status.includes("BLOCKED") ? "#ff4444" : "#00ffaa" }}>{status}</div>
          </div>
          <div style={{ fontSize: 6, letterSpacing: 1.5, opacity: 0.1 }}>SWAN YI HTET & DAVID YOUNG</div>
        </div>
      </div>
    </div>
  );
}