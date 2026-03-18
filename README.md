# ATHENA

**Autonomous Terrain & Hazard Exploration Navigation Agent**

An ORION Subsystem

---

ATHENA is an interactive 3D Mars rover autonomy simulator built for the browser. It demonstrates real-time autonomous navigation, terrain hazard classification, and pathfinding algorithm visualization on procedurally generated, infinite Martian terrain.

## Live Demo

[Launch ATHENA](https://shtet100.github.io/ATHENA/)

## Features

**Infinite Procedural Mars Terrain**
Chunk-based terrain generation using layered fractal Brownian motion noise with deterministic crater placement. The terrain extends infinitely in every direction with no boundaries — drive as far as you want.

**Autonomous Navigation with A\* Pathfinding**
Click anywhere on the terrain to set a waypoint. The rover computes and executes a slope-aware optimal path in real time, avoiding steep gradients and impassable hazards.

**Algorithm Visualization**
Watch the A\* search algorithm expand across the terrain in real time. Explored nodes are rendered as a green-to-red cost heat map (green = low traversal cost, red = high slope penalty). Cyan frontier dots show the search boundary. Adjustable speed control from slow-motion analysis to rapid expansion.

**Hazard Classification Overlay**
Toggle a terrain-wide hazard map classifying every point as traversable (green), caution (yellow), or impassable (red) based on slope analysis.

**3D Rover Engineering Viewport**
An interactive 3D schematic of the rover with orbit and zoom controls. The viewport samples real terrain geometry from the main simulation, showing a ghost wireframe of the actual ground beneath the rover. Live telemetry displays wheel RPM, motor power, suspension angle, and terrain grade.

**Martian Environment**
Realistic sky gradient, atmospheric dust particles, sun disc, exponential depth fog, and environment data (gravity, atmospheric pressure, temperature, wind speed) matching Mars surface conditions.

## Controls

| Input | Action |
|-------|--------|
| Left Click | Set navigation target |
| Right Drag | Orbit camera |
| Shift + Drag | Orbit camera (alt) |
| Scroll | Zoom in/out |
| W / S | Manual drive forward/reverse |
| A / D | Manual steer left/right |

## Tech Stack

- React
- Three.js (3D rendering)
- Vite (build tooling)
- Custom A\* pathfinding with slope-based cost functions
- Procedural terrain generation (fBm noise, deterministic crater hashing)
- Chunk-based infinite world system

## The ORION Connection

ATHENA is a subsystem of [ORION](https://github.com/shtet100) — a personal AI assistant system designed as a brain for future robotics platforms. If ORION is the brain, ATHENA is how it navigates. Future phases will expand ATHENA to other planetary environments (Venus, Europa, Titan) with adaptive autonomy that learns from each world's unique constraints.

## Getting Started

```bash
git clone https://github.com/shtet100/ATHENA.git
cd ATHENA
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Authors

**Swan Yi Htet** — Columbia University, Fu Foundation School of Engineering and Applied Science
Terrain physics, rover dynamics, sensor modeling, algorithm design

**David Young** — University of Pennsylvania
Pathfinding architecture, search visualization, perception logic

## License

MIT