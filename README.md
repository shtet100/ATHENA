# ATHENA

**Autonomous Terrain & Hazard Exploration Navigation Agent**

---

ATHENA is a real-time 3D rover autonomy simulator with multi-algorithm pathfinding, infinite procedural terrain, and multi-planetary deployment. It serves as the navigation layer of [ORION](https://github.com/shtet100), a modular robotics ecosystem designed to control physical robot hardware.

The entire system runs in-browser. No backend required.

> **[Try the live demo →](https://shtet100.github.io/ATHENA/)**

---

## What It Does

<div align="center">

https://github.com/user-attachments/assets/00659384-aaff-48d5-878a-f43558738320

*Full demonstration: multi-algorithm pathfinding, manual driving, terrain controls, and planetary environment switching.*

</div>

A rover is placed on an infinite procedurally generated planetary surface. The operator clicks anywhere on the terrain and the rover plans a path using one of four selectable algorithms, visualizing the search in real-time. The terrain extends infinitely in every direction with no boundaries. Slopes, craters, and rocks are all generated deterministically from seeded noise.

The operator can switch between Mars, Venus, Europa, and Titan. Each planet rebuilds the entire terrain, sky, lighting, fog, and surface colors to match its real physical conditions.

---

## Pathfinding Algorithms

ATHENA implements four search algorithms. Each uses the same slope-based cost function but explores the search space in fundamentally different ways. The operator selects an algorithm, clicks a target, and watches the search expand in real-time with a step-by-step visualizer.

### A* Search

The default. Explores by estimated total cost: actual cost traveled (g) plus a heuristic estimate to the goal (h). This produces a focused beam that expands toward the target, exploring far fewer nodes than uninformed search.

The heuristic is Euclidean distance. The cost function penalizes slope: flat terrain costs 1, moderate slopes cost up to 15x, and slopes above 35° cost 200x, effectively creating impassable barriers.

### Dijkstra's Algorithm

A* without the heuristic. Explores by actual cost only, producing a uniform circular flood outward from the rover. Guaranteed to find the optimal path, but explores significantly more nodes than A* because it has no directional bias.

The visual difference is immediate: where A* expands in a narrow beam, Dijkstra floods outward in concentric rings.

### RRT (Rapidly-exploring Random Trees)

A probabilistic planner. Instead of expanding on a grid, RRT randomly samples points in the world, finds the nearest existing tree node, and extends a branch toward the sample. A 15% goal bias ensures the tree grows toward the target rather than uniformly.

The visualization is a branching tree structure, completely different from the grid-based expansion of A* and Dijkstra. RRT handles high-dimensional spaces well and produces non-optimal but viable paths quickly.

Branches are rejected if they land on slopes above 35°, producing natural avoidance of hazardous terrain without explicit obstacle modeling.

### D* Lite

Searches backward from the goal to the rover. The expansion wave originates at the target and floods toward the rover's position, producing a visually reversed search pattern compared to A*.

D* Lite is designed for replanning: when the environment changes mid-traverse, only the affected portion of the search needs to be recomputed. This makes it the natural choice for dynamic environments where terrain data updates as the rover drives.

---

## Terrain Generation

The terrain is infinite. There are no edges, no loading screens, no prebuilt maps. The rover can drive in any direction forever.

### Chunk System

The world is divided into 80-unit chunks. A 7×7 grid of chunks (49 total) is maintained around the rover at all times. As the rover moves, chunks behind it are unloaded and new chunks ahead are generated. Each chunk contains terrain geometry, surface rocks, and a hazard overlay.

Chunks are keyed by grid coordinates and stored in a Map. Generation and disposal happen every 0.5 seconds based on the rover's current position.

### Height Function

Terrain height is computed from a continuous world-space function. Three layers of fractal Brownian motion (fBm) noise at different frequencies are blended together, each sampled from independent 256×256 seeded noise textures. A ridge noise layer adds sharp geological features.

On top of the base terrain, deterministic craters are placed using spatial cell hashing. The world is divided into 50-unit cells, each cell's hash determines how many craters it contains and where they sit. Crater geometry uses a parabolic depression with a raised rim.

Every height query — whether for terrain mesh vertices, rover ground contact, pathfinding cost, or the engineering viewport — calls the same `getWorldHeight()` function. There is one source of truth.

### Surface Rocks

Each chunk scatters 8-20 rocks using seeded random placement. Rock shapes are dodecahedra, octahedra, and tetrahedra with randomized scale and rotation. Rock density scales with crater density. Colors adapt to the current planet.

---

## Terrain Analysis

Toggling the TERRAIN overlay renders a continuous slope gradient across the entire visible terrain. Every vertex is scored by its slope angle and colored on a smooth ramp:

- **Green (low opacity)** — Flat terrain, safe traversal, slope below 10°
- **Yellow-orange (moderate opacity)** — Moderate slopes, caution zone, 10-20°
- **Red (high opacity)** — Steep terrain, hazardous or impassable, above 30°

The overlay sits 0.15 units above the terrain surface with transparency, so the underlying ground texture is still visible beneath. Opacity increases with hazard severity, making dangerous zones visually prominent without obscuring safe terrain.

---

## Multi-Waypoint Mission Planning

The operator can plan multi-stop missions:

1. Click **MULTI-WAYPOINT** to enter planning mode
2. Click locations on the terrain to drop waypoints (each marked with an amber beacon)
3. Click **EXECUTE** to plan and chain paths between all waypoints

The system plans each leg sequentially using the selected algorithm: rover → waypoint 1 → waypoint 2 → ... → waypoint N. Paths are concatenated into a single continuous route. The rover traverses the full mission automatically.

Mission stats display total distance and waypoint count. If any leg is blocked by impassable terrain, the mission reports the failure before the rover moves.

---

## Planetary Environments

ATHENA deploys to four planetary bodies. Each changes the entire simulation:

| | Mars | Venus | Europa | Titan |
|:---|:---|:---|:---|:---|
| **Gravity** | 3.72 m/s² | 8.87 m/s² | 1.31 m/s² | 1.35 m/s² |
| **Atmosphere** | 0.6 kPa CO₂ | 9200 kPa CO₂ | ~0 Pa | 146.7 kPa N₂ |
| **Surface Temp** | −60°C | 462°C | −160°C | −179°C |
| **Terrain** | Rocky, cratered | Flat volcanic | Icy ridges | Smooth dunes |
| **Visibility** | Moderate dust | Thick haze | Crystal clear | Dense haze |
| **Craters** | Dense | Very sparse | Moderate | Almost none |

Switching planets rebuilds all terrain chunks with new height scales, surface colors, rock colors, crater density, roughness, sky gradients, fog density, sun color and intensity, and dust particle colors. The rover resets to origin on each switch. Trail, path, waypoints, and search visualizations are all cleared.

---

## Engineering Viewport

A separate Three.js renderer draws a detailed 3D rover model with orbit and zoom controls. The viewport shows:

- **Wheel spin** synced to the rover's actual speed
- **Chassis tilt** synced to the terrain slope beneath the rover
- **Wireframe terrain patch** sampling the real `getWorldHeight()` function around the rover's position
- **Telemetry readout**: wheel RPM, motor power, suspension angle, terrain grade

The viewport has independent camera controls. The operator can orbit and zoom the engineering view without affecting the main simulation camera.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     ATHENA v2.0                          │
│              React + Three.js · Browser                  │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │  Terrain    │  │  Pathfinding │  │   Planetary    │   │
│  │  Engine     │  │  Engine      │  │   Environment  │   │
│  │             │  │              │  │                │   │
│  │ fBm noise   │  │ A*           │  │ Mars           │   │
│  │ Crater hash │  │ Dijkstra     │  │ Venus          │   │
│  │ Chunk mgmt  │  │ RRT          │  │ Europa         │   │
│  │ Slope calc  │  │ D* Lite      │  │ Titan          │   │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘   │
│         │                │                  │            │
│  ┌──────▼────────────────▼──────────────────▼────────┐   │
│  │              Unified World Model                  │   │
│  │    getWorldHeight() · getWorldSlope() · chunks    │   │
│  └─────────────────────┬─────────────────────────────┘   │
│  ┌─────────────────────▼─────────────────────────────┐   │
│  │            Visualization Layer                    │   │
│  │   Cost heat map · Frontier · Trail · Waypoints    │   │
│  └─────────────────────┬─────────────────────────────┘   │
│  ┌─────────────────────▼─────────────────────────────┐   │
│  │     Rover Simulation · Engineering Viewport       │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## Part of ORION

ATHENA is one piece of a larger system:

```
ORION (Brain · Intelligence Core)
 ├── ATHENA    - Navigation · Procedural terrain · Multi-algorithm pathfinding
 ├── BROTEUS   - Perception · Grasp intelligence · Gestures & animations
 ├── CHIRON    - Motor cortex · ROS 2 bridge · Hardware abstraction
 ├── DAEDALUS  - Self-calibrating physics discovery (SINDy)
 └── RL Pipeline - PPO/SAC in sim · ONNX deployment at 50-200 Hz
```

**ATHENA navigates. BROTEUS sees. ORION decides. CHIRON moves. DAEDALUS calibrates.**

---

## Setup

```bash
# Clone
git clone https://github.com/shtet100/ATHENA.git
cd ATHENA

# Install
npm install

# Run
npm run dev
```

Open `http://localhost:5173` in a browser.

### Deploy to GitHub Pages

```bash
npm run build
npx gh-pages -d dist
```

---

## Tech Stack

| | |
|:---|:---|
| **Rendering** | Three.js (WebGL) |
| **Framework** | React 18 + Vite |
| **Pathfinding** | A*, Dijkstra, RRT, D* Lite |
| **Terrain** | Seeded fBm noise (3-layer), deterministic crater hashing |
| **Chunks** | 7×7 dynamic grid, 80-unit chunks, 64×64 vertex resolution |
| **Visualization** | Pre-allocated 50K point buffers, cost heat map, frontier rendering |
| **Planets** | Mars, Venus, Europa, Titan (full environment swap) |
| **Engineering** | Separate Three.js renderer, real-time telemetry sync |

---

## Design Decisions

**One height function.** Every system that needs terrain height — mesh generation, rover contact, pathfinding cost, hazard overlay, engineering viewport — calls the same `getWorldHeight()`. There is no duplication, no drift, no inconsistency.

**Infinite terrain with no loading.** The chunk system generates and disposes terrain on the fly. The rover never hits a wall. The world extends as far as the operator wants to drive.

**Algorithm-agnostic visualization.** All four pathfinding algorithms implement the same stepper interface: `step()`, `getClosedPositions()`, `getOpenPositions()`. The visualization layer doesn't know or care which algorithm is running. Swap the algorithm, the viz just works.

**Deterministic everything.** Terrain, craters, and rocks are all generated from seeded random functions. The same seed produces the same world every time. Changing the seed produces a completely different planet.

**Planet parameters, not planet code.** Each planet is a data object, not a separate code path. Mars and Europa run identical terrain generation logic with different parameter values. Adding a new planet is adding one object to a dictionary.

**Slope is the universal cost.** Pathfinding, hazard classification, terrain analysis, and traversability scoring all derive from the same `getWorldSlope()` function. One physical quantity drives all navigation intelligence.

---

*Built by Swan Yi Htet & David Young.*
