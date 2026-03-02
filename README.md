# SharpView · Monocular 3DGS

**A single-image → 3D Gaussian Splat pipeline built on [apple/ml-sharp](https://github.com/apple/ml-sharp)**

Live: [project-sharp-view.vercel.app](https://project-sharp-view.vercel.app) · Repo: [github.com/projectmehari/SharpView](https://github.com/projectmehari/SharpView)

---

## What is SHARP?

SHARP (Single-image Human-centric Appearance Reconstruction and Point cloud) is Apple's feedforward model that reconstructs a 3D Gaussian Splat from a single photograph in sub-second time on GPU. The original research release is a Python inference script with no interactive viewer or web interface.

---

## Original vs. This Implementation

### Original (apple/ml-sharp)

| Aspect | Original |
|---|---|
| Interface | Python CLI only |
| Input | Local image file via command line |
| Output | `.ply` file written to disk |
| Viewer | None — requires external tools (MeshLab, SuperSplat, etc.) |
| Inference | Local GPU required |
| Sharing | Manual file transfer |
| Mobile | Not applicable |

### This Implementation (SharpView)

| Aspect | SharpView |
|---|---|
| Interface | Single-file web app (`sharpview.html`) |
| Input | Drag-and-drop or click-to-upload, any JPG/PNG/WEBP |
| Output | Interactive 3D point cloud rendered in-browser via Three.js |
| Viewer | Built-in WebGL viewer with orbit controls, zoom, WASD movement |
| Inference | Modal serverless GPU (A10G · CUDA 12.4) — no local GPU needed |
| Sharing | One-click shareable links via Modal Dict storage (`?share=<id>`) |
| Mobile | Fully responsive with touch controls, bottom sheet share UI |

---

## Architecture

```
Browser (sharpview.html)
    │
    ├── Upload image → POST to Modal endpoint
    │                      │
    │              Modal (A10G GPU)
    │              └── SHARP inference → PLY buffer
    │                      │
    ├── Receive PLY ←───────┘
    │
    ├── Parse PLY (ASCII + Binary support)
    ├── Three.js BufferGeometry → Points mesh
    └── Render loop (WebGL)
```

---

## Features Added Over Original

### Viewer
- **Three.js WebGL renderer** — real-time interactive point cloud, no external tools required
- **Orbit controls** — mouse drag to rotate, scroll to zoom, WASD/arrow keys to pan
- **Render modes** — Points (default), Wire (micro-point), Heat (density colormap)
- **Heat mode** — bins points into a 3D spatial grid, counts local density, maps to an inferno colormap (sparse → deep purple, dense → orange → white)
- **Grid floor** — `THREE.GridHelper` ground reference plane
- **Fog effect** — depth fog toggle for atmosphere
- **Axes auto-rotate / Still** — animated orbit or locked camera
- **Density slider** — subsample the point cloud (1k → full) for performance
- **Zoom buttons** — `+` / `−` in addition to scroll

### Depth Map
- **Z-depth canvas** — client-side depth projection rendered from PLY positions using a plasma colormap
- **Collapsible panel** — expand/collapse with animated height transition
- **Far/Close legend** — colormap legend strip

### Export
- **PLY** — re-download the raw output
- **OBJ** — vertex-only Wavefront OBJ
- **PTS** — XYZ + RGB point cloud format
- **PNG screenshot** — captures the current WebGL canvas

### Sharing
- **Share links** — splat stored server-side in Modal Dict, retrieved via `?share=<id>` URL param
- **Desktop**: toast notification with copyable URL
- **Mobile**: bottom sheet with full-width URL input and Copy button, slides up from screen edge with backdrop dismiss

### Onboarding
- **Embedded sample image** — `monsieur-la-flute.jpg` (stilt-walking flutist, Montréal metro) base64-encoded into the HTML; populates on first load so new visitors see something immediately
- **Sample badge** — `⬡ sample` overlay on the preview
- **"or try your own" CTA** — dashed upload button with divider appears below the sample, disappears when user picks their own photo

### Mobile
- **Stacked layout** — panels stack vertically on ≤768px
- **Two-row controls bar** — view mode buttons (Row 1, horizontally scrollable) + action buttons (Row 2, full-width flush segmented bar)
- **Touch controls** — single-finger orbit, two-finger pinch-to-zoom
- **Status strip** — status badge, GitHub link, Online indicator shown below header (header badges hidden on mobile)
- **iOS safe area** — `env(safe-area-inset-bottom)` applied to share sheet

### Infrastructure
- **Vercel** — static hosting, zero config
- **Modal** — serverless GPU inference endpoint (`sharpview-serve`), scales to zero when idle
- **`scaledown_window`** — replaces deprecated `container_idle_timeout`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS — single file, no build step |
| 3D rendering | Three.js r128 |
| Fonts | IBM Plex Mono + Instrument Serif (Google Fonts) |
| Inference backend | Modal (Python, serverless A10G GPU) |
| Model | apple/ml-sharp |
| Storage | Modal Dict (PLY share cache) |
| Hosting | Vercel |

---

## File Structure

```
SharpView/
└── sharpview.html     # Entire frontend — ~2,300 lines
                       # CSS, HTML, JS, base64 sample image all in one file
```

The backend lives in a separate Modal deployment (`app.py`) and is not included in this repo.

---

## Known Limitations

- **Single image only** — SHARP is monocular; multi-view reconstruction is not supported
- **`SAMPLE_SHARE_ID`** — the "Load sample splat" button is disabled until you run inference once on the sample image and paste the returned share ID into the constant at the top of the script
- **PLY only** — the viewer parses raw PLY; full Gaussian Splat rendering (with opacity, covariance, spherical harmonics) is not yet implemented — points are rendered as colored vertices
- **Share link expiry** — Modal Dict entries are not currently set to expire; storage should be monitored for large deployments
