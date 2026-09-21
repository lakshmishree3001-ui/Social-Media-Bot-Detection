# GRAPHWARDEN // DESIGN SYSTEM & AESTHETIC COMPLIANCE
**Classification:** Product Design Specification  
**Theme:** Forensic Case File / Archival Intelligence Dossier  
**Status:** Certified Zero-Fluff Standard

---

## 1. Color Palette Tokens

| Token | Hex Value | Semantic Meaning |
| :--- | :--- | :--- |
| `--paper` | `#F2EDE3` | Archival cream paper base surface |
| `--paper-deep` | `#E7E0D2` | Secondary surface, table headers, metadata cards |
| `--paper-hover` | `#DFD7C7` | Interactive hover state |
| `--ink` | `#191713` | Deep near-black archival ink typography and primary rules |
| `--ink-muted` | `#665F52` | Secondary metadata and timestamp typography |
| `--ink-veil` | `#2A2721` | Deep charcoal dark canvas background for interactive graphs |
| `--rule` | `#8C8578` | Neutral warm gray for hairline table and panel borders |
| `--rule-light` | `#D1C8B6` | Secondary grid separation borders |
| `--stamp` | `#A93B26` | Terracotta / vermilion stamp for automated bot verdicts & alerts |
| `--stamp-bg` | `#F8ECE8` | Low-opacity stamp tint for alert backgrounds |
| `--ledger` | `#2C5D4F` | Muted forest green for authentic human accounts & verification |
| `--ledger-bg` | `#EAF3EF` | Low-opacity ledger tint for human status |
| `--signal` | `#C4892B` | Amber ochre for graph canvas edges and graphics |
| `--signal-text` | `#A8731F` | WCAG AA compliant contrast (> 4.5:1) amber for text on `--paper` |

---

## 2. Typography Hierarchy

* **Display Font:** `Fraunces` (Optical sizing, weights 600, 700). Used for major case file titles and branding.
* **Body Font:** `IBM Plex Sans` (Weights 400, 500, 600). Used for explanatory text and legal descriptions.
* **Data Font:** `IBM Plex Mono` (Weights 400, 500, 600, tabular-nums). Used for node indices, probabilities, timestamps, and metric values.
* **Prohibited Fonts:** Inter, Geist, Space Grotesk, Roboto, Arial, System fonts.

---

## 3. Geometric Hard Constraints

* **Sharp Corners Rule:** `border-radius: 0` on all cards, tables, inputs, buttons, and panels.
* **Explicit Exceptions:**
  1. `border-radius: 2px` on the verdict classification pill stamp.
  2. `border-radius: 50%` on graph topological nodes in Cytoscape canvases.
* **Single Elevation Shadow Rule:** Exactly one box shadow exists in the entire application:
  `--shadow-command-bar: 0 4px 12px rgba(25, 23, 19, 0.08)` for the sticky command bar on scroll. No other drop shadows, inner shadows, or glow effects exist anywhere.
* **Zero Gradients Rule:** No CSS gradients anywhere. All surfaces and borders use solid flat colors.
* **Zero Emoji Rule:** Zero emoji anywhere in the UI or codebase. Custom inline SVGs provide all iconography.

---

## 4. The 30 Banned Patterns Audit Table

| # | Prohibited Pattern | Replaced With in Graphwarden |
| :--- | :--- | :--- |
| 1 | Lucide/Feather icon packs | 7 custom archival SVG marks (`IconNodeRing`, `IconEdgeRay`, `IconAttentionCone`, etc.) |
| 2 | Emoji anywhere in UI | Named probabilities and textual forensic tags |
| 3 | Purple/violet gradients | Solid `--ink` and `--paper` tones |
| 4 | Glassmorphism / backdrop blur | Solid `--paper-deep` with 1px solid `--rule` borders |
| 5 | Em dashes (—) | Clean forward slashes (`//`) or hyphens |
| 6 | Rounded buttons (`border-radius: 8px+`) | Sharp 90-degree corners (`border-radius: 0`) |
| 7 | Inter / Geist fonts | `Fraunces` (display), `IBM Plex Sans`, `IBM Plex Mono` |
| 8 | Bento grid layouts | Precise forensic split-screen consoles and tabular layouts |
| 9 | Fake terminal chrome with red/yellow/green dots | Clean archival metadata bars |
| 10 | Fake social proof / testimonials | Empirical test set evaluation benchmarks |
| 11 | Rainbow chips & random tag colors | Calibrated 3-class semantic stamping (`--stamp`, `--ledger`, `--signal`) |
| 12 | Drop shadows on cards | Hairline borders (`1px solid var(--rule)`) |
| 13 | Floating pill navigation | Archival tab strip with solid active states |
| 14 | Marketing buzzwords ("Supercharge", "Unleash") | Exact scientific prose ("Multimodal Topological Inference") |
| 15 | Hardcoded mock JSON arrays | Real rows queried directly from database engine |
| 16 | Skeleton shimmer animations | Monospace diagnostic status indicators |
| 17 | Infinite scroll on tables | Precise pagination with total record count tracking |
| 18 | Low contrast amber on cream | High-contrast `--signal-text: #A8731F` (WCAG AA certified) |
| 19 | Dark mode toggles with neon highlights | Purposeful dark canvas (`--ink-veil`) exclusively for graph topology |
| 20 | Carousel hero sliders | Focused 58/42 asymmetric split with live settling force layout |
