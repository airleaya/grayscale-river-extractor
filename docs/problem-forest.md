# Problem Forest

## Usage Rules

- `[ ]` means unresolved.
- `[x]` means resolved.
- `Status` records the latest judgment.
- `Source` records which parent problem this item derives from.
- `ID` is stable and can be referenced by later notes, code, or discussions.
- New derived questions should be appended as children under their source node.

## Active Todo List

- [x] `P-001` Clarify the baseline data scale of a `10000 x 10000` image.
- [x] `P-002` Estimate the cost of a per-pixel linear traversal with depth `x`.
- [x] `P-003` Judge whether a four-neighborhood depth traversal is computationally feasible.
- [ ] `P-004` Redesign the algorithm model as a drainage / river-channel extraction problem.
- [ ] `P-005` Decide the software architecture shape: web, desktop, or hybrid.
- [ ] `P-006` Decide the first implementation stack for the compute engine.
- [ ] `P-007` Design task management for long-running jobs.
- [ ] `P-008` Design memory and tiling strategy for large images.
- [ ] `P-009` Define correctness validation and benchmark methodology.
  - Progress note: a first small-sample unit-test baseline now exists for strict-slope routing, flat-region outlet routing, and weighted flat-routing fallback.
- [ ] `P-010` Redesign flow-direction computation from strict local D8 into an extensible weighted decision system.
- [ ] `P-011` Migrate heavy raster kernels from Python to Rust while preserving correctness and progress visibility.
  - Progress note: strict `D8`, flat-region labeling, and flat-outlet drop-map kernels now have Rust implementations and can be toggled from the current Python pipeline.

## Problem Forest

### `P-000` Build software for large-image pixel-level computation

- Status: in progress
- Source: root
- Notes: The target is a front-end/back-end application for large image processing, with long-running pixel-level algorithms and future architecture discussion.

#### `P-001` What is the baseline scale of a `10000 x 10000` image?

- Status: resolved
- Source: `P-000`
- Notes:
  - Pixel count is `1e8`.
  - Raw grayscale data is about `100 MB`.
  - Raw RGB data is about `300 MB`.
  - Raw RGBA data is about `400 MB`.

#### `P-002` If each pixel performs one linear traversal of depth `x`, how long will it take on the current machine?

- Status: resolved with rough estimate
- Source: `P-000`
- Notes:
  - Total work is approximately `1e8 * x`.
  - A local rough Python loop benchmark on this machine was about `1e7` iterations per second for a trivial single-thread loop.
  - A practical Python implementation with boundary checks and pixel access would usually be slower than the trivial loop benchmark.
  - This gives a rough lower-bound estimate near `10x` seconds in pure Python, before accounting for real memory access and branching cost.

#### `P-003` If each pixel performs four-neighborhood traversal with depth `x`, what scale does the problem reach?

- Status: resolved as infeasible in the naive form
- Source: `P-000`
- Notes:
  - If traversal expands as a tree without de-duplication, per-pixel complexity grows exponentially, near `O(4^x)`.
  - If immediate backtracking is pruned but paths are still expanded, the growth is still exponential, near `O(3^x)`.
  - If positions are de-duplicated with `visited`, one source pixel covers about `1 + 2x(x + 1)` positions, which is `O(x^2)`.
  - Running such a neighborhood search independently from every pixel causes overwhelming repeated work.
  - Conclusion: naive per-pixel deep four-neighborhood traversal should not be used.

##### `P-004` How should the algorithm be redesigned as a drainage / river-channel extraction problem?

- Status: open
- Source: `P-003`
- Open directions:
  - Can the problem be modeled as terrain drainage instead of repeated neighborhood traversal?
  - Can we compute local flow direction once per pixel, then derive accumulated flow globally?
  - Can river channels be extracted by thresholding or classifying flow accumulation and terrain structure?
  - What preprocessing is needed to make the height map hydrologically usable?

###### `P-004-A` What is the exact mathematical definition of the target computation?

- Status: reframed
- Source: `P-004`
- Notes:
  - The target is now clarified at a higher level:
    - Input is a height grayscale map.
    - Desired output is a river-channel map.
  - This reframing suggests the core pipeline may be:
    - preprocess the terrain,
    - compute flow direction,
    - compute flow accumulation,
    - extract channels.
  - The previous path-by-path formulation was too low-level and may not be the right primary abstraction.

####### `P-004-A-1` How is the input scalar field defined from the image?

- Status: open
- Source: `P-004-A`
- Notes:
  - We need to know whether computation uses one grayscale channel, a conversion from RGB, or a custom scalar field.

####### `P-004-A-2` What exactly counts as a "lower point"?

- Status: open
- Source: `P-004-A`
- Notes:
  - Candidate meanings include lower intensity, lower height, lower score, or lower value after preprocessing.

####### `P-004-A-3` Is the traversal a single-path descent or a multi-branch expansion?

- Status: deprioritized after reframing
- Source: `P-004-A`
- Notes:
  - This may become an implementation detail of the flow model rather than the main problem statement.

####### `P-004-A-4` What is the exact stopping rule for a path?

- Status: deprioritized after reframing
- Source: `P-004-A`
- Notes:
  - This may be absorbed into a standard hydrological flow model.

####### `P-004-A-5` What exact data structure should represent the output "direction matrix"?

- Status: updated
- Source: `P-004-A`
- Notes:
  - The main output target is now a river-channel map.
  - Supporting intermediate products may still include a flow-direction raster and flow-accumulation raster.

####### `P-004-A-6` What exact river output is desired?

- Status: resolved for v1
- Source: `P-004-A`
- Notes:
  - Version 1 output will be a binary river-channel mask.
  - The design should reserve extension points for future width-aware raster output and vector river-network output.

####### `P-004-A-7` What hydrological model should be used first?

- Status: partially resolved
- Source: `P-004-A`
- Notes:
  - Version 1 is currently implemented with a conservative `D8` single-flow model.
  - Current known limitation: strict-lower-only routing breaks on flat areas and can fragment downstream accumulation.
  - The next design direction is not to discard `D8`, but to evolve it into a weighted direction-selection framework with deterministic output.
  - Current process split:
    - no-outlet flat regions are repaired during preprocessing
    - outlet-reachable flat regions are routed during the flow-direction stage

####### `P-004-A-8` What preprocessing is required for the height map?

- Status: partially resolved
- Source: `P-004-A`
- Notes:
  - The current implementation already includes:
    - grayscale scalar extraction
    - bright/dark height mapping
    - optional smoothing
    - optional local sink filling
    - optional `nodata` preservation
  - Remaining open issue: preprocessing and flow-direction logic are not yet fully coupled through an explicit valid-data mask.
  - Current implementation split is now explicit:
    - no-outlet flat basins are repaired during preprocessing
    - outlet-reachable flats are routed during the flow-direction stage

####### `P-004-A-9` How do we preserve extensibility while shipping a binary-raster v1?

- Status: open
- Source: `P-004-A`
- Notes:
  - Intermediate products should likely be modeled explicitly, not hidden in a one-off pipeline.
  - Likely reusable artifacts include preprocessed DEM, flow-direction raster, flow-accumulation raster, and channel mask.
  - Output contracts should allow adding width estimation and vectorization later without breaking v1.

###### `P-004-B` What is the true asymptotic complexity after reformulation?

- Status: open
- Source: `P-004`
- Notes:
  - We need to determine whether the redesigned problem falls into `O(N)`, `O(N log N)`, `O(Nx)`, `O(Nx^2)`, or another class.

###### `P-010` How should flow-direction computation be redesigned as an extensible weighted decision system?

- Status: in progress
- Source: `P-004`
- Notes:
  - The current strict `D8` implementation is useful as a deterministic baseline, but it is too rigid for flat terrain and future multi-factor routing.
  - New discussion direction:
    - preserve deterministic reproducibility as the default
    - introduce direction weights per candidate neighbor
    - combine multiple factors into one composite score
    - reserve stochastic routing as a future optional mode rather than the default
  - Current implementation progress:
    - frontend/backend config already exposes slope, flat-escape, outlet-proximity, outlet-length, and outlet-distance weights
    - current code now uses a deterministic weighted selector in both strict downhill and flat-region routing
    - continuity weighting has also been integrated for outlet-reachable flat routing
    - next major step is behavior calibration, broader correctness validation, and remaining `valid_mask` consistency work rather than first-time integration

####### `P-010-A` How should flat terrain be handled without introducing cycles?

- Status: open
- Source: `P-010`
- Notes:
  - Flat areas with reachable lower outlets should not immediately become `-1` no-flow cells.
  - We should avoid naive equal-height random routing because it can introduce cycles and break topological accumulation.
  - Likely direction:
    - first compute strict-lower flow where possible
    - then resolve connected flat regions with outlet-aware routing

####### `P-010-B` Should weighted routing be deterministic or probabilistic?

- Status: decision pending, current recommendation is deterministic-first
- Source: `P-010`
- Notes:
  - Probabilistic routing is attractive for realism and variability, but it weakens reproducibility and makes debugging harder.
  - Recommended near-term decision:
    - compute weights for all candidate directions
    - choose the maximum composite weight deterministically
    - reserve probabilistic sampling as an experimental future mode

####### `P-010-C` What factors should contribute to direction weight in v1.5?

- Status: open
- Source: `P-010`
- Notes:
  - Candidate factors already discussed:
    - slope / height drop
    - equal-height flat-routing allowance
    - outlet proximity or escape preference on flat regions
    - optional continuity or inertia factor later
  - Recommended cut-in path is to begin with only three factors:
    - slope weight
    - flat-region routing weight
    - outlet proximity weight

####### `P-010-D` How should the weighted flow module be structured in code?

- Status: open
- Source: `P-010`
- Notes:
  - The current `compute_d8_flow_directions` function is still mostly a direct local scan.
  - Recommended refactor direction:
    - candidate construction
    - factor scoring
    - composite weight calculation
    - deterministic selection
    - flat-region post-resolution
  - This keeps low coupling and avoids a single deeply nested function.

####### `P-010-E` How should valid-mask / nodata handling be threaded into flow direction?

- Status: open
- Source: `P-010`
- Notes:
  - The preprocess stage already builds a valid-data mask, but the flow-direction pure function does not yet consume it explicitly.
  - Progress update:
    - `valid_mask` now participates in strict `D8`, flat-region labeling, and channel-thresholding
    - remaining work is to audit every flat-routing helper and accumulation edge case for full consistency

####### `P-010-F` What is the safest implementation sequence for the weighted flow update?

- Status: open
- Source: `P-010`
- Notes:
  - Recommended order:
    - step 1: pass valid mask into flow direction
    - step 2: refactor strict `D8` into candidate + scoring helpers with identical behavior
    - step 3: add flat-region outlet-aware routing
    - step 4: add multi-factor weighted scoring while keeping deterministic max-weight selection
    - step 5: expose weights and mode controls in frontend/API contracts

###### `P-011` How should heavy raster kernels be migrated from Python to Rust?

- Status: in progress
- Source: `P-000`
- Notes:
  - The current architecture is already suitable for staged migration:
    - `pipeline.py` owns orchestration, artifacts, and progress
    - `raster_algorithms.py` owns pure computation
    - `rust_bridge.py` owns optional acceleration and fallback
  - Migration principle:
    - keep Python orchestration
    - move only hot array kernels
    - preserve deterministic output and side-by-side verification

####### `P-011-A` Which kernels have already been migrated?

- Status: partially resolved
- Source: `P-011`
- Notes:
  - Rust kernels currently implemented:
    - strict `D8`
    - equal-height flat-region labeling
    - flat-outlet drop-map computation
  - All three are wired into the Python bridge and guarded by the existing `use_rust_kernel` switch.

####### `P-011-B` Which kernels should be migrated next?

- Status: open
- Source: `P-011`
- Notes:
  - Recommended next targets:
    - flat-outlet segment grouping
    - flat-outlet segment strength computation
    - flow accumulation
  - These will continue reducing Python-side nested loops without forcing a rewrite of the weighted flat-routing policy layer.

####### `P-011-C` How do we verify Python/Rust consistency during migration?

- Status: partially resolved
- Source: `P-011`
- Notes:
  - A dedicated regression test now compares Python and Rust flow-direction results on a flat-routing sample.
  - Remaining work:
    - add more terrain fixtures
    - benchmark Python vs Rust on representative large rasters
    - keep the Python path as a rollback reference until the Rust path covers more of the hot loop

####### `P-011-D` How do we preserve observability when more work moves into Rust?

- Status: open
- Source: `P-011`
- Notes:
  - Current Rust integration still reports progress at stage boundaries through Python.
  - As more kernels migrate, we will need either:
    - chunk-level progress callbacks from Rust, or
    - coarse sub-stage boundaries that still make forward motion visible to the user.

###### `P-004-C` What accuracy or approximation trade-offs are acceptable?

- Status: open
- Source: `P-004`
- Notes:
  - If exact computation is too expensive, we may need approximation, bounded radius sampling, or hierarchical coarse-to-fine strategies.

#### `P-005` What product architecture is appropriate for this software?

- Status: open
- Source: `P-000`
- Open directions:
  - Pure web architecture
  - Native desktop architecture
  - Hybrid architecture with UI plus local compute service

##### `P-005-A` Does the compute job need to run fully offline on the user's machine?

- Status: open
- Source: `P-005`

##### `P-005-B` Do we need to support very large local files without uploading them to a remote server?

- Status: open
- Source: `P-005`

##### `P-005-C` Is the primary usage single-user local analysis or shared multi-user task management?

- Status: open
- Source: `P-005`

#### `P-006` What should the first compute-engine implementation stack be?

- Status: open
- Source: `P-000`
- Open directions:
  - Python prototype first
  - Rust-first compute core
  - Python API plus Rust acceleration path

##### `P-006-A` Which parts require fast iteration and which parts require peak performance first?

- Status: open
- Source: `P-006`

##### `P-006-B` Will the first version prioritize algorithm exploration or production throughput?

- Status: open
- Source: `P-006`

#### `P-007` How should long-running tasks be managed?

- Status: open
- Source: `P-000`
- Notes:
  - Single runs may take one to two hours.
  - The system should not rely on synchronous request-response execution for such jobs.

##### `P-007-A` How do we represent task lifecycle states?

- Status: open
- Source: `P-007`
- Notes:
  - Suggested states: queued, running, paused, failed, canceled, completed.

##### `P-007-B` Do we need resumable execution and checkpointing?

- Status: open
- Source: `P-007`

##### `P-007-C` How should progress, logs, and ETA be exposed to the frontend?

- Status: open
- Source: `P-007`

#### `P-008` How should large-image memory usage be controlled?

- Status: open
- Source: `P-000`

##### `P-008-A` Can the algorithm operate tile-by-tile instead of loading the full image?

- Status: open
- Source: `P-008`

##### `P-008-B` What intermediate buffers are required, and how large can they grow?

- Status: open
- Source: `P-008`

##### `P-008-C` Are tile boundaries independent, or do we need overlap/halo regions?

- Status: open
- Source: `P-008`

#### `P-009` How do we validate correctness and estimate runtime safely?

- Status: open
- Source: `P-000`

##### `P-009-A` What small synthetic images can serve as correctness fixtures?

- Status: open
- Source: `P-009`

##### `P-009-B` What benchmark image set should represent real workloads?

- Status: open
- Source: `P-009`

##### `P-009-C` What metrics should we track besides wall-clock runtime?

- Status: open
- Source: `P-009`
- Notes:
  - Suggested metrics: memory peak, throughput, IO time, CPU utilization, correctness checks, and scaling trend versus image size and depth.

## MVP Design Principles

### `D-000` Global engineering constraints

- Status: active
- Notes:
  - Code comments should be complete and detailed enough to explain non-trivial intent, data flow, and algorithmic choices.
  - Code should avoid deep nesting as much as possible by splitting responsibilities into smaller functions and simplifying control flow.
  - Project structure should avoid high coupling; modules should communicate through clear contracts and stable data structures.
  - These constraints apply at every stage, not only during later refactoring.

### `D-001` Build the smallest runnable pipeline, but keep stage boundaries explicit

- Status: active
- Notes:
  - Version 1 should target one end-to-end result: input height map -> binary river-channel mask.
  - Each algorithm stage should have a named input/output contract so it can later be replaced or extended independently.
  - A stage-oriented pipeline is preferred over one monolithic function.

### `D-002` Preserve reusable intermediate rasters as first-class outputs

- Status: active
- Notes:
  - Even if version 1 only exposes the binary channel mask to users, internal contracts should preserve:
    - preprocessed terrain
    - flow direction
    - flow accumulation
    - channel mask
  - These products are the main extension points for later width estimation, confidence maps, and vectorization.

### `D-003` Long-running computation must be task-based, asynchronous, and observable

- Status: active
- Notes:
  - The system should create a task, execute it asynchronously, and expose task state instead of blocking a request until completion.
  - Required task-facing fields:
    - task id
    - lifecycle state
    - current stage
    - progress percent
    - processed units / total units
    - recent log messages
    - optional ETA

### `D-004` Progress reporting must be derived from tiled or chunked work units

- Status: active
- Notes:
  - For large images, progress should not depend on a single whole-image loop with no checkpoints.
  - Work should be partitioned into tiles, rows, blocks, or another measurable unit so progress can be emitted incrementally.
  - Stage-level progress alone is not sufficient; intra-stage progress is also needed.

### `D-005` Choose v1 algorithms that are simple, deterministic, and replaceable

- Status: active
- Notes:
  - Prefer a simple hydrological baseline such as:
    - minimal preprocessing
    - D8 flow direction
    - flow accumulation
    - threshold-based channel extraction
  - The first implementation should prioritize correctness, observability, and replaceable interfaces over peak performance.

### `D-006` The frontend should expose visual inspection as a first-class workflow

- Status: active
- Notes:
  - The UI should allow users to inspect:
    - the input image
    - preprocessed terrain
    - flow direction
    - flow accumulation
    - channel mask
  - Visualization should be available during task execution as soon as each artifact is generated.
  - Artifact viewing should use stable tabs or another persistent selector instead of reshaping the page per stage.

### `D-007` Intermediate artifacts should be persisted per task and remain viewable after completion

- Status: active
- Notes:
  - Each task should own a dedicated artifact directory.
  - Intermediate preview images should be written stage-by-stage instead of only after full completion.
  - Task metadata should expose artifact paths, preview eligibility, and generation status so the frontend can render progressively.

### `D-008` Raster viewing should support pixel-faithful zoom rather than only smooth image scaling

- Status: active
- Notes:
  - The viewer should support:
    - zoom
    - pan
    - nearest-neighbor rendering
    - a high-zoom pixel-grid overlay
  - At sufficiently high zoom, each source pixel should expand into a same-color display block with no smoothing.
  - Grid lines should appear only after a configurable zoom threshold to preserve readability and performance.
