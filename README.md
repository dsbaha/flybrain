# flybrain — fruit-fly-scale LIF spiking network on WebGPU (Vulkan)

A Go + WebGPU (wgpu-native v29, Vulkan backend) leaky integrate-and-fire
simulator sized like the adult fruit fly brain connectome (138,639
neurons, 54.5M synapses from the real FlyWire proofread wiring), plus a
pure-Go CPU reference backend with bit-identical math.

Two single-static-binary companions ship from this repo:

- **flybrain** — the simulator: benchmark, brain server (TCP), protocol
  client/verifier, connectome fetcher, named experiments.
- **flyworld** — a 3D multi-brain arena (embedded Three.js web UI): 1–4
  brain servers, one embodied fly each, shared odor-plume world.

## Binaries

Prebuilt `flybrain-linux-amd64` and `flyworld-linux-amd64` are attached
to every [release](../../releases). To cut one:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

GitHub Actions (`.github/workflows/release.yml`) builds both binaries
with `CGO_ENABLED=1`, vets the tree, writes `SHA256SUMS.txt`, and
publishes all three as release assets. No GPU is needed on the build
runner — the Vulkan driver is only required at runtime.

## Model (Shiu et al., replicated)

The default model follows the published whole-brain Drosophila model
(Shiu et al.; reference implementation
[fly-brain_sim](https://github.com/nalin-dhiman/fly-brain_sim)):

- v_rest = v_reset = −52 mV, v_th = −45 mV, τ_m = 20 ms, refractory 2.2 ms
- Exponential synapses: conductance decays with τ_syn = 5 ms, 1.8 ms
  axonal+synaptic delay, conductance reset to 0 on spike
- Weight = sign × synapse count × 0.275 mV (sign from neurotransmitter
  predictions; fly glutamate is inhibitory)
- dt = 1 ms (reference uses 0.1 ms; 1 ms keeps 4 brains real-time)

Validation: the reference "sugar" experiment (21 sugar GRNs @ 200 Hz
Poisson) gives 0.12 Hz mean / 457 active neurons in Brian2. This binary
reproduces it: **0.13 Hz / 369 active** (`--experiment sugar`). CPU and
GPU backends are bit-identical (`--selftest` → `SELFTEST PASS`).

`--model classic` restores the legacy −70/−50 mV delta-synapse behavior.

## Quick start (BC-250)

```bash
./flybrain --selftest
./flybrain --fetch-connectome   # downloads FlyWire proofread data (0.85 GB), writes flywire-data/flywire783.fbc
./flybrain --backend=webgpu --graph flywire-data/flywire783.fbc \
  --experiment sugar --bg-rate 0 --inject-frac 0 --steps 20000
```

## Brain server + protocol

```bash
# Terminal 1: serve a brain (poisson sensory mode for embodied use)
./flybrain --server 0.0.0.0:5555 --graph flywire-data/flywire783.fbc \
  --sens-mode poisson

# Terminal 2: verify the protocol end to end
./flybrain --client localhost:5555
# ... CLIENT PASS: protocol verified end to end
```

TCP. Every message: 8-byte header (4-byte ASCII tag + uint32 LE payload
length) + payload. On connect: `FLYBRAIN/1 N=<neurons> DT=<dt-ms>\n`.

| Client → server | Payload | Server replies |
|---|---|---|
| `SENS` | u32 offset, then f32 values (persistent) | `OKAY` |
| `SENV` | u32 count, then count × (u32 index, f32 value) | `OKAY` |
| `STEP` | u32 nsteps (1..100000) | `DONE`: u64 totalSpikes, u64 simSteps |
| `READ` | u32 start, u32 count | `VOLT`: u32 start, u32 count, f32[count] mV |
| `SPIK` | u32 start, u32 count | `SPIK`: u32 start, u32 count, u32[count] 0/1 flags |
| `SPKC` | u32 start, u32 count | `SPKC`: u32 start, u32 count, u64[count] cumulative counts |
| `STAT` | (empty) | `STAT`: u64 simSteps, f64 simTimeMs, u64 totalSpikes, f64 lastStepWallSec |
| `QUIT` | (empty) | `BYE`, connection closes |

`SENS`/`SENV` value meaning follows `--sens-mode`: mV/step in `current`
mode, Hz Poisson in `poisson` mode. `SPKC` counts are exact per-neuron
cumulatives — use them for firing-rate readouts (no sampling noise).

## flyworld — 3D multi-brain world

```bash
# On each of up to 4 nodes (or 4 local ports):
./flybrain --server 0.0.0.0:5555 --graph flywire-data/flywire783.fbc \
  --sens-mode poisson

# Then the world (reaches all brains):
./flyworld --brains host1:5555,host2:5555,host3:5555,host4:5555 \
  --listen :8080 --graph flywire-data/flywire783.fbc
# open http://localhost:8080
```

Each fly, every tick runs a genuine sensorimotor loop over the real
FlyWire connectome:

- **Smell**: bilateral antennae (Gaussian plume + flicker) → Poisson-drive
  256 food-odor ORNs (ORN_DM1/VA2/DM2/VM5d, left/right) → real antennal-lobe
  wiring → 17 food PNs (read left/right separately). The PN L/R difference
  (EMA-smoothed) steers tumble direction when vision is unavailable.
- **Vision**: retinotopic LC9/LC31 (179 + 47 neurons) driven by object angle
  (left/right) with size tuning (peaks ~20 units). The DNp09 L/R rate
  difference steers toward seen objects (1.5 rad/s at full lateralization).
- **Walk**: DNp09 (central arousal + visual drive) gates forward speed.
  DNb05 (olfactory DN, genuinely suppressed by odor via AL inhibition)
  gates stopping at food.
- **Reverse**: collision → drive MDN's real excitatory inputs (LAL/DNp) →
  moonwalker DNs fire (170+ Hz) → backward walking (Bidaye et al. 2014).
- **Taste**: near food, 21 sugar GRNs (20/21 verified LB3) @ 200 Hz; recharge.

Hunger gates odor gain. Click the floor to drop food; the world
auto-redials restarted brains.

Honest status (v0.1.2): the full loop is neural — no raw world values in
steering. Validated: 0.96 path efficiency toward food, MDN fires on
reverse, DNb05 suppresses at food. LC9/LC31 tuning is a placeholder
(retinotopy + size, not true receptive fields); LC9's real role is
object tracking per courtship literature.

## Flags (selection)

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `webgpu` | `webgpu` or `cpu` |
| `--model` | `shiu` | `shiu` (reference) or `classic` (legacy) |
| `--tau-syn-ms` | `5.0` | synaptic conductance decay (0 = delta) |
| `--syn-delay-ms` | `1.8` | axonal+synaptic delay |
| `--sens-mode` | `current` | `current` (mV/step) or `poisson` (Hz) |
| `--poisson-kick-mv` | `68.75` | mV per Poisson sensory event |
| `--experiment` | — | `sugar` or `p9` (needs `--graph`) |
| `--w-scale` | `0.275` | mV per synapse at conversion |
| `--w-cap` | `25.0` | per-edge weight cap (mV) |
| `--bg-rate` | `0` (shiu) | background Poisson prob / neuron / ms |
| `--steps` | `2000` | sim steps |
| `--dt-ms` | `1.0` | timestep (ms) |
| `--selftest` | — | CPU-vs-GPU bit-comparison, then exit |
| `--graph` | — | `.fbc` connectome dump |

With `--model shiu` the background/inject drive defaults are 0 — the
reference regime is stimulus-evoked and sparse. Explicit flags win.

## Source layout

- `main.go`, `experiments.go`, `graph.go`, `lif.go` (CPU), `webgpu.go`,
  `shaders.wgsl`, `server.go`, `client.go`, `fetch.go`
- `viz/world3d/` — flyworld (Go + embedded Three.js)
- `viz/` — legacy 2D visualizer (tuned for the old model)

## Notes

- Go 1.26; WebGPU via `github.com/townsendmerino/wgpu` (CGO,
  wgpu-native v29 statically linked). Runtime needs only libc/libm and
  a Vulkan-capable GPU (`libvulkan1` on Debian/Ubuntu).
- Dataset: [FlyWire proofread connectome](https://zenodo.org/records/10676866);
  paper: <https://doi.org/10.1038/s41586-024-07558-y>.
