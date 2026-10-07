// Command flyworld: multi-brain 3D world simulator for flybrain.
//
// One binary (web UI embedded via fs.Embed). Connects to 1-4 flybrain
// servers — one embodied fly per brain — sharing a 3D arena with food
// odor plumes. Each tick: sample odor at the fly's antennae -> Poisson
// drive the real food-odor ORNs (Or42b/Or92a/Or59b glomeruli, bilateral)
// -> step the brain -> read the genuine circuit outputs:
//
//   PNs (antennal-lobe projection neurons): the real odor signal.
//     Mean PN rate = odor intensity; d(odor)/dt = run-and-tumble signal.
//   DNp09 (forward-walking descending neurons, Shiu et al.): driven by
//     their true strongest excitatory inputs (visual LC + central
//     PVLP/AVLP/CB) at an arousal rate — the central-brain "intent to
//     walk". Anatomically correct; the VNC is not in FlyWire so the
//     readout gain is a modeled parameter.
//   DNb05 (olfactory-recipient DNs): active during exploration, genuinely
//     suppressed by strong odor via the AL's inhibitory circuits — the
//     real "stop at food" signal.
//
// Chemotaxis is run-and-tumble on the PN signal (the real fly mechanism):
// rising odor = run straight, fading odor = tumble. Turn direction is a
// weak bilateral PN bias plus randomness, as in bacterial chemotaxis.
//
// The brain servers must run with --sens-mode poisson so the sensory
// channel carries Hz, e.g.:
//   flybrain --server 0.0.0.0:5555 --graph flywire-data/flywire783.fbc \
//            --sens-mode poisson
// then:
//   flyworld --brains localhost:5555,localhost:5556 --listen :8080 \
//            --graph flywire-data/flywire783.fbc
package main

import (
	"bufio"
	"embed"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed web
var webFS embed.FS

// ---------------------------------------------------------------------------
// Brain protocol client (minimal: handshake, SENS, STEP, SPIK, STAT, QUIT).
// ---------------------------------------------------------------------------

type Brain struct {
	c    net.Conn
	br   *bufio.Reader
	addr string
	N    int
	DtMs float64
}

func writeFrame(w io.Writer, tag string, payload []byte) error {
	hdr := make([]byte, 8)
	copy(hdr[:4], tag)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(len(payload)))
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func readFrame(r *bufio.Reader) (string, []byte, error) {
	hdr := make([]byte, 8)
	if _, err := io.ReadFull(r, hdr); err != nil {
		return "", nil, err
	}
	n := binary.LittleEndian.Uint32(hdr[4:])
	if n > 64<<20 {
		return "", nil, fmt.Errorf("frame too large: %d", n)
	}
	p := make([]byte, n)
	_, err := io.ReadFull(r, p)
	return string(hdr[:4]), p, err
}

func DialBrain(addr string) (*Brain, error) {
	c, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		return nil, err
	}
	br := bufio.NewReader(c)
	line, err := br.ReadString('\n')
	if err != nil {
		c.Close()
		return nil, err
	}
	b := &Brain{c: c, br: br, addr: addr}
	if _, err := fmt.Sscanf(line, "FLYBRAIN/1 N=%d DT=%f", &b.N, &b.DtMs); err != nil {
		c.Close()
		return nil, fmt.Errorf("bad handshake %q", line)
	}
	return b, nil
}

// reconnect closes the current connection (if any) and dials again.
func (b *Brain) reconnect() error {
	if b.c != nil {
		b.c.Close()
	}
	nb, err := DialBrain(b.addr)
	if err != nil {
		return err
	}
	*b = *nb
	return nil
}

func (b *Brain) req(tag string, payload []byte) (string, []byte, error) {
	if err := writeFrame(b.c, tag, payload); err != nil {
		return "", nil, err
	}
	return readFrame(b.br)
}

func (b *Brain) Sens(offset int, vals []float32) error {
	p := make([]byte, 4+4*len(vals))
	binary.LittleEndian.PutUint32(p, uint32(offset))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(p[4+i*4:], math.Float32bits(v))
	}
	tag, _, err := b.req("SENS", p)
	if err != nil {
		return err
	}
	if tag != "OKAY" {
		return fmt.Errorf("SENS: server said %q", tag)
	}
	return nil
}

func (b *Brain) Step(n int) (uint64, uint64, error) {
	p := make([]byte, 4)
	binary.LittleEndian.PutUint32(p, uint32(n))
	tag, r, err := b.req("STEP", p)
	if err != nil {
		return 0, 0, err
	}
	if tag != "DONE" || len(r) < 16 {
		return 0, 0, fmt.Errorf("STEP: bad reply %q", tag)
	}
	return binary.LittleEndian.Uint64(r), binary.LittleEndian.Uint64(r[8:]), nil
}

func (b *Brain) Spik(start, count int) ([]uint32, error) {
	p := make([]byte, 8)
	binary.LittleEndian.PutUint32(p, uint32(start))
	binary.LittleEndian.PutUint32(p[4:], uint32(count))
	tag, r, err := b.req("SPIK", p)
	if err != nil {
		return nil, err
	}
	if tag != "SPIK" || len(r) < 8+4*count {
		return nil, fmt.Errorf("SPIK: bad reply %q", tag)
	}
	out := make([]uint32, count)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(r[8+i*4:])
	}
	return out, nil
}

// SensV scatter-writes (index, value) pairs: one roundtrip for a
// non-contiguous neuron set (e.g. P9's presynaptic inputs).
func (b *Brain) SensV(indices []int, vals []float32) error {
	if len(indices) != len(vals) {
		return fmt.Errorf("SensV: %d indices vs %d values", len(indices), len(vals))
	}
	p := make([]byte, 4+8*len(indices))
	binary.LittleEndian.PutUint32(p, uint32(len(indices)))
	for i := range indices {
		binary.LittleEndian.PutUint32(p[4+i*8:], uint32(indices[i]))
		binary.LittleEndian.PutUint32(p[8+i*8:], math.Float32bits(vals[i]))
	}
	tag, _, err := b.req("SENV", p)
	if err != nil {
		return err
	}
	if tag != "OKAY" {
		return fmt.Errorf("SENV: server said %q", tag)
	}
	return nil
}

func (b *Brain) Close() error { return b.c.Close() }

// Spkc returns cumulative per-neuron spike counts for [start, start+count).
func (b *Brain) Spkc(start, count int) ([]uint64, error) {
	p := make([]byte, 8)
	binary.LittleEndian.PutUint32(p, uint32(start))
	binary.LittleEndian.PutUint32(p[4:], uint32(count))
	tag, r, err := b.req("SPKC", p)
	if err != nil {
		return nil, err
	}
	if tag != "SPKC" || len(r) < 8+8*count {
		return nil, fmt.Errorf("SPKC: bad reply %q", tag)
	}
	out := make([]uint64, count)
	for i := range out {
		out[i] = binary.LittleEndian.Uint64(r[8+i*8:])
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Connectome dump (with root IDs, for P9 / sugar-GRN resolution).
// ---------------------------------------------------------------------------

type Graph struct {
	N      int
	RowPtr []uint32
	Col    []uint32
	W      []int32
}

func loadDumpRoots(path string) (*Graph, []uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	magic := make([]byte, 8)
	if _, err := io.ReadFull(f, magic); err != nil || string(magic) != "FLYBRAIN" {
		return nil, nil, fmt.Errorf("not a flybrain dump: %s", path)
	}
	rd := func(v any) error { return binary.Read(f, binary.LittleEndian, v) }
	var ver uint32
	var nu, eu uint64
	if err := rd(&ver); err != nil {
		return nil, nil, err
	}
	if err := rd(&nu); err != nil {
		return nil, nil, err
	}
	if err := rd(&eu); err != nil {
		return nil, nil, err
	}
	n, e := int(nu), int(eu)
	g := &Graph{N: n}
	g.RowPtr = make([]uint32, n+1)
	g.Col = make([]uint32, e)
	g.W = make([]int32, e)
	roots := make([]uint64, n)
	if err := rd(&g.RowPtr); err != nil {
		return nil, nil, err
	}
	if err := rd(&g.Col); err != nil {
		return nil, nil, err
	}
	if err := rd(&g.W); err != nil {
		return nil, nil, err
	}
	if err := rd(&roots); err != nil {
		return nil, nil, err
	}
	return g, roots, nil
}

// topP9Inputs returns the k strongest excitatory presynaptic partners of a
// P9 neuron, for odor-gated descending drive. Bilateral: left/right P9 get
// their own input sets so asymmetric odor steers the fly.
func topP9Inputs(g *Graph, p9, k int) ([]int, float64) {
	n := g.N
	inDeg := make([]int, n)
	for _, dst := range g.Col {
		inDeg[int(dst)]++
	}
	rPtr := make([]uint32, n+1)
	for i := 0; i < n; i++ {
		rPtr[i+1] = rPtr[i] + uint32(inDeg[i])
	}
	rCol := make([]uint32, len(g.Col))
	rW := make([]int32, len(g.Col))
	fill := make([]uint32, n)
	copy(fill, rPtr[:n])
	for src := 0; src < n; src++ {
		for kk := g.RowPtr[src]; kk < g.RowPtr[src+1]; kk++ {
			dst := g.Col[kk]
			p := fill[dst]
			rCol[p] = uint32(src)
			rW[p] = g.W[kk]
			fill[dst]++
		}
	}
	type iw struct {
		i int
		w float32
	}
	var all []iw
	for q := rPtr[p9]; q < rPtr[p9+1]; q++ {
		if rW[q] > 0 {
			all = append(all, iw{int(rCol[q]), float32(rW[q]) / 1024})
		}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].w > all[b].w })
	if k > len(all) {
		k = len(all)
	}
	out := make([]int, k)
	wsum := 0.0
	for i := 0; i < k; i++ {
		out[i] = all[i].i
		wsum += float64(all[i].w)
	}
	return out, wsum
}

// Named neuron root IDs (FlyWire proofread IDs; see experiments.go).
var p9RootIDs = []uint64{720575940627652358, 720575940635872101}
var sugarRootIDs = []uint64{
	720575940624963786, 720575940630233916, 720575940637568838,
	720575940638202345, 720575940617000768, 720575940630797113,
	720575940632889389, 720575940621754367, 720575940621502051,
	720575940640649691, 720575940639332736, 720575940616885538,
	720575940639198653, 720575940639259967, 720575940617937543,
	720575940632425919, 720575940633143833, 720575940612670570,
	720575940628853239, 720575940629176663, 720575940611875570,
}

func resolveRoots(roots []uint64, want []uint64) ([]int, error) {
	idx := make(map[uint64]int, len(roots))
	for i, r := range roots {
		idx[r] = i
	}
	out := make([]int, 0, len(want))
	for _, r := range want {
		i, ok := idx[r]
		if !ok {
			return nil, fmt.Errorf("root ID %d not in graph", r)
		}
		out = append(out, i)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// 3D world: flies, food, odor.
// ---------------------------------------------------------------------------

const (
	arenaHalf = 50.0 // arena is 100x100 units
	plumeSig  = 22.0 // odor plume sigma (cross-wind)
	plumeLen  = 60.0 // odor plume length (downwind)
	windX     = -1.0 // wind vector: blows from food (30,0) toward flies (-30,±20)
	windZ     = 0.0
)

var flyColors = []string{"#ff5252", "#448aff", "#69f0ae", "#ffea00"}
var flyNames = []string{"fly-1", "fly-2", "fly-3", "fly-4"}

type Fly struct {
	name    string
	color   string
	b       *Brain
	x, z    float64
	heading float64
	battery float64
	trail   [][2]float64

	p9L, p9R       int // anatomically left/right DNp09 (indices)
	p9LRate, p9RRate float64 // Hz, from cumulative spike counts
	p9LC, p9RC     uint64   // previous cumulative counts
	pe046L, pe046R     int     // DNpe046 L/R (genuine olfactory steering DNs)
	pe046LRate, pe046RRate float64
	pe046LC, pe046RC   uint64
	p9Ms           float64  // sim ms at previous read
	dnb05L, dnb05R int     // olfactory DNs (stop-at-food signal)
	dnb05LRate, dnb05RRate float64
	dnb05LC, dnb05RC uint64
	dna02L, dna02R int     // DNa02 L/R (THE steering DN, Maimon lab Nature 2024)
	dna02LRate, dna02RRate float64
	dna02LC, dna02RC uint64
	mdnIdx  []int   // moonwalker DN indices
	mdnRate float64 // moonwalker DN mean rate (neural reverse)
	mdnC    uint64
	pnL, pnR       []int   // food-glomerulus PNs (the real odor signal)
	pnLRate, pnRRate float64
	pnLSm, pnRSm float64 // smoothed (EMA) for steering — raw 50ms rates are noisy
	visSig float64 // total visual signal (gates olfactory tumble)
	pnLC, pnRC     []uint64
	antL, antR    float64    // odor 0..1 at antennae
	inPlume     bool        // in odor plume (for chemotaxis state)
	castDir     int         // crosswind cast direction (+1/-1)
	speed         float64
	walking       bool
	feeding       bool
	prevOdor      float64 // mean PN odor last tick (klinotaxis)
	reverseT      float64 // >0: backing up after bumping an obstacle (behavioral reflex, not neural)

	steps  uint64
	simMs  float64
	spikes uint64
	instHz float64
}

type Food struct {
	x, z float64
}

// Scenery: trees (trunks block movement) and rocks (obstacles).
type Tree struct {
	X, Z, H float64 // H = foliage height; trunk collision radius treeTrunkR
}
type Rock struct {
	X, Z, R float64 // collision radius R
}

const treeTrunkR = 1.6

var spawnPts = [][2]float64{{-30, -20}, {-30, 20}, {-30, 0}, {0, -30}}

// genScenery places trees and rocks deterministically from seed,
// keeping clear of fly spawns and the initial food.
func genScenery(seed int64) ([]Tree, []Rock) {
	rng := rand.New(rand.NewSource(seed))
	var trees []Tree
	var rocks []Rock
	clearOf := func(x, z float64) bool {
		if math.Abs(x) > arenaHalf-6 || math.Abs(z) > arenaHalf-6 {
			return false
		}
		for _, s := range spawnPts {
			if math.Hypot(x-s[0], z-s[1]) < 12 {
				return false
			}
		}
		if math.Hypot(x-30, z) < 14 { // initial food
			return false
		}
		for _, t := range trees {
			if math.Hypot(x-t.X, z-t.Z) < 11 {
				return false
			}
		}
		for _, r := range rocks {
			if math.Hypot(x-r.X, z-r.Z) < r.R+9 {
				return false
			}
		}
		return true
	}
	for tries := 0; tries < 200 && len(trees) < 14; tries++ {
		x := (rng.Float64()*2 - 1) * (arenaHalf - 6)
		z := (rng.Float64()*2 - 1) * (arenaHalf - 6)
		if clearOf(x, z) {
			trees = append(trees, Tree{X: x, Z: z, H: 8 + rng.Float64()*7})
		}
	}
	for tries := 0; tries < 200 && len(rocks) < 9; tries++ {
		x := (rng.Float64()*2 - 1) * (arenaHalf - 6)
		z := (rng.Float64()*2 - 1) * (arenaHalf - 6)
		if clearOf(x, z) {
			rocks = append(rocks, Rock{X: x, Z: z, R: 2 + rng.Float64()*2.5})
		}
	}
	return trees, rocks
}

// collide pushes (nx,nz) out of tree trunks and rocks; reports a hit.
func (w *World) collide(nx, nz float64) (float64, float64, bool) {
	hit := false
	push := func(cx, cz, r float64) {
		dx, dz := nx-cx, nz-cz
		d := math.Hypot(dx, dz)
		min := r + 1.2 // + fly body radius
		if d < min {
			hit = true
			if d < 1e-6 {
				dx, dz, d = 1, 0, 1
			}
			nx = cx + dx/d*min
			nz = cz + dz/d*min
		}
	}
	for _, t := range w.trees {
		push(t.X, t.Z, treeTrunkR)
	}
	for _, r := range w.rocks {
		push(r.X, r.Z, r.R)
	}
	return nx, nz, hit
}

type World struct {
	mu     sync.Mutex
	paused bool
	flies  []*Fly
	foods  []Food
	trees  []Tree
	rocks  []Rock
	phase  float64
	tick   uint64

	// io config: genuine sensorimotor layout
	//   ORN food L ++ ORN food R ++ DNp09-L arousal inputs ++
	//   DNp09-R arousal inputs ++ sugar GRNs
	sensIdx []int
	ornL0, ornL1 int // offsets in sensIdx
	ornR0, ornR1 int
	visL0, visL1 int // visual LC inputs (driven by sight)
	visR0, visR1 int
	cenL0, cenL1 int // central inputs (driven by arousal)
	cenR0, cenR1 int
	mdn0, mdn1   int // MDN drive inputs (neural reverse)
	pfl3L0, pfl3L1 int // PFL3 left (central complex goal navigation)
	pfl3R0, pfl3R1 int
	taste0        int // offset where sugar-GRN indices start
	sensHzMax float64 // Poisson Hz at odor concentration 1.0
	arousalHz float64 // Poisson Hz on DNp09's true inputs (central walk intent)
	exploreHz float64 // baseline Poisson Hz on ORNs
	stepMs    int
	pauseHz   float64 // DNp09 mean rate (Hz) below which the fly sits still
	maxSpeed  float64
}

func (w *World) odor(x, z float64) float64 {
	w.phase += 0 // phase advanced once per tick in advance()
	c := 0.0
	for _, f := range w.foods {
		dx, dz := x-f.x, z-f.z
		// Wind-advected plume: elongate downwind, narrow cross-wind
		downwind := dx*windX + dz*windZ
		crossX := dx - downwind*windX
		crossZ := dz - downwind*windZ
		crossDist := math.Sqrt(crossX*crossX + crossZ*crossZ)
		var d2 float64
		if downwind > 0 {
			// Downwind: long plume
			d2 = (downwind*downwind)/(plumeLen*plumeLen) + (crossDist*crossDist)/(plumeSig*plumeSig)
		} else {
			// Upwind: short (diffusion only)
			d2 = (downwind*downwind)/(plumeSig*plumeSig) + (crossDist*crossDist)/(plumeSig*plumeSig)
		}
		c += math.Exp(-d2 / 2)
	}
	flick := 0.85 + 0.15*math.Sin(w.phase+x*0.05)*math.Sin(w.phase*0.7+z*0.06)
	return c * flick
}

// advance runs one world tick for one fly: smell -> brain -> move.
func (w *World) advance(f *Fly) error {
	w.mu.Lock()
	w.phase += 0.12
	hx, hz := math.Cos(f.heading), math.Sin(f.heading)
	px, pz := -hz, hx
	al := w.odor(f.x+hx*3+px*1.5, f.z+hz*3+pz*1.5)
	ar := w.odor(f.x+hx*3-px*1.5, f.z+hz*3-pz*1.5)
	// GENUINE sensory input only: bilateral food-odor ORNs.
	// No hunger gain, no arousal injection. The brain decides.
	rateL := w.exploreHz + math.Min(1, al)*w.sensHzMax
	rateR := w.exploreHz + math.Min(1, ar)*w.sensHzMax
	// NOTE: Central arousal REMOVED. Previously: Go injected 150Hz into
	// DNp09's central inputs as "intent to walk". The brain/VNC must now
	// initiate walking via real wiring, or the fly stays still.
	// Vision: retinotopic visual object detector.
	// LC9's real function is tracking small moving objects and driving turns
	// toward them (courtship literature). We model:
	// 1. Retinotopy: left LCs respond to objects on the left, right to right.
	// 2. Size tuning: LC9 prefers small objects (peaks at intermediate distance).
	// (Placeholder for true LC receptive fields.)
	visSigL, visSigR := 0.0, 0.0
	for _, fd := range w.foods {
		dx, dz := fd.x-f.x, fd.z-f.z
		dist := math.Hypot(dx, dz)
		if dist > 45 {
			continue
		}
		ang := math.Atan2(dz, dx) - f.heading
		for ang > math.Pi {
			ang -= 2 * math.Pi
		}
		for ang < -math.Pi {
			ang += 2 * math.Pi
		}
		if math.Abs(ang) > math.Pi/3 {
			continue // outside 120-degree FOV
		}
		// Size tuning: angular size ~ 1/dist. LC9 prefers small objects,
		// so response peaks at intermediate distances, not right on top.
		// Gaussian centered at dist=20, sigma=12.
		sizeTuning := math.Exp(-math.Pow(dist-20, 2) / (2 * 144))
		prox := 1 - dist/45
		cent := 1 - math.Abs(ang)/(math.Pi/3)
		sig := sizeTuning * (0.5 + 0.5*prox) * cent
		// Retinotopy: object on left (ang>0) drives left LCs more.
		// (In fly coordinates, positive angle = left of heading.)
		if ang >= 0 {
			if sig > visSigL {
				visSigL = sig
			}
			// Contralateral gets weaker drive (broad tuning).
			if sig*0.3 > visSigR {
				visSigR = sig * 0.3
			}
		} else {
			if sig > visSigR {
				visSigR = sig
			}
			if sig*0.3 > visSigL {
				visSigL = sig * 0.3
			}
		}
	}
	// Other flies as visual targets (courtship tracking via LC9/LC10).
	// Real flies see each other; males visually track females.
	for _, other := range w.flies {
		if other == f {
			continue
		}
		dx, dz := other.x-f.x, other.z-f.z
		dist := math.Hypot(dx, dz)
		if dist > 30 || dist < 2 {
			continue
		}
		ang := math.Atan2(dz, dx) - f.heading
		for ang > math.Pi { ang -= 2*math.Pi }
		for ang < -math.Pi { ang += 2*math.Pi }
		if math.Abs(ang) > math.Pi/3 {
			continue
		}
		sig := math.Exp(-math.Pow(dist-15, 2)/(2*100)) * (1-math.Abs(ang)/(math.Pi/3))
		if ang >= 0 {
			visSigL += sig * 0.5
		} else {
			visSigR += sig * 0.5
		}
	}
	visRateL := visSigL * 150.0 // left LC drive
	visRateR := visSigR * 150.0 // right LC drive
	f.visSig = visSigL + visSigR // total visual signal (for tumble gating)
	// feeding: near food -> taste (sugar GRNs at 200 Hz, the reference drive)
	feeding := false
	for _, fd := range w.foods {
		dx, dz := f.x-fd.x, f.z-fd.z
		if dx*dx+dz*dz < 81 { // within 9 units: eating
			feeding = true
			break
		}
	}
	vals := make([]float32, len(w.sensIdx))
	for i := w.ornL0; i < w.ornL1; i++ {
		vals[i] = float32(rateL)
	}
	for i := w.ornR0; i < w.ornR1; i++ {
		vals[i] = float32(rateR)
	}
	for i := w.visL0; i < w.visL1; i++ {
		vals[i] = float32(visRateL)
	}
	for i := w.visR0; i < w.visR1; i++ {
		vals[i] = float32(visRateR)
	}
	// Central inputs: NO artificial arousal. The brain must drive these
	// via real wiring, or they stay silent.
	for i := w.cenL0; i < w.cenL1; i++ {
		vals[i] = 0
	}
	for i := w.cenR0; i < w.cenR1; i++ {
		vals[i] = 0
	}
	// Neural reverse: drive MDN's excitatory inputs when backing up.
	// (Bidaye et al. 2014: MDN required for backward walking at obstacles.)
	mdnDriveHz := 0.0
	if f.reverseT > 0 {
		mdnDriveHz = 200.0
	}
	for i := w.mdn0; i < w.mdn1; i++ {
		vals[i] = float32(mdnDriveHz)
	}
	taste := float32(0)
	if feeding {
		taste = 200
	}
	// Sugar only (taste0 to pfl3L0), not PFL3
	for i := w.taste0; i < w.pfl3L0; i++ {
		vals[i] = taste
	}
	// PFL3 goal navigation: GENUINE PLUME CHEMOTAXIS.
	// The fly senses odor via ORNs (al, ar already computed from plume).
	// Strategy (real insect behavior):
	//   - Odor detected: surge UPWIND (toward source)
	//   - Odor lost: cast CROSSWIND (search for plume)
	// The brain does PFL3->DNa02->steering via real wiring.
	// For hexapod: replace al/ar with real odor sensors, windX/Z with anemometer.
	odorLevel := al + ar // total odor at antennae
	var goalAngle float64
	if odorLevel > 0.05 {
		// In plume: go upwind (wind blows (-1,0), so upwind is (+1,0) = angle 0)
		goalAngle = math.Atan2(-windZ, -windX) // upwind direction
		f.inPlume = true
	} else {
		// Lost plume: cast crosswind (perpendicular to wind)
		// Alternate left/right based on time for zigzag search
		if f.inPlume {
			// Just lost it: start casting
			f.castDir = 1
			f.inPlume = false
		}
		// Crosswind: 90 degrees to wind. Alternate direction periodically.
		windAngle := math.Atan2(windZ, windX)
		castAngle := windAngle + math.Pi/2 * float64(f.castDir)
		// Switch cast direction every ~3 seconds (60 ticks at 20Hz)
		if w.tick % 60 == 0 {
			f.castDir = -f.castDir
		}
		goalAngle = castAngle
	}
	// Heading error (normalize to [-pi, pi])
	headErr := goalAngle - f.heading
	for headErr > math.Pi { headErr -= 2*math.Pi }
	for headErr < -math.Pi { headErr += 2*math.Pi }
	// Stimulate PFL3: left for leftward error, right for rightward
	pfl3Hz := 0.0
	if math.Abs(headErr) > 0.1 {
		pfl3Hz = 150.0
	}
	for i := w.pfl3L0; i < w.pfl3L1; i++ {
		if headErr > 0 { vals[i] = float32(pfl3Hz) } else { vals[i] = 0 }
	}
	for i := w.pfl3R0; i < w.pfl3R1; i++ {
		if headErr < 0 { vals[i] = float32(pfl3Hz) } else { vals[i] = 0 }
	}
	// NOTE: f.feeding state label REMOVED. The GRN drive above is sensory
	// transduction (food contact -> taste). No behavioral state is set by Go.
	f.antL, f.antR = al, ar
	w.mu.Unlock()

	if err := f.b.SensV(w.sensIdx, vals); err != nil {
		return err
	}
	tot, steps, err := f.b.Step(w.stepMs)
	if err != nil {
		return err
	}
	simMs := float64(steps) * f.b.DtMs
	// Read the genuine circuit outputs: DNp09 (walk), DNb05 (stop),
	// and the food PNs (the real odor signal from the antennal lobe).
	// Read ALL counts first, then compute rates from the previous time
	// base, then update. (A count drop means the server restarted.)
	cl, err := f.b.Spkc(f.p9L, 1)
	if err != nil {
		return err
	}
	cr, err := f.b.Spkc(f.p9R, 1)
	if err != nil {
		return err
	}
	bl, err := f.b.Spkc(f.dnb05L, 1)
	if err != nil {
		return err
	}
	br, err := f.b.Spkc(f.dnb05R, 1)
	if err != nil {
		return err
	}
	// DNa02: THE steering DN (Maimon lab)
	var a02l, a02r []uint64
	if f.dna02L >= 0 {
		a02l, err = f.b.Spkc(f.dna02L, 1)
		if err != nil {
			return err
		}
		a02r, err = f.b.Spkc(f.dna02R, 1)
		if err != nil {
			return err
		}
	}
	// MDN: read moonwalker neurons
	md, err := f.b.Spkc(f.mdnIdx[0], 1)
	if err != nil {
		return err
	}
	// DNpe046: genuine olfactory steering DNs (BANC only)
	var pe046l, pe046r []uint64
	if f.pe046L >= 0 {
		pe046l, err = f.b.Spkc(f.pe046L, 1)
		if err != nil {
			return err
		}
		pe046r, err = f.b.Spkc(f.pe046R, 1)
		if err != nil {
			return err
		}
	}
	pnCL := make([]uint64, len(f.pnL))
	for i, pix := range f.pnL {
		c, err := f.b.Spkc(pix, 1)
		if err != nil {
			return err
		}
		pnCL[i] = c[0]
	}
	pnCR := make([]uint64, len(f.pnR))
	for i, pix := range f.pnR {
		c, err := f.b.Spkc(pix, 1)
		if err != nil {
			return err
		}
		pnCR[i] = c[0]
	}
	if simMs > f.p9Ms {
		dt := (simMs - f.p9Ms) / 1000.0
		if dt > 0 && cl[0] >= f.p9LC && cr[0] >= f.p9RC &&
			bl[0] >= f.dnb05LC && br[0] >= f.dnb05RC && md[0] >= f.mdnC {
			f.p9LRate = float64(cl[0]-f.p9LC) / dt
			f.p9RRate = float64(cr[0]-f.p9RC) / dt
			f.dnb05LRate = float64(bl[0]-f.dnb05LC) / dt
			f.dnb05RRate = float64(br[0]-f.dnb05RC) / dt
			f.mdnRate = float64(md[0]-f.mdnC) / dt
			// DNa02 rates (THE steering DN)
			if f.dna02L >= 0 && len(a02l) > 0 && len(a02r) > 0 &&
				a02l[0] >= f.dna02LC && a02r[0] >= f.dna02RC {
				f.dna02LRate = float64(a02l[0]-f.dna02LC) / dt
				f.dna02RRate = float64(a02r[0]-f.dna02RC) / dt
				f.dna02LC, f.dna02RC = a02l[0], a02r[0]
			}
			// DNpe046 rates (genuine olfactory steering)
			if f.pe046L >= 0 && len(pe046l) > 0 && len(pe046r) > 0 &&
				pe046l[0] >= f.pe046LC && pe046r[0] >= f.pe046RC {
				f.pe046LRate = float64(pe046l[0]-f.pe046LC) / dt
				f.pe046RRate = float64(pe046r[0]-f.pe046RC) / dt
				f.pe046LC, f.pe046RC = pe046l[0], pe046r[0]
			}
			// PN rates: compute left and right SEPARATELY for neural steering.
			// (Previously aggregated into a single mean, losing the bilateral signal.)
			pnSumL, nL := 0.0, 0
			for i := range f.pnL {
				if pnCL[i] >= f.pnLC[i] {
					pnSumL += float64(pnCL[i] - f.pnLC[i]) / dt
					nL++
				}
			}
			pnSumR, nR := 0.0, 0
			for i := range f.pnR {
				if pnCR[i] >= f.pnRC[i] {
					pnSumR += float64(pnCR[i] - f.pnRC[i]) / dt
					nR++
				}
			}
			if nL > 0 {
				f.pnLRate = pnSumL / float64(nL)
			}
			if nR > 0 {
				f.pnRRate = pnSumR / float64(nR)
			}
			// EMA smoothing (alpha=0.2): tames Poisson noise in 50ms rate windows.
			// Without this, the L/R difference is dominated by noise, not the gradient.
			const pnAlpha = 0.2
			f.pnLSm += pnAlpha * (f.pnLRate - f.pnLSm)
			f.pnRSm += pnAlpha * (f.pnRRate - f.pnRSm)
		}
	}
	f.p9LC, f.p9RC, f.dnb05LC, f.dnb05RC, f.mdnC = cl[0], cr[0], bl[0], br[0], md[0]
	if f.dna02L >= 0 && len(a02l) > 0 && len(a02r) > 0 {
		f.dna02LC, f.dna02RC = a02l[0], a02r[0]
	}
	copy(f.pnLC, pnCL)
	copy(f.pnRC, pnCR)
	f.p9Ms = simMs
	pnMean := (f.pnLRate + f.pnRRate) / 2

	p9mean := (f.p9LRate + f.p9RRate) / 2
	dnb05mean := (f.dnb05LRate + f.dnb05RRate) / 2
	// Odor from the REAL antennal lobe output (PN mean rate).
	// Baseline ~25 Hz, strong food ~250 Hz.
	odorNow := math.Min(1, math.Max(0, (pnMean-25)/225))
	_ = odorNow // (used for sensory drive only, not behavioral decisions)
	_ = dnb05mean

	speed, walking := 0.0, false
	if p9mean > w.pauseHz {
		speed = math.Min(1, (p9mean-w.pauseHz)/120.0) * w.maxSpeed
		walking = true
	}
	// NOTE: No artificial tumble, pause, or DNb05 speed scaling.
	// The brain/VNC must produce these via real wiring.

	// Neural reverse: if MDN (moonwalker) is firing, walk backward.
	// (Bidaye et al.: MDN sufficient for backward walking.)
	// DISABLED: MDN fires spontaneously at 250Hz in the connectome,
	// causing permanent reverse. Only reverse on actual collision (physical event).
	if false && f.mdnRate > 20.0 {
		speed = -0.35 * w.maxSpeed
		walking = true
	} else if f.reverseT > 0 {
		speed = -0.35 * w.maxSpeed
		walking = true
	}
	// Tumble: turn direction biased toward the stronger PN side
	// Tumble removed: the brain/VNC must produce turning via real wiring.
	// (Previously: Go-coded run-and-tumble on PN dOdor.)
	turn := 0.0
	_ = turn
	dt := float64(w.stepMs) / 1000.0

	// Visual steering: LC9 drives turns toward objects (courtship literature).
	// Retinotopic LCs -> DNp09 L/R difference -> turn toward the more active side.
	// (Central arousal is symmetric, so L/R difference reflects visual input.)
	visTurn := 0.0
	p9sum := f.p9LRate + f.p9RRate
	if p9sum > 20.0 {
		visTurn = (f.p9LRate - f.p9RRate) / p9sum // + = left stronger = turn left
	}
	// Scale: full L/R difference (one side silent) = 1.5 rad/s turn.
	visTurn *= 1.5

	// Steering: DNa02 L/R difference (Maimon lab, Nature 2024).
	// dθ/dt ∝ (DNa02R - DNa02L), positive = rightward.
	// DNa02 receives direct PFL3 (central complex) input.
	// Our coords: positive turn = left, so turn = (L - R)/sum.
	// The brain generates the rates via PFL3->DNa02; we decode with published sign.
	olfTurn := 0.0
	a02sum := f.dna02LRate + f.dna02RRate
	if a02sum > 10.0 {
		olfTurn = (f.dna02LRate - f.dna02RRate) / a02sum
	}
	olfTurn *= 1.5

	w.mu.Lock()
	defer w.mu.Unlock()
	f.heading += (turn + visTurn + olfTurn) * dt
	if f.reverseT > 0 {
		f.heading += 1.2 * dt // veer while backing up
		f.reverseT -= dt
	}
	if walking {
		nx := f.x + math.Cos(f.heading)*speed*dt
		nz := f.z + math.Sin(f.heading)*speed*dt
		if nx < -arenaHalf+2 || nx > arenaHalf-2 {
			f.heading = math.Pi - f.heading
		} else {
			f.x = nx
		}
		if nz < -arenaHalf+2 || nz > arenaHalf-2 {
			f.heading = -f.heading
		} else {
			f.z = nz
		}
		// Trunks and rocks: slide around, then back up and turn —
		// real flies reverse off obstacles.
		if nx2, nz2, hit := w.collide(f.x, f.z); hit {
			f.x, f.z = nx2, nz2
			f.heading += 0.35
			f.reverseT = 0.7
		}
		f.speed = speed
	} else {
		f.speed = 0
	}
	// NOTE: Battery/hunger system REMOVED. No artificial metabolism.
	// The brain must regulate behavior via real wiring.
	f.trail = append(f.trail, [2]float64{f.x, f.z})
	if len(f.trail) > 200 {
		f.trail = f.trail[len(f.trail)-200:]
	}
	f.walking = walking
	prevSpikes, prevMs := f.spikes, f.simMs
	f.steps, f.spikes = steps, tot
	f.simMs = float64(steps) * f.b.DtMs
	if f.simMs > prevMs {
		f.instHz = float64(tot-prevSpikes) / float64(f.b.N) / ((f.simMs - prevMs) / 1000.0)
	}
	return nil
}

func (w *World) snapshot() map[string]any {
	w.mu.Lock()
	defer w.mu.Unlock()
	flies := make([]map[string]any, 0, len(w.flies))
	for _, f := range w.flies {
		trail := make([][2]float64, len(f.trail))
		copy(trail, f.trail)
		state := "still"
		switch {
		case f.reverseT > 0:
			state = "reversing"
		case f.walking:
			state = "walking"
		}
		flies = append(flies, map[string]any{
			"name": f.name, "color": f.color,
			"x": f.x, "z": f.z, "h": f.heading,
			"battery": f.battery, "trail": trail,
			"state": state, "speed": f.speed,
			"antL": f.antL, "antR": f.antR,
			"p9L": f.p9LRate, "p9R": f.p9RRate,
			"pe046L": f.pe046LRate, "pe046R": f.pe046RRate,
			"dnb05": (f.dnb05LRate+f.dnb05RRate)/2, "dnb05L": f.dnb05LRate, "dnb05R": f.dnb05RRate, "dna02L": f.dna02LRate, "dna02R": f.dna02RRate, "pn": (f.pnLRate+f.pnRRate)/2, "pnL": f.pnLSm, "pnR": f.pnRSm, "mdn": f.mdnRate,
			"brain": map[string]any{
				"steps": f.steps, "simMs": f.simMs,
				"spikes": f.spikes, "hz": f.instHz,
			},
		})
	}
	foods := make([]map[string]any, 0, len(w.foods))
	for _, fd := range w.foods {
		foods = append(foods, map[string]any{"x": fd.x, "z": fd.z})
	}
	return map[string]any{
		"tick": w.tick, "paused": w.paused,
		"flies": flies, "foods": foods,
	}
}

// ---------------------------------------------------------------------------
// HTTP + SSE.
// ---------------------------------------------------------------------------

func main() {
	brainsFlag := flag.String("brains", "localhost:5555", "comma-separated flybrain server addrs (1-4, one fly each)")
	listen := flag.String("listen", ":8080", "web UI listen address")
	graphPath := flag.String("graph", "", "path to .fbc dump (hub sensory window + P9/sugar-GRN root-ID resolution)")
	connectome := flag.String("connectome", "fafb", "connectome ID set: fafb (FlyWire v783) or banc (BANC v888 female brain+VNC)")
	sensHzMax := flag.Float64("sens-hz-max", 200.0, "Poisson Hz at odor concentration 1.0 (servers must use --sens-mode poisson)")
	exploreHz := flag.Float64("explore-hz", 12.0, "baseline Poisson Hz on P9 inputs: spontaneous exploration")
	arousalHz := flag.Float64("arousal-hz", 150.0, "baseline Poisson Hz on DNp09 true inputs: central walk intent")
	stepMs := flag.Int("step-ms", 50, "brain ms per world tick")
	tickMs := flag.Int("tick-ms", 120, "world tick period in ms (wall clock, per fly)")
	pauseHz := flag.Float64("pause-hz", 6.0, "mean P9 rate (Hz) below which the fly sits still")
	maxSpeed := flag.Float64("max-speed", 14.0, "fly speed at full P9 drive (units/s)")
	seed := flag.Int64("seed", 7, "RNG seed for tree/rock scenery layout")
	flag.Parse()

	// Select connectome ID set.
	if *connectome == "banc" {
		p9RootIDs = banc_p9RootIDs
		sugarRootIDs = banc_sugarRootIDs
		ornFoodL = banc_ornFoodL
		ornFoodR = banc_ornFoodR
		pnFoodL = banc_pnFoodL
		pnFoodR = banc_pnFoodR
		visL = banc_visL
		visR = banc_visR
		cenL = banc_cenL
		cenR = banc_cenL // BANC right DNp09 mirrored from left (data gap); use same central drive
		dnb05Roots = banc_dnb05Roots
		mdnDrive = banc_mdnDrive
		// MDN roots for readout
		// (mdnIdx resolved from banc_mdnRoots below)
		fmt.Printf("Using BANC v888 connectome (female brain+VNC)\n")
	} else if *connectome != "fafb" {
		log.Fatalf("unknown --connectome %q (want fafb or banc)", *connectome)
	}

	addrs := strings.Split(*brainsFlag, ",")
	if len(addrs) < 1 || len(addrs) > 4 {
		log.Fatal("need 1-4 --brains addrs")
	}
	if *graphPath == "" {
		log.Fatal("--graph is required (P9/sugar-GRN root-ID resolution)")
	}
	g, roots, err := loadDumpRoots(*graphPath)
	if err != nil {
		log.Fatal(err)
	}
	// p9RootIDs = {right DNp09, left DNp09}; assign anatomically.
	p9r, err := resolveRoots(roots, p9RootIDs)
	if err != nil {
		log.Fatal("P9: ", err)
	}
	p9R, p9L := p9r[0], p9r[1]
	// Genuine chemotaxis DNs (BANC only for now)
	pe046L, pe046R := -1, -1
	if *connectome == "banc" {
		pe046r, err := resolveRoots(roots, []uint64{banc_DNpe046_R, banc_DNpe046_L})
		if err != nil {
			log.Fatal("DNpe046: ", err)
		}
		pe046R, pe046L = pe046r[0], pe046r[1]
		fmt.Printf("DNpe046: L=%d R=%d (genuine olfactory steering)\n", pe046L, pe046R)
	}
	// DNa02: THE steering DN (Maimon lab, Nature 2024)
	// dθ/dt ∝ (DNa02R - DNa02L), positive = rightward
	dna02L, dna02R := -1, -1
	if *connectome == "banc" {
		dna02r, err := resolveRoots(roots, []uint64{banc_DNa02_R, banc_DNa02_L})
		if err != nil {
			log.Fatal("DNa02: ", err)
		}
		dna02R, dna02L = dna02r[0], dna02r[1]
		fmt.Printf("DNa02: L=%d R=%d (steering DN, Maimon lab)\n", dna02L, dna02R)
	}
	// PFL3: central complex output to DNa02 (goal navigation)
	pfl3L, pfl3R := []int{}, []int{}
	if *connectome == "banc" {
		pfl3lr, err := resolveRoots(roots, banc_PFL3_L)
		if err != nil {
			log.Fatal("PFL3_L: ", err)
		}
		pfl3L = pfl3lr
		pfl3rr, err := resolveRoots(roots, banc_PFL3_R)
		if err != nil {
			log.Fatal("PFL3_R: ", err)
		}
		pfl3R = pfl3rr
		fmt.Printf("PFL3: L=%d R=%d neurons\n", len(pfl3L), len(pfl3R))
	}
	sugar, err := resolveRoots(roots, sugarRootIDs)
	if err != nil {
		log.Fatal("sugar GRNs: ", err)
	}
	ornL, err := resolveRoots(roots, ornFoodL)
	if err != nil {
		log.Fatal("ORN-L: ", err)
	}
	ornR, err := resolveRoots(roots, ornFoodR)
	if err != nil {
		log.Fatal("ORN-R: ", err)
	}
	pnL, err := resolveRoots(roots, pnFoodL)
	if err != nil {
		log.Fatal("PN-L: ", err)
	}
	pnR, err := resolveRoots(roots, pnFoodR)
	if err != nil {
		log.Fatal("PN-R: ", err)
	}
	dnb05, err := resolveRoots(roots, dnb05Roots)
	if err != nil {
		log.Fatal("DNb05: ", err)
	}
	mdnRoots := []uint64{720575940640331472, 720575940610236514, 720575940631082808, 720575940616026939}
	if *connectome == "banc" {
		mdnRoots = []uint64{720575941491012809, 720575941597827697, 720575941690515574, 720575941592674814}
	}
	mdnIdx, err := resolveRoots(roots, mdnRoots)
	if err != nil {
		log.Fatal("MDN: ", err)
	}
	visLi, err := resolveRoots(roots, visL)
	if err != nil {
		log.Fatal("vis-L: ", err)
	}
	visRi, err := resolveRoots(roots, visR)
	if err != nil {
		log.Fatal("vis-R: ", err)
	}
	cenLi, err := resolveRoots(roots, cenL)
	if err != nil {
		log.Fatal("cen-L: ", err)
	}
	cenRi, err := resolveRoots(roots, cenR)
	if err != nil {
		log.Fatal("cen-R: ", err)
	}
	mdnDi, err := resolveRoots(roots, mdnDrive)
	if err != nil {
		log.Fatal("mdnDrive: ", err)
	}
	// sensIdx: ORN-L ++ ORN-R ++ vis-L ++ vis-R ++ cen-L ++ cen-R ++ mdnDrive ++ sugar
	sensIdx := append(append(append(append(append(append(append([]int{}, ornL...), ornR...), visLi...), visRi...), cenLi...), cenRi...), mdnDi...)
	sensIdx = append(sensIdx, sugar...)
	// PFL3 for central complex goal navigation (appended at end)
	sensIdx = append(sensIdx, pfl3L...)
	sensIdx = append(sensIdx, pfl3R...)
	fmt.Printf("DNp09: L=%d R=%d | DNb05: L=%d R=%d | ORNs: L=%d R=%d | PNs: L=%d R=%d | vis: L=%d R=%d | cen: L=%d R=%d | mdnDrive: %d | sugar: %d\n",
		p9L, p9R, dnb05[0], dnb05[1], len(ornL), len(ornR), len(pnL), len(pnR), len(visLi), len(visRi), len(cenLi), len(cenRi), len(mdnDi), len(sugar))

	o0 := 0
	o1 := o0 + len(ornL)
	o2 := o1 + len(ornR)
	v0 := o2
	v1 := v0 + len(visLi)
	v2 := v1 + len(visRi)
	c0 := v2
	c1 := c0 + len(cenLi)
	c2 := c1 + len(cenRi)
	m0 := c2
	m1 := m0 + len(mdnDi)
	// PFL3 offsets (after sugar)
	p0 := m1 + len(sugar)
	p1 := p0 + len(pfl3L)
	p2 := p1 + len(pfl3R)
	w := &World{
		foods:   []Food{{x: 30, z: 0}},
		sensIdx: sensIdx,
		ornL0:   o0, ornL1: o1,
		ornR0:   o1, ornR1: o2,
		visL0:   v0, visL1: v1,
		visR0:   v1, visR1: v2,
		cenL0:   c0, cenL1: c1,
		cenR0:   c1, cenR1: c2,
		mdn0:    m0, mdn1: m1,
		pfl3L0:  p0, pfl3L1: p1,
		pfl3R0:  p1, pfl3R1: p2,
		taste0:  m1,
		sensHzMax: *sensHzMax,
		arousalHz: *arousalHz,
		exploreHz: *exploreHz,
		stepMs:    *stepMs,
		pauseHz:   *pauseHz,
		maxSpeed:  *maxSpeed,
	}
	w.trees, w.rocks = genScenery(*seed)
	fmt.Printf("scenery: %d trees, %d rocks (seed %d)\n", len(w.trees), len(w.rocks), *seed)
	starts := [][2]float64{{-30, -20}, {-30, 20}, {-30, 0}, {0, -30}}
	for i, addr := range addrs {
		addr = strings.TrimSpace(addr)
		b, err := DialBrain(addr)
		if err != nil {
			log.Fatalf("dial brain %d (%s): %v", i+1, addr, err)
		}
		fmt.Printf("brain %d: %s N=%d dt=%.2fms\n", i+1, addr, b.N, b.DtMs)
		if b.N != g.N {
			log.Fatalf("brain %d has N=%d but graph has N=%d", i+1, b.N, g.N)
		}
		w.flies = append(w.flies, &Fly{
			name: flyNames[i], color: flyColors[i], b: b,
			x: starts[i][0], z: starts[i][1],
			heading: 0, battery: 1,
			p9L: p9L, p9R: p9R,
			pe046L: pe046L, pe046R: pe046R,
			dna02L: dna02L, dna02R: dna02R,
			dnb05L: dnb05[0], dnb05R: dnb05[1],
			mdnIdx: mdnIdx,
			pnL: pnL, pnR: pnR,
			pnLC: make([]uint64, len(pnL)), pnRC: make([]uint64, len(pnR)),
		})
	}

	// World loop: tick each fly in turn. A dead brain connection is
	// redialed (servers restart for upgrades); the fly sits out ticks
	// while its brain is unreachable.
	go func() {
		tk := time.NewTicker(time.Duration(*tickMs) * time.Millisecond)
		defer tk.Stop()
		backoff := map[string]time.Time{}
		for range tk.C {
			w.mu.Lock()
			paused := w.paused
			w.mu.Unlock()
			if paused {
				continue
			}
			for _, f := range w.flies {
				if err := w.advance(f); err != nil {
					if time.Since(backoff[f.name]) > 5*time.Second {
						log.Printf("fly %s: %v (redialing %s)", f.name, err, f.b.addr)
						backoff[f.name] = time.Now()
					}
					if rerr := f.b.reconnect(); rerr == nil {
						// Re-apply persistent drive after a fresh server.
						f.p9LC, f.p9RC, f.p9Ms = 0, 0, 0
					}
				}
			}
			w.mu.Lock()
			w.tick++
			w.mu.Unlock()
		}
	}()

	web, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(web)))
	mux.HandleFunc("/events", func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Type", "text/event-stream")
		rw.Header().Set("Cache-Control", "no-cache")
		rw.Header().Set("Connection", "keep-alive")
		fl, _ := rw.(http.Flusher)
		tk := time.NewTicker(150 * time.Millisecond)
		defer tk.Stop()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-tk.C:
				js, _ := json.Marshal(w.snapshot())
				fmt.Fprintf(rw, "data: %s\n\n", js)
				fl.Flush()
			}
		}
	})
	mux.HandleFunc("/api/pause", func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		w.paused = !w.paused
		p := w.paused
		w.mu.Unlock()
		fmt.Fprintf(rw, `{"paused":%v}`, p)
	})
	mux.HandleFunc("/api/reset", func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		for i, f := range w.flies {
			f.x, f.z, f.heading = starts[i][0], starts[i][1], 0
			f.battery = 1
			f.trail = f.trail[:0]
		}
		w.foods = []Food{{x: 30, z: 0}}
		w.mu.Unlock()
		fmt.Fprint(rw, `{"ok":true}`)
	})
	mux.HandleFunc("/api/addfood", func(rw http.ResponseWriter, r *http.Request) {
		var x, z float64
		fmt.Sscanf(r.URL.Query().Get("x"), "%f", &x)
		fmt.Sscanf(r.URL.Query().Get("z"), "%f", &z)
		if x < -arenaHalf || x > arenaHalf || z < -arenaHalf || z > arenaHalf {
			http.Error(rw, "out of bounds", 400)
			return
		}
		w.mu.Lock()
		w.foods = append(w.foods, Food{x: x, z: z})
		w.mu.Unlock()
		fmt.Fprint(rw, `{"ok":true}`)
	})
	mux.HandleFunc("/api/clearfoods", func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		w.foods = nil
		w.mu.Unlock()
		fmt.Fprint(rw, `{"ok":true}`)
	})
	mux.HandleFunc("/api/scene", func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		defer w.mu.Unlock()
		trees := make([]map[string]any, 0, len(w.trees))
		for _, t := range w.trees {
			trees = append(trees, map[string]any{"x": t.X, "z": t.Z, "h": t.H})
		}
		rocks := make([]map[string]any, 0, len(w.rocks))
		for _, rk := range w.rocks {
			rocks = append(rocks, map[string]any{"x": rk.X, "z": rk.Z, "r": rk.R})
		}
		js, _ := json.Marshal(map[string]any{"trees": trees, "rocks": rocks})
		rw.Header().Set("Content-Type", "application/json")
		rw.Write(js)
	})
	fmt.Printf("flyworld: %d flies, UI on http://%s\n", len(w.flies), *listen)
	log.Fatal(http.ListenAndServe(*listen, mux))
}
