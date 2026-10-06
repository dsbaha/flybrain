// Command flyworld: multi-brain 3D world simulator for flybrain.
//
// One binary (web UI embedded via fs.Embed). Connects to 1-4 flybrain
// servers — one embodied fly per brain — sharing a 3D arena with food
// odor plumes. Each tick: sample odor at the fly's antennae -> Poisson
// drive a hub-neuron sensory window -> step the brain -> read the P9
// descending-neuron pair (the reference forward-walking command neurons)
// -> steer.
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
func topP9Inputs(g *Graph, p9, k int) []int {
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
	for i := 0; i < k; i++ {
		out[i] = all[i].i
	}
	return out
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
	plumeSig  = 22.0 // odor plume sigma
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

	p9L, p9R       int
	p9LRate, p9RRate float64 // Hz, from cumulative spike counts
	p9LC, p9RC     uint64   // previous cumulative counts
	p9Ms           float64  // sim ms at previous read
	antL, antR    float64    // odor 0..1 at antennae
	speed         float64
	walking       bool
	feeding       bool

	steps  uint64
	simMs  float64
	spikes uint64
	instHz float64
}

type Food struct {
	x, z float64
}

type World struct {
	mu     sync.Mutex
	paused bool
	flies  []*Fly
	foods  []Food
	phase  float64
	tick   uint64

	// io config
	sensIdx   []int   // P9L top inputs ++ P9R top inputs ++ sugar GRNs
	sensHalf  int     // left/right split
	sensTaste int     // offset where sugar-GRN indices start
	sensHzMax float64
	exploreHz float64 // baseline Poisson Hz: spontaneous exploration
	stepMs    int
	pauseHz   float64 // P9 mean rate below this -> sit still
	maxSpeed  float64
}

func (w *World) odor(x, z float64) float64 {
	w.phase += 0 // phase advanced once per tick in advance()
	c := 0.0
	for _, f := range w.foods {
		dx, dz := x-f.x, z-f.z
		c += math.Exp(-(dx*dx + dz*dz) / (2 * plumeSig * plumeSig))
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
	hunger := 1.0 - f.battery
	gain := 0.4 + hunger
	// Bilateral odor drive on P9's strongest excitatory inputs:
	// left antenna -> P9L inputs, right antenna -> P9R inputs.
	rateL := w.exploreHz + math.Min(1, al)*w.sensHzMax*gain
	rateR := w.exploreHz + math.Min(1, ar)*w.sensHzMax*gain
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
	for i := 0; i < w.sensHalf; i++ {
		vals[i] = float32(rateL)
	}
	for i := w.sensHalf; i < w.sensTaste; i++ {
		vals[i] = float32(rateR)
	}
	taste := float32(0)
	if feeding {
		taste = 200
	}
	for i := w.sensTaste; i < len(w.sensIdx); i++ {
		vals[i] = taste
	}
	f.feeding = feeding
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
	cl, err := f.b.Spkc(f.p9L, 1)
	if err != nil {
		return err
	}
	cr, err := f.b.Spkc(f.p9R, 1)
	if err != nil {
		return err
	}
	// Exact P9 rates from cumulative counts over the elapsed sim time.
	// (A count drop means the server restarted: re-baseline.)
	if simMs > f.p9Ms && cl[0] >= f.p9LC && cr[0] >= f.p9RC {
		dt := (simMs - f.p9Ms) / 1000.0
		f.p9LRate = float64(cl[0]-f.p9LC) / dt
		f.p9RRate = float64(cr[0]-f.p9RC) / dt
	}
	f.p9LC, f.p9RC, f.p9Ms = cl[0], cr[0], simMs
	mean := (f.p9LRate + f.p9RRate) / 2

	// Stop-and-go on P9 drive; turning stays live (look-around saccades).
	// Turn toward the faster P9: ipsilateral P9 drive steers that way.
	speed, walking := 0.0, false
	if mean > w.pauseHz {
		speed = math.Min(1, (mean-w.pauseHz)/120.0) * w.maxSpeed
		walking = true
	}
	turn := (f.p9LRate - f.p9RRate) / 120.0 * 2.5
	dt := float64(w.stepMs) / 1000.0

	w.mu.Lock()
	defer w.mu.Unlock()
	f.heading += turn * dt
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
		f.speed = speed
	} else {
		f.speed = 0
	}
	if feeding {
		f.battery = math.Min(1, f.battery+0.02)
	} else {
		f.battery = math.Max(0, f.battery-0.0006)
	}
	if f.battery <= 0 {
		f.x, f.z, f.heading = -30, 0, 0
		f.battery = 1
		f.trail = f.trail[:0]
	}
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
		case f.feeding:
			state = "feeding"
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
	sensTopK := flag.Int("sens-topk", 32, "strongest excitatory P9 inputs driven per side (bilateral odor)")
	sensHzMax := flag.Float64("sens-hz-max", 200.0, "Poisson Hz at odor concentration 1.0 (servers must use --sens-mode poisson)")
	exploreHz := flag.Float64("explore-hz", 12.0, "baseline Poisson Hz on P9 inputs: spontaneous exploration")
	stepMs := flag.Int("step-ms", 50, "brain ms per world tick")
	tickMs := flag.Int("tick-ms", 120, "world tick period in ms (wall clock, per fly)")
	pauseHz := flag.Float64("pause-hz", 6.0, "mean P9 rate (Hz) below which the fly sits still")
	maxSpeed := flag.Float64("max-speed", 14.0, "fly speed at full P9 drive (units/s)")
	flag.Parse()

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
	p9, err := resolveRoots(roots, p9RootIDs)
	if err != nil {
		log.Fatal("P9: ", err)
	}
	sugar, err := resolveRoots(roots, sugarRootIDs)
	if err != nil {
		log.Fatal("sugar GRNs: ", err)
	}
	inL := topP9Inputs(g, p9[0], *sensTopK)
	inR := topP9Inputs(g, p9[1], *sensTopK)
	sensIdx := append(append(append([]int{}, inL...), inR...), sugar...)
	fmt.Printf("P9L=%d (%d inputs) P9R=%d (%d inputs)  sugar GRNs: %d mapped\n",
		p9[0], len(inL), p9[1], len(inR), len(sugar))

	w := &World{
		foods:     []Food{{x: 30, z: 0}},
		sensIdx:   sensIdx,
		sensHalf:  len(inL),
		sensTaste: len(inL) + len(inR),
		sensHzMax: *sensHzMax,
		exploreHz: *exploreHz,
		stepMs:    *stepMs,
		pauseHz:   *pauseHz,
		maxSpeed:  *maxSpeed,
	}
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
			p9L: p9[0], p9R: p9[1],
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
	fmt.Printf("flyworld: %d flies, UI on http://%s\n", len(w.flies), *listen)
	log.Fatal(http.ListenAndServe(*listen, mux))
}
