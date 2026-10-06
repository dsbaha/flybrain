// Package main implements flyviz: a single-binary web visualizer for a
// flybrain TCP server.
//
// It runs a tiny 2D world (an agent + an odor food source), feeds the
// agent's bilateral antennae into the brain via SENS, steps the brain,
// decodes two motor-neuron populations into movement, and streams the
// whole thing to the browser over server-sent events. All web assets are
// embedded via fs.Embed: one binary, no extra files.
//
// I/O mapping: with --graph, sensory neurons are the contiguous window
// with max out-degree ("hubs", for strongest broadcast) and the two
// motor populations are auto-mapped as the strongest net-excitatory
// downstream of the left/right sensory halves in the real wiring.
// Without --graph, raw --sens-off/--motor-off indices are used.
// Cell-type-accurate mapping (real ORNs -> PNs -> KCs -> DNs) still
// needs the FlyWire annotations; the UI says so.
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
	"strings"
	"sync"
	"time"
)

//go:embed web
var webFS embed.FS

// ---------------------------------------------------------------------------
// Minimal copy of the flybrain TCP protocol (see server.go in the main pkg).
// ---------------------------------------------------------------------------

func writeFrame(w io.Writer, tag string, payload []byte) error {
	hdr := make([]byte, 8)
	copy(hdr[:4], tag)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(len(payload)))
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	if len(payload) > 0 {
		_, err := w.Write(payload)
		return err
	}
	return nil
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
	if _, err := io.ReadFull(r, p); err != nil {
		return "", nil, err
	}
	return string(hdr[:4]), p, nil
}

// Brain is a thin client of the flybrain server protocol.
type Brain struct {
	conn net.Conn
	br   *bufio.Reader
	N    int
	DtMs float64
	mu   sync.Mutex
}

func DialBrain(addr string) (*Brain, error) {
	c, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, err
	}
	b := &Brain{conn: c, br: bufio.NewReader(c)}
	line, err := b.br.ReadString('\n')
	if err != nil {
		c.Close()
		return nil, err
	}
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "FLYBRAIN/1 ") {
		c.Close()
		return nil, fmt.Errorf("bad handshake: %q", line)
	}
	for _, f := range strings.Fields(line[len("FLYBRAIN/1 "):]) {
		if _, err := fmt.Sscanf(f, "N=%d", &b.N); err == nil {
			continue
		}
		var v float64
		if _, err := fmt.Sscanf(f, "DT=%f", &v); err == nil {
			b.DtMs = v
		}
	}
	if b.N <= 0 {
		c.Close()
		return nil, fmt.Errorf("bad handshake: %q", line)
	}
	return b, nil
}

func (b *Brain) req(tag string, payload []byte) (string, []byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := writeFrame(b.conn, tag, payload); err != nil {
		return "", nil, err
	}
	return readFrame(b.br)
}

func (b *Brain) Sens(offset int, vals []float32) error {
	p := make([]byte, 4+len(vals)*4)
	binary.LittleEndian.PutUint32(p, uint32(offset))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(p[4+i*4:], math.Float32bits(v))
	}
	tag, _, err := b.req("SENS", p)
	if err != nil {
		return err
	}
	if tag != "OKAY" {
		return fmt.Errorf("SENS: %q", tag)
	}
	return nil
}

func (b *Brain) Step(n int) (totSpikes, simSteps uint64, err error) {
	p := make([]byte, 4)
	binary.LittleEndian.PutUint32(p, uint32(n))
	tag, out, err := b.req("STEP", p)
	if err != nil || tag != "DONE" || len(out) != 16 {
		return 0, 0, fmt.Errorf("STEP: %v %q", err, tag)
	}
	return binary.LittleEndian.Uint64(out[:8]),
		binary.LittleEndian.Uint64(out[8:]), nil
}

func (b *Brain) ReadVolt(start, count int) ([]float32, error) {
	p := make([]byte, 8)
	binary.LittleEndian.PutUint32(p, uint32(start))
	binary.LittleEndian.PutUint32(p[4:], uint32(count))
	tag, out, err := b.req("READ", p)
	if err != nil || tag != "VOLT" || len(out) != 8+count*4 {
		return nil, fmt.Errorf("READ: %v %q", err, tag)
	}
	vals := make([]float32, count)
	for i := range vals {
		vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(out[8+i*4:]))
	}
	return vals, nil
}

func (b *Brain) Stat() (steps uint64, simMs float64, tot uint64, err error) {
	tag, out, err := b.req("STAT", nil)
	if err != nil || tag != "STAT" || len(out) != 32 {
		return 0, 0, 0, fmt.Errorf("STAT: %v %q", err, tag)
	}
	return binary.LittleEndian.Uint64(out[:8]),
		math.Float64frombits(binary.LittleEndian.Uint64(out[8:])),
		binary.LittleEndian.Uint64(out[16:]), nil
}

// ---------------------------------------------------------------------------
// .fbc loading (minimal copy of the loader in the main package).
// ---------------------------------------------------------------------------

type Graph struct {
	N      int
	RowPtr []uint32
	Col    []uint32
	W      []int32
}

func loadDump(path string) (*Graph, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	magic := make([]byte, 8)
	if _, err := io.ReadFull(f, magic); err != nil || string(magic) != "FLYBRAIN" {
		return nil, fmt.Errorf("not a flybrain dump: %s", path)
	}
	rd := func(v any) error { return binary.Read(f, binary.LittleEndian, v) }
	var ver uint32
	var nu, eu uint64
	if err := rd(&ver); err != nil {
		return nil, err
	}
	if err := rd(&nu); err != nil {
		return nil, err
	}
	if err := rd(&eu); err != nil {
		return nil, err
	}
	n, e := int(nu), int(eu)
	g := &Graph{N: n}
	g.RowPtr = make([]uint32, n+1)
	g.Col = make([]uint32, e)
	g.W = make([]int32, e)
	if err := rd(&g.RowPtr); err != nil {
		return nil, err
	}
	if err := rd(&g.Col); err != nil {
		return nil, err
	}
	if err := rd(&g.W); err != nil {
		return nil, err
	}
	return g, nil
}

// bestSensoryWindow finds the contiguous n-window with max total out-degree:
// hub neurons, for strongest broadcast of the odor signal.
func bestSensoryWindow(g *Graph, n int) int {
	best, bestSum := 0, uint64(0)
	var sum uint64
	for i := 0; i < n && i < g.N; i++ {
		sum += uint64(g.RowPtr[i+1] - g.RowPtr[i])
	}
	bestSum = sum
	for off := 1; off+n <= g.N; off++ {
		sum += uint64(g.RowPtr[off+n]-g.RowPtr[off+n-1]) - uint64(g.RowPtr[off]-g.RowPtr[off-1])
		if sum > bestSum {
			bestSum, best = sum, off
		}
	}
	return best
}

// autoMapMotor picks a contiguous window of motorN neurons with the
// strongest net excitatory influence from a sensory range, walking the
// real wiring up to hops out-edges. Signed weights: disinhibition counts
// as excitation, so the odor arouses rather than silences the readout.
func autoMapMotor(g *Graph, sensOff, sensN, motorN, hops int) int {
	n := g.N
	score := make([]float64, n)
	frontier := map[int]float64{}
	for s := sensOff; s < sensOff+sensN; s++ {
		frontier[s] = 1.0
	}
	visited := make([]bool, n)
	for h := 0; h < hops; h++ {
		next := map[int]float64{}
		for src, f := range frontier {
			if visited[src] {
				continue
			}
			visited[src] = true
			for k := g.RowPtr[src]; k < g.RowPtr[src+1]; k++ {
				dst := int(g.Col[k])
				add := f * float64(g.W[k]) / 1024.0
				score[dst] += add
				next[dst] += add
			}
		}
		frontier = next
	}
	bestOff, bestScore := 0, math.Inf(-1)
	for off := 0; off+motorN <= n; off++ {
		if off < sensOff+sensN && off+motorN > sensOff {
			continue // keep clear of the sensory window
		}
		s := 0.0
		for i := 0; i < motorN; i++ {
			s += score[off+i]
		}
		if s > bestScore {
			bestScore, bestOff = s, off
		}
	}
	return bestOff
}

// ---------------------------------------------------------------------------
// 2D world: agent + odor food source.
// ---------------------------------------------------------------------------

const (
	worldW = 100.0
	worldH = 100.0
)

type World struct {
	mu      sync.Mutex
	paused  bool
	ax, ay  float64
	heading float64
	bx, by  float64
	battery float64
	trail   [][2]float64
	phase   float64

	antL, antR     float32
	hunger         float32
	popL, popR     float32
	meanVL, meanVR float32
	walking        bool

	steps  uint64
	simMs  float64
	spikes uint64
	instHz float64
	tick   uint64
}

func NewWorld() *World {
	return &World{
		ax: 20, ay: 50, heading: 0,
		bx: 80, by: 50,
		battery: 1,
	}
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// odor returns 0..~1 odor concentration at (x,y): Gaussian plume with
// a slow flicker, like a turbulent food-odor plume averaged over time.
func (w *World) odor(x, y float64) float64 {
	dx, dy := x-w.bx, y-w.by
	base := math.Exp(-(dx*dx + dy*dy) / (2 * 25 * 25))
	flick := 0.85 + 0.15*math.Sin(w.phase+x*0.05)*math.Sin(w.phase*0.7+y*0.06)
	return base * flick
}

type ioMap struct {
	sensOff, sensN       int
	motorLOff, motorROff int
	motorN               int // per-side population
	sensScale            float64
	stepMs               int
	pauseThresh          float64
}

// popStats: mean 0..1 activation and mean voltage of a population.
func popStats(v []float32) (act, meanV float64) {
	for _, x := range v {
		a := (float64(x) + 70.0) / 20.0
		if a < 0 {
			a = 0
		} else if a > 1 {
			a = 1
		}
		act += a
		meanV += float64(x)
	}
	return act / float64(len(v)), meanV / float64(len(v))
}

// advance runs one world tick: smell -> brain -> move (or pause).
func (w *World) advance(b *Brain, io ioMap) error {
	w.mu.Lock()
	paused := w.paused
	w.mu.Unlock()
	if paused {
		return nil
	}

	w.mu.Lock()
	w.phase += 0.12
	hx, hy := math.Cos(w.heading), math.Sin(w.heading)
	px, py := -hy, hx
	// antenna tips: 3 ahead, +/-1.5 lateral
	al := w.odor(w.ax+hx*3+px*1.5, w.ay+hy*3+py*1.5)
	ar := w.odor(w.ax+hx*3-px*1.5, w.ay+hy*3-py*1.5)
	hunger := 1.0 - w.battery
	gain := 0.4 + hunger // hungry flies are more odor-responsive
	n2 := io.sensN / 2
	vals := make([]float32, io.sensN)
	for i := 0; i < n2; i++ {
		vals[i] = float32(al * io.sensScale * gain)
	}
	for i := n2; i < io.sensN; i++ {
		vals[i] = float32(ar * io.sensScale * gain)
	}
	w.mu.Unlock()

	if err := b.Sens(io.sensOff, vals); err != nil {
		return err
	}
	tot, steps, err := b.Step(io.stepMs)
	if err != nil {
		return err
	}
	voltL, err := b.ReadVolt(io.motorLOff, io.motorN)
	if err != nil {
		return err
	}
	voltR, err := b.ReadVolt(io.motorROff, io.motorN)
	if err != nil {
		return err
	}
	popL, meanVL := popStats(voltL)
	popR, meanVR := popStats(voltR)
	mean := (popL + popR) / 2

	// Stop-and-go: below-threshold descending drive -> sit still.
	// Turn stays live, so pauses become look-around saccades.
	speed, walking := 0.0, false
	if mean > io.pauseThresh {
		speed = (mean - io.pauseThresh) / (1 - io.pauseThresh) * 14.0
		walking = true
	}
	turn := (popR - popL) * 3.0
	dt := float64(io.stepMs) / 1000.0

	w.mu.Lock()
	defer w.mu.Unlock()
	w.heading += turn * dt
	if walking {
		nx := w.ax + math.Cos(w.heading)*speed*dt
		ny := w.ay + math.Sin(w.heading)*speed*dt
		if nx < 2 || nx > worldW-2 {
			w.heading = math.Pi - w.heading
		} else {
			w.ax = nx
		}
		if ny < 2 || ny > worldH-2 {
			w.heading = -w.heading
		} else {
			w.ay = ny
		}
	}
	dx, dy := w.ax-w.bx, w.ay-w.by
	if dx*dx+dy*dy < 100 {
		w.battery = clamp(w.battery+0.03, 0, 1)
	} else {
		w.battery = clamp(w.battery-0.0008, 0, 1)
	}
	if w.battery <= 0 {
		w.ax, w.ay, w.heading = 20, 50, 0
		w.battery = 1
		w.trail = w.trail[:0]
	}
	w.trail = append(w.trail, [2]float64{w.ax, w.ay})
	if len(w.trail) > 240 {
		w.trail = w.trail[len(w.trail)-240:]
	}
	w.antL = vals[0]
	w.antR = vals[n2]
	w.hunger = float32(hunger)
	w.popL, w.popR = float32(popL), float32(popR)
	w.meanVL, w.meanVR = float32(meanVL), float32(meanVR)
	w.walking = walking
	prevSpikes, prevMs := w.spikes, w.simMs
	w.steps, w.spikes = steps, tot
	w.simMs = float64(steps) * b.DtMs
	if w.simMs > prevMs {
		w.instHz = float64(tot-prevSpikes) / float64(b.N) / ((w.simMs - prevMs) / 1000.0)
	}
	w.tick++
	return nil
}

func (w *World) snapshot() map[string]any {
	w.mu.Lock()
	defer w.mu.Unlock()
	trail := make([][2]float64, len(w.trail))
	copy(trail, w.trail)
	state := "paused"
	if w.walking {
		state = "walking"
	}
	return map[string]any{
		"tick":    w.tick,
		"paused":  w.paused,
		"state":   state,
		"agent":   map[string]any{"x": w.ax, "y": w.ay, "h": w.heading},
		"beacon":  map[string]any{"x": w.bx, "y": w.by},
		"battery": w.battery,
		"trail":   trail,
		"sens":    []float32{w.antL, w.antR},
		"hunger":  w.hunger,
		"motor":   []float32{w.popL, w.popR},
		"motorV":  []float32{w.meanVL, w.meanVR},
		"brain": map[string]any{
			"steps": w.steps, "simMs": w.simMs,
			"spikes": w.spikes, "hz": w.instHz,
		},
	}
}

func main() {
	brainAddr := flag.String("brain", "localhost:5555", "flybrain server address")
	listen := flag.String("listen", ":8080", "web UI listen address")
	sensOff := flag.Int("sens-off", 0, "first sensory neuron index (unused with --graph)")
	sensN := flag.Int("sens-n", 32, "sensory channels (half left antenna, half right)")
	motorN := flag.Int("motor-n", 32, "motor neurons per side population")
	sensScale := flag.Float64("sens-scale", 20.0, "mV per unit odor concentration")
	stepMs := flag.Int("step-ms", 20, "brain ms per world tick")
	tickMs := flag.Int("tick-ms", 100, "world tick period in ms (wall clock)")
	pauseThresh := flag.Float64("pause-thresh", 0.15, "mean motor activation below which the fly sits still")
	graphPath := flag.String("graph", "", "path to .fbc dump: enables hub sensory + auto-mapped motor populations")
	mapHops := flag.Int("map-hops", 2, "hop depth for motor auto-map")
	flag.Parse()

	io := ioMap{
		sensOff: *sensOff, sensN: *sensN,
		motorN: *motorN, sensScale: *sensScale,
		stepMs: *stepMs, pauseThresh: *pauseThresh,
	}

	if *graphPath != "" {
		log.Printf("loading %s ...", *graphPath)
		g, err := loadDump(*graphPath)
		if err != nil {
			log.Fatalf("graph: %v", err)
		}
		io.sensOff = bestSensoryWindow(g, io.sensN)
		n2 := io.sensN / 2
		io.motorLOff = autoMapMotor(g, io.sensOff, n2, io.motorN, *mapHops)
		io.motorROff = autoMapMotor(g, io.sensOff+n2, n2, io.motorN, *mapHops)
		log.Printf("mapping: sensory [%d,%d) (hub window) -> motorL [%d,%d) motorR [%d,%d)",
			io.sensOff, io.sensOff+io.sensN,
			io.motorLOff, io.motorLOff+io.motorN,
			io.motorROff, io.motorROff+io.motorN)
	} else {
		io.motorLOff, io.motorROff = 0, 0
		log.Printf("no --graph: using raw offsets (sens %d, motor %d/%d) - placeholder",
			io.sensOff, io.motorLOff, io.motorROff)
	}

	log.Printf("connecting to brain at %s ...", *brainAddr)
	brain, err := DialBrain(*brainAddr)
	if err != nil {
		log.Fatalf("brain: %v", err)
	}
	log.Printf("brain: N=%d dt=%.2fms", brain.N, brain.DtMs)
	if io.sensOff+io.sensN > brain.N ||
		io.motorLOff+io.motorN > brain.N || io.motorROff+io.motorN > brain.N {
		log.Fatalf("I/O range exceeds N=%d", brain.N)
	}

	w := NewWorld()

	go func() {
		tk := time.NewTicker(time.Duration(*tickMs) * time.Millisecond)
		defer tk.Stop()
		for range tk.C {
			if err := w.advance(brain, io); err != nil {
				log.Printf("brain error: %v (retrying...)", err)
				time.Sleep(2 * time.Second)
				if nb, err := DialBrain(*brainAddr); err == nil {
					brain = nb
					log.Printf("reconnected to brain")
				}
			}
		}
	}()

	sub, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(sub)))

	var subsMu sync.Mutex
	subs := map[chan []byte]bool{}
	go func() {
		tk := time.NewTicker(100 * time.Millisecond)
		defer tk.Stop()
		for range tk.C {
			data, _ := json.Marshal(w.snapshot())
			msg := append(append([]byte("data: "), data...), '\n', '\n')
			subsMu.Lock()
			for ch := range subs {
				select {
				case ch <- msg:
				default:
				}
			}
			subsMu.Unlock()
		}
	}()
	mux.HandleFunc("/events", func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Type", "text/event-stream")
		rw.Header().Set("Cache-Control", "no-cache")
		rw.Header().Set("Connection", "keep-alive")
		ch := make(chan []byte, 4)
		subsMu.Lock()
		subs[ch] = true
		subsMu.Unlock()
		defer func() {
			subsMu.Lock()
			delete(subs, ch)
			subsMu.Unlock()
		}()
		fl, _ := rw.(http.Flusher)
		for {
			select {
			case msg := <-ch:
				rw.Write(msg)
				if fl != nil {
					fl.Flush()
				}
			case <-r.Context().Done():
				return
			}
		}
	})

	mux.HandleFunc("/api/beacon", func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(rw, "POST only", 405)
			return
		}
		var p struct{ X, Y float64 }
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			http.Error(rw, "bad json", 400)
			return
		}
		w.mu.Lock()
		w.bx = clamp(p.X, 2, worldW-2)
		w.by = clamp(p.Y, 2, worldH-2)
		w.mu.Unlock()
		rw.WriteHeader(204)
	})
	mux.HandleFunc("/api/reset", func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		*w = *NewWorld()
		w.mu.Unlock()
		rw.WriteHeader(204)
	})
	mux.HandleFunc("/api/pause", func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		w.paused = !w.paused
		p := w.paused
		w.mu.Unlock()
		json.NewEncoder(rw).Encode(map[string]any{"paused": p})
	})
	mux.HandleFunc("/api/info", func(rw http.ResponseWriter, r *http.Request) {
		note := fmt.Sprintf(
			"odor plume -> %d antenna channels (L/R) at [%d,%d), hunger-gated gain; "+
				"motor <- 2x%d populations at [%d,%d)/[%d,%d), strongest net-excitatory "+
				"downstream in the real wiring. Cell-type-accurate mapping pending annotations.",
			io.sensN, io.sensOff, io.sensOff+io.sensN, io.motorN,
			io.motorLOff, io.motorLOff+io.motorN, io.motorROff, io.motorROff+io.motorN)
		if *graphPath == "" {
			note = "placeholder I/O mapping (no --graph). Cell-type-accurate mapping pending annotations."
		}
		json.NewEncoder(rw).Encode(map[string]any{
			"brainN": brain.N, "sensOff": io.sensOff, "sensN": io.sensN,
			"motorLOff": io.motorLOff, "motorROff": io.motorROff, "motorN": io.motorN,
			"sensScale": io.sensScale, "pauseThresh": io.pauseThresh,
			"note": note,
		})
	})

	log.Printf("flyviz on http://localhost%s  (brain %s)", *listen, *brainAddr)
	log.Fatal(http.ListenAndServe(*listen, mux))
}
