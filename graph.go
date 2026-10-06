package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// Graph is a directed weighted network in CSR layout with Dale's-law
// source-neuron E/I identity. Weights are int32 fixed-point (FPScale).
type Graph struct {
	N      int
	RowPtr []uint32 // N+1
	Col    []uint32
	W      []int32
	Inhib  []bool // per source neuron
}

// genSynthetic builds a connectome-shaped random graph: every neuron gets
// ~E/N distinct pseudo-random targets via a per-neuron modular stride
// (cheap, deterministic, no rejection sampling). 19% of source neurons are
// inhibitory (Dale's law), matching measured neurotransmitter ratios.
func genSynthetic(n int, e int64, inhibFrac float64, seed int64) (*Graph, error) {
	if n <= 0 || e <= 0 {
		return nil, fmt.Errorf("need positive N and E")
	}
	if e > int64(n)*int64(n-1) {
		return nil, fmt.Errorf("too many synapses for %d neurons", n)
	}
	rng := rand.New(rand.NewSource(seed))
	g := &Graph{
		N:      n,
		RowPtr: make([]uint32, n+1),
		Inhib:  make([]bool, n),
	}
	// E/I identity per source neuron.
	nInh := int(float64(n) * inhibFrac)
	perm := rng.Perm(n)
	for _, i := range perm[:nInh] {
		g.Inhib[i] = true
	}
	wExc := int32(WExcMv * FPScale)
	wInh := int32(-WInhMv * FPScale)

	base := e / int64(n)
	rem := e % int64(n)
	// First pass: row sizes.
	degs := make([]int, n)
	var total int64
	for i := 0; i < n; i++ {
		d := int(base)
		if int64(i) < rem {
			d++
		}
		if d >= n { // cannot have more distinct targets than N-1
			d = n - 1
		}
		degs[i] = d
		total += int64(d)
	}
	g.Col = make([]uint32, total)
	g.W = make([]int32, total)
	// Second pass: fill with strided pseudo-random targets.
	var off uint32
	for i := 0; i < n; i++ {
		g.RowPtr[i] = off
		d := degs[i]
		// Random odd stride, avoiding factors shared with N so the
		// first d multiples stay distinct.
		var stride int
		for {
			stride = 1 + 2*rng.Intn(n/2)
			if stride%5 != 0 && int64(stride)%27851 != 0 {
				break
			}
		}
		baseOff := rng.Intn(n)
		w := wExc
		if g.Inhib[i] {
			w = wInh
		}
		for k := 0; k < d; k++ {
			t := (baseOff + k*stride) % n
			if t == i { // never self-connect; nudge by one (keeps distinctness for d << n)
				t = (t + 1) % n
			}
			g.Col[off] = uint32(t)
			g.W[off] = w
			off++
		}
	}
	g.RowPtr[n] = off
	return g, nil
}

// loadEdgeList reads a TSV edge list "src dst weight_mV" (0-based indices)
// and builds the CSR graph. Source-neuron E/I identity is derived from the
// sign of its outgoing weights (majority vote).
func loadEdgeList(path string, n int) (*Graph, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	type edge struct {
		s, d int
		w    float64
	}
	var edges []edge
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		t := strings.TrimSpace(sc.Text())
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		flds := strings.Fields(t)
		if len(flds) < 3 {
			return nil, fmt.Errorf("line %d: need 3 fields", line)
		}
		s, err1 := strconv.Atoi(flds[0])
		d, err2 := strconv.Atoi(flds[1])
		w, err3 := strconv.ParseFloat(flds[2], 64)
		if err1 != nil || err2 != nil || err3 != nil {
			return nil, fmt.Errorf("line %d: bad fields", line)
		}
		if s < 0 || s >= n || d < 0 || d >= n || s == d {
			return nil, fmt.Errorf("line %d: index out of range", line)
		}
		edges = append(edges, edge{s, d, w})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	g := &Graph{N: n, RowPtr: make([]uint32, n+1), Inhib: make([]bool, n)}
	for _, ed := range edges {
		g.RowPtr[ed.s+1]++
	}
	var neg, pos []int64
	neg = make([]int64, n)
	pos = make([]int64, n)
	for _, ed := range edges {
		if ed.w < 0 {
			neg[ed.s]++
		} else {
			pos[ed.s]++
		}
	}
	for i := 0; i < n; i++ {
		g.Inhib[i] = neg[i] > pos[i]
	}
	for i := 0; i < n; i++ {
		g.RowPtr[i+1] += g.RowPtr[i]
	}
	total := g.RowPtr[n]
	g.Col = make([]uint32, total)
	g.W = make([]int32, total)
	cursor := append([]uint32(nil), g.RowPtr[:n]...)
	for _, ed := range edges {
		k := cursor[ed.s]
		cursor[ed.s]++
		g.Col[k] = uint32(ed.d)
		g.W[k] = int32(ed.w * FPScale)
	}
	return g, nil
}
