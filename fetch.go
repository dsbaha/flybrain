package main

// --fetch-connectome: download the FlyWire proofread connectivity Feather
// file from Zenodo (once), convert it to the compact .fbc binary dump, and
// exit. The dump is then loadable with --graph for benchmark or --server.
//
// Mapping (documented, deterministic):
//   - Rows are (pre_root_id, post_root_id, neuropil, syn_count, NT avgs).
//     We aggregate over neuropils: total synapse count plus synapse-weighted
//     neurotransmitter means per neuron pair.
//   - FlyWire root IDs (uint64) are remapped to dense 0-based indices in
//     first-seen order; the root_id -> index table is stored in the dump
//     (needed later to join cell-type annotations).
//   - Edge sign from transmitter predictions: inhibitory when
//     gaba_avg + glut_avg > ach_avg (fly: ACh excitatory; GABA/glutamate
//     inhibitory), else excitatory. Modulatory (oct/ser/da) ignored.
//   - Edge weight: sign * syn_count * wScale mV, capped at +/-wCap mV
//     (a point LIF has no dendritic tree; uncapped, the tail reaches
//     hundreds of mV per spike). Stored fixed-point x1024, exactly the
//     GPU's w-buffer format, so loading is a straight memcpy.
//
// .fbc layout (all little-endian):
//   magic "FLYBRAIN" (8B), version u32, N u64, E u64,
//   row_ptr (N+1 x u32), col (E x u32), w (E x i32 fixed-point),
//   root_ids (N x u64), inhib (N x u8, synapse-weighted majority per source)

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/apache/arrow/go/v18/arrow/array"
	"github.com/apache/arrow/go/v18/arrow/ipc"
)

const (
	connectomeURL   = "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"
	connectomeSize  = 852022274 // bytes, as published on Zenodo
	featherFileName = "proofread_connections_783.feather"
	dumpFileName    = "flywire783.fbc"
	dumpMagic       = "FLYBRAIN"
	dumpVersion     = 1
)

// fetchConnectome downloads (if needed) and converts the FlyWire data.
func fetchConnectome(dir string, wScale, wCap float64) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	featherPath := filepath.Join(dir, featherFileName)
	dumpPath := filepath.Join(dir, dumpFileName)
	if err := downloadFeather(featherPath); err != nil {
		return err
	}
	return convertFeather(featherPath, dumpPath, wScale, wCap)
}

func downloadFeather(dest string) error {
	if st, err := os.Stat(dest); err == nil {
		if st.Size() == connectomeSize {
			fmt.Printf("connectome already present: %s (%.0f MB)\n",
				dest, float64(st.Size())/1e6)
			return nil
		}
		fmt.Printf("resuming partial download (%d / %d bytes)\n", st.Size(), connectomeSize)
	}
	f, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	off, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	req, err := http.NewRequest("GET", connectomeURL, nil)
	if err != nil {
		return err
	}
	if off > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", off))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if off > 0 && resp.StatusCode != http.StatusPartialContent {
		// Server ignored Range; restart from scratch.
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return err
		}
		if err := f.Truncate(0); err != nil {
			return err
		}
		off = 0
	} else if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("download: HTTP %s", resp.Status)
	}
	total := connectomeSize
	done := off
	lastPrint := time.Now()
	buf := make([]byte, 1<<20)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				return werr
			}
			done += int64(n)
			if time.Since(lastPrint) > 2*time.Second {
				fmt.Printf("\r  downloading: %.1f / %.0f MB (%.0f%%)",
					float64(done)/1e6, float64(total)/1e6, 100*float64(done)/float64(total))
				lastPrint = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return fmt.Errorf("download: %w (partial file kept, rerun to resume)", rerr)
		}
	}
	fmt.Printf("\r  downloading: %.0f MB complete\n", float64(done)/1e6)
	if done != int64(total) {
		return fmt.Errorf("size mismatch: got %d, want %d", done, total)
	}
	return nil
}

// rawRow is one (pre, post, neuropil) row with dense indices.
type rawRow struct {
	src, dst  uint32
	syn       uint32
	gaba, ach float32
	glut      float32
}

func convertFeather(featherPath, dumpPath string, wScale, wCap float64) error {
	t0 := time.Now()
	f, err := os.Open(featherPath)
	if err != nil {
		return err
	}
	defer f.Close()
	rdr, err := ipc.NewFileReader(f)
	if err != nil {
		return fmt.Errorf("feather open: %w", err)
	}
	defer rdr.Close()
	colIdx := func(name string) int {
		ix := rdr.Schema().FieldIndices(name)
		if len(ix) == 0 {
			return -1
		}
		return ix[0]
	}
	iPre, iPost, iSyn := colIdx("pre_pt_root_id"), colIdx("post_pt_root_id"), colIdx("syn_count")
	iGaba, iAch, iGlut := colIdx("gaba_avg"), colIdx("ach_avg"), colIdx("glut_avg")
	if iPre < 0 || iPost < 0 || iSyn < 0 || iGaba < 0 || iAch < 0 || iGlut < 0 {
		return fmt.Errorf("unexpected feather schema: %s", rdr.Schema())
	}

	idmap := make(map[int64]uint32, 140000)
	rootIDs := make([]uint64, 0, 140000)
	var rows []rawRow
	var nRows int64
	for i := 0; i < rdr.NumRecords(); i++ {
		rec, err := rdr.Record(i)
		if err != nil {
			return err
		}
		pre := rec.Column(iPre).(*array.Int64)
		post := rec.Column(iPost).(*array.Int64)
		syn := rec.Column(iSyn).(*array.Int64)
		gaba := rec.Column(iGaba).(*array.Float64)
		ach := rec.Column(iAch).(*array.Float64)
		glut := rec.Column(iGlut).(*array.Float64)
		for j := 0; j < int(rec.NumRows()); j++ {
			s := syn.Value(j)
			if s <= 0 {
				continue
			}
			getID := func(root int64) uint32 {
				if ix, ok := idmap[root]; ok {
					return ix
				}
				ix := uint32(len(rootIDs))
				idmap[root] = ix
				rootIDs = append(rootIDs, uint64(root))
				return ix
			}
			rows = append(rows, rawRow{
				src: getID(pre.Value(j)), dst: getID(post.Value(j)),
				syn:  uint32(s),
				gaba: float32(gaba.Value(j)), ach: float32(ach.Value(j)),
				glut: float32(glut.Value(j)),
			})
		}
		nRows += rec.NumRows()
		rec.Release()
	}
	fmt.Printf("parsed %d rows -> %d unique neurons, %d (pre,post,neuropil) rows (%.0fs)\n",
		nRows, len(rootIDs), len(rows), time.Since(t0).Seconds())

	// Sort by (src, dst) so aggregation is a single linear pass.
	t1 := time.Now()
	sort.Slice(rows, func(a, b int) bool {
		if rows[a].src != rows[b].src {
			return rows[a].src < rows[b].src
		}
		return rows[a].dst < rows[b].dst
	})
	fmt.Printf("sorted (%.0fs)\n", time.Since(t1).Seconds())

	n := len(rootIDs)
	type edge struct {
		src, dst uint32
		syn      int64
		inh      bool
	}
	var edges []edge
	inhPairs := 0
	var totSyn int64
	// Per-source synapse-weighted inhibitory majority for Graph.Inhib.
	inhSyn := make([]int64, n)
	excSyn := make([]int64, n)
	flush := func(src, dst uint32, i, j int) {
		var syn int64
		var gaba, ach, glut float64
		for k := i; k < j; k++ {
			r := rows[k]
			syn += int64(r.syn)
			gaba += float64(r.gaba) * float64(r.syn)
			ach += float64(r.ach) * float64(r.syn)
			glut += float64(r.glut) * float64(r.syn)
		}
		gaba /= float64(syn)
		ach /= float64(syn)
		glut /= float64(syn)
		inh := gaba+glut > ach
		if inh {
			inhPairs++
			inhSyn[src] += syn
		} else {
			excSyn[src] += syn
		}
		totSyn += syn
		edges = append(edges, edge{src, dst, syn, inh})
	}
	t2 := time.Now()
	for i := 0; i < len(rows); {
		j := i + 1
		for j < len(rows) && rows[j].src == rows[i].src && rows[j].dst == rows[i].dst {
			j++
		}
		flush(rows[i].src, rows[i].dst, i, j)
		i = j
	}
	rows = nil // release before CSR alloc
	fmt.Printf("aggregated to %d neuron pairs, %d synapses (%.0fs)\n",
		len(edges), totSyn, time.Since(t2).Seconds())
	fmt.Printf("inhibitory pairs: %.3f  inhibitory-majority neurons: %.3f\n",
		float64(inhPairs)/float64(len(edges)), fracTrue(inhSyn, excSyn))

	// Build CSR.
	g := &Graph{N: n, Inhib: make([]bool, n)}
	for i := range g.Inhib {
		g.Inhib[i] = inhSyn[i] > excSyn[i]
	}
	deg := make([]uint32, n)
	for _, e := range edges {
		deg[e.src]++
	}
	g.RowPtr = make([]uint32, n+1)
	for i := 0; i < n; i++ {
		g.RowPtr[i+1] = g.RowPtr[i] + deg[i]
	}
	m := len(edges)
	g.Col = make([]uint32, m)
	g.W = make([]int32, m)
	next := make([]uint32, n)
	copy(next, g.RowPtr[:n])
	var wMin, wMax float64
	capped := 0
	for _, e := range edges {
		wMv := float64(e.syn) * wScale
		if e.inh {
			wMv = -wMv
		}
		// Cap per-spike kicks: a point LIF has no dendritic tree, so the
		// raw tail (hundreds of mV per spike) is unphysical. The cap only
		// touches the extreme tail; typical edges are a few mV.
		if wMv > wCap {
			wMv = wCap
			capped++
		} else if wMv < -wCap {
			wMv = -wCap
			capped++
		}
		if wMv < wMin {
			wMin = wMv
		}
		if wMv > wMax {
			wMax = wMv
		}
		k := next[e.src]
		next[e.src]++
		g.Col[k] = e.dst
		g.W[k] = int32(math.Round(wMv * FPScale))
	}
	fmt.Printf("CSR: N=%d E=%d  weight range [%.2f, %.2f] mV (%d edges capped at %.1f mV)\n",
		n, m, wMin, wMax, capped, wCap)

	if err := writeDump(dumpPath, g, rootIDs); err != nil {
		return err
	}
	fi, _ := os.Stat(dumpPath)
	fmt.Printf("wrote %s (%.0f MB) in %.0fs total\n",
		dumpPath, float64(fi.Size())/1e6, time.Since(t0).Seconds())
	return nil
}

func fracTrue(inhSyn, excSyn []int64) float64 {
	c := 0
	for i := range inhSyn {
		if inhSyn[i] > excSyn[i] {
			c++
		}
	}
	return float64(c) / float64(len(inhSyn))
}

// writeDump serializes the graph plus root-ID table to the .fbc format.
func writeDump(path string, g *Graph, rootIDs []uint64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := func(v any) error { return binaryWrite(f, v) }
	if _, err := f.WriteString(dumpMagic); err != nil {
		return err
	}
	for _, v := range []any{
		uint32(dumpVersion), uint64(g.N), uint64(len(g.Col)),
	} {
		if err := w(v); err != nil {
			return err
		}
	}
	for _, arr := range []any{g.RowPtr, g.Col, g.W, rootIDs} {
		if err := w(arr); err != nil {
			return err
		}
	}
	inh := make([]byte, g.N)
	for i, b := range g.Inhib {
		if b {
			inh[i] = 1
		}
	}
	_, err = f.Write(inh)
	return err
}

// loadDump reads a .fbc file back into a Graph plus its root-ID table.
func loadDump(path string) (*Graph, []uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	magic := make([]byte, 8)
	if _, err := io.ReadFull(f, magic); err != nil || string(magic) != dumpMagic {
		return nil, nil, fmt.Errorf("not a flybrain dump: %s", path)
	}
	var ver uint32
	var nu, eu uint64
	for _, v := range []any{&ver, &nu, &eu} {
		if err := binaryRead(f, v); err != nil {
			return nil, nil, err
		}
	}
	if ver != dumpVersion {
		return nil, nil, fmt.Errorf("dump version %d, want %d", ver, dumpVersion)
	}
	n, e := int(nu), int(eu)
	g := &Graph{N: n}
	g.RowPtr = make([]uint32, n+1)
	g.Col = make([]uint32, e)
	g.W = make([]int32, e)
	rootIDs := make([]uint64, n)
	for _, arr := range []any{&g.RowPtr, &g.Col, &g.W, &rootIDs} {
		if err := binaryRead(f, arr); err != nil {
			return nil, nil, err
		}
	}
	inh := make([]byte, n)
	if _, err := io.ReadFull(f, inh); err != nil {
		return nil, nil, err
	}
	g.Inhib = make([]bool, n)
	for i, b := range inh {
		g.Inhib[i] = b == 1
	}
	return g, rootIDs, nil
}

func binaryWrite(w io.Writer, v any) error {
	return binary.Write(w, binary.LittleEndian, v)
}

func binaryRead(r io.Reader, v any) error {
	return binary.Read(r, binary.LittleEndian, v)
}
