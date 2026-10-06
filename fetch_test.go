package main

// Unit test for the Feather -> .fbc conversion using a synthetic Feather
// file with the same schema as proofread_connections_783.feather.

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/apache/arrow/go/v18/arrow"
	"github.com/apache/arrow/go/v18/arrow/array"
	"github.com/apache/arrow/go/v18/arrow/ipc"
	"github.com/apache/arrow/go/v18/arrow/memory"
)

func writeTestFeather(t *testing.T, path string) {
	t.Helper()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "pre_pt_root_id", Type: arrow.PrimitiveTypes.Int64},
		{Name: "post_pt_root_id", Type: arrow.PrimitiveTypes.Int64},
		{Name: "neuropil", Type: arrow.BinaryTypes.String},
		{Name: "syn_count", Type: arrow.PrimitiveTypes.Int64},
		{Name: "gaba_avg", Type: arrow.PrimitiveTypes.Float64},
		{Name: "ach_avg", Type: arrow.PrimitiveTypes.Float64},
		{Name: "glut_avg", Type: arrow.PrimitiveTypes.Float64},
	}, nil)
	mem := memory.DefaultAllocator
	b := array.NewRecordBuilder(mem, schema)
	defer b.Release()
	// Rows: (pre, post, neuropil, syn, gaba, ach, glut)
	// Pair (A->B): two neuropil rows, excitatory (ach dominant).
	// Pair (B->A): one row, inhibitory (gaba dominant).
	// Pair (C->C): autapse, excitatory.
	type row struct {
		pre, post       int64
		np              string
		syn             int64
		gaba, ach, glut float64
	}
	rows := []row{
		{101, 202, "ME", 10, 0.1, 0.8, 0.1},
		{101, 202, "LO", 20, 0.2, 0.7, 0.1},
		{202, 101, "ME", 5, 0.7, 0.2, 0.1},
		{303, 303, "CX", 4, 0.1, 0.1, 0.8}, // glut -> inhibitory
	}
	for _, r := range rows {
		b.Field(0).(*array.Int64Builder).Append(r.pre)
		b.Field(1).(*array.Int64Builder).Append(r.post)
		b.Field(2).(*array.StringBuilder).Append(r.np)
		b.Field(3).(*array.Int64Builder).Append(r.syn)
		b.Field(4).(*array.Float64Builder).Append(r.gaba)
		b.Field(5).(*array.Float64Builder).Append(r.ach)
		b.Field(6).(*array.Float64Builder).Append(r.glut)
	}
	rec := b.NewRecord()
	defer rec.Release()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w, err := ipc.NewFileWriter(f, ipc.WithSchema(schema), ipc.WithAllocator(mem))
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(rec); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestConvertFeather(t *testing.T) {
	dir := t.TempDir()
	featherPath := filepath.Join(dir, "test.feather")
	dumpPath := filepath.Join(dir, "test.fbc")
	writeTestFeather(t, featherPath)

	if err := convertFeather(featherPath, dumpPath, 0.25, 25.0); err != nil {
		t.Fatalf("convert: %v", err)
	}
	g, rootIDs, err := loadDump(dumpPath)
	if err != nil {
		t.Fatalf("loadDump: %v", err)
	}
	if g.N != 3 {
		t.Fatalf("N=%d, want 3", g.N)
	}
	if len(g.Col) != 3 {
		t.Fatalf("E=%d, want 3", len(g.Col))
	}
	// root ID -> index: first-seen order 101->0, 202->1, 303->2
	wantRoots := []uint64{101, 202, 303}
	for i, w := range wantRoots {
		if rootIDs[i] != w {
			t.Fatalf("rootIDs[%d]=%d, want %d", i, rootIDs[i], w)
		}
	}
	// Edge 0->1: syn 30, excitatory -> +7.5 mV -> 7680 fp
	// Edge 1->0: syn 5, inhibitory (gaba) -> -1.25 mV -> -1280 fp
	// Edge 2->2: syn 4, inhibitory (glut) -> -1.0 mV -> -1024 fp
	type want struct {
		src, dst uint32
		w        int32
	}
	wants := []want{{0, 1, 7680}, {1, 0, -1280}, {2, 2, -1024}}
	got := map[[2]uint32]int32{}
	for src := 0; src < g.N; src++ {
		for k := g.RowPtr[src]; k < g.RowPtr[src+1]; k++ {
			got[[2]uint32{uint32(src), g.Col[k]}] = g.W[k]
		}
	}
	for _, wn := range wants {
		w, ok := got[[2]uint32{wn.src, wn.dst}]
		if !ok {
			t.Fatalf("missing edge %d->%d", wn.src, wn.dst)
		}
		if w != wn.w {
			t.Fatalf("edge %d->%d: w=%d, want %d", wn.src, wn.dst, w, wn.w)
		}
	}
	// Inhib flags: neuron 0 mostly excitatory out, 1 and 2 inhibitory.
	if g.Inhib[0] || !g.Inhib[1] || !g.Inhib[2] {
		t.Fatalf("Inhib flags wrong: %v", g.Inhib)
	}
}
