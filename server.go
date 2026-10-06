package main

// flybrain TCP server: exposes a live brain to remote clients (e.g. a
// Webots robot controller) over a tiny binary protocol.
//
// Wire format: every message is an 8-byte header (4-byte ASCII tag +
// uint32 little-endian payload length) followed by the payload.
// The server greets each connection with one text line:
//     FLYBRAIN/1 N=<neurons> DT=<dt-ms>\n
//
// Client -> server tags:
//   SENS  u32 offset, then f32 values (mV/step or Hz, persistent)
//         -> OKAY (empty)
//   SENV  u32 count, then count x (u32 index, f32 value): scatter sensory
//         write for non-contiguous neuron sets -> OKAY (empty)
//   STEP  u32 nsteps (1..100000) -> DONE: u64 totalSpikes, u64 simSteps
//   READ  u32 start, u32 count    -> VOLT: u32 start, u32 count, f32[count] voltages (mV)
//   SPIK  u32 start, u32 count    -> SPIK: u32 start, u32 count, u32[count] last-step 0/1 flags
//   SPKC  u32 start, u32 count    -> SPKC: u32 start, u32 count, u64[count] cumulative spike counts
//   STAT  (empty)                 -> STAT: u64 simSteps, f64 simTimeMs,
//                                          u64 totalSpikes, f64 lastStepWallSec
//   QUIT  (empty)                 -> BYE (empty), then the server closes the connection
// Any malformed request or unknown tag -> ERR with a text message payload.
// Out-of-range offsets/counts are rejected; they never touch the simulation.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net"
	"sync"
	"time"
)

const (
	maxFrameBytes = 64 << 20
	maxStepN      = 100000
)

func writeFrame(w io.Writer, tag string, payload []byte) error {
	if len(tag) != 4 {
		return fmt.Errorf("tag %q must be 4 bytes", tag)
	}
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
	if n > maxFrameBytes {
		return "", nil, fmt.Errorf("frame too large: %d bytes", n)
	}
	payload := make([]byte, n)
	if _, err := io.ReadFull(r, payload); err != nil {
		return "", nil, err
	}
	return string(hdr[:4]), payload, nil
}

func errFrame(w io.Writer, format string, args ...any) {
	_ = writeFrame(w, "ERR ", []byte(fmt.Sprintf(format, args...)))
}

func getU32(p []byte, off int) (uint32, bool) {
	if len(p) < off+4 {
		return 0, false
	}
	return binary.LittleEndian.Uint32(p[off:]), true
}

func getF32s(p []byte, off int) ([]float32, bool) {
	if (len(p)-off)%4 != 0 || len(p) < off {
		return nil, false
	}
	n := (len(p) - off) / 4
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(p[off+i*4:]))
	}
	return out, true
}

func putF32s(out []byte, off int, vals []float32) {
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[off+i*4:], math.Float32bits(v))
	}
}

// srvState holds the shared simulation plus per-connection bookkeeping.
type srvState struct {
	mu           sync.Mutex
	be           backend
	n            int
	dtMs         float64
	lastStepWall float64
}

func (st *srvState) handle(c net.Conn) {
	defer c.Close()
	br := bufio.NewReader(c)
	fmt.Fprintf(c, "FLYBRAIN/1 N=%d DT=%.3f\n", st.n, st.dtMs)
	for {
		tag, p, err := readFrame(br)
		if err != nil {
			return // client went away
		}
		st.mu.Lock()
		quit := st.dispatch(c, tag, p)
		st.mu.Unlock()
		if quit {
			return
		}
	}
}

// dispatch handles one request; caller holds st.mu. Returns true on QUIT.
func (st *srvState) dispatch(c net.Conn, tag string, p []byte) bool {
	switch tag {
	case "SENS":
		off, ok := getU32(p, 0)
		if !ok {
			errFrame(c, "SENS needs u32 offset")
			return false
		}
		vals, ok := getF32s(p, 4)
		if !ok || len(vals) == 0 {
			errFrame(c, "SENS needs offset + at least one f32")
			return false
		}
		if err := st.be.setSens(int(off), vals); err != nil {
			errFrame(c, "SENS: %v", err)
			return false
		}
		_ = writeFrame(c, "OKAY", nil)

	case "SENV":
		count, ok := getU32(p, 0)
		if !ok || len(p) != 4+int(count)*8 {
			errFrame(c, "SENV needs u32 count + count x (u32 index, f32 value)")
			return false
		}
		for i := 0; i < int(count); i++ {
			idx, _ := getU32(p, 4+i*8)
			bits, _ := getU32(p, 8+i*8)
			val := math.Float32frombits(bits)
			if err := st.be.setSens(int(idx), []float32{val}); err != nil {
				errFrame(c, "SENV: %v", err)
				return false
			}
		}
		_ = writeFrame(c, "OKAY", nil)

	case "STEP":
		n, ok := getU32(p, 0)
		if !ok || n < 1 || n > maxStepN {
			errFrame(c, "STEP needs u32 nsteps in [1, %d]", maxStepN)
			return false
		}
		t0 := time.Now()
		if err := st.be.stepN(int(n)); err != nil {
			errFrame(c, "STEP: %v", err)
			return false
		}
		st.lastStepWall = time.Since(t0).Seconds()
		tot, err := st.be.totalSpikes()
		if err != nil {
			errFrame(c, "STEP readback: %v", err)
			return false
		}
		out := make([]byte, 16)
		binary.LittleEndian.PutUint64(out[0:], tot)
		binary.LittleEndian.PutUint64(out[8:], st.be.simSteps())
		_ = writeFrame(c, "DONE", out)

	case "READ", "SPIK", "SPKC":
		start, ok1 := getU32(p, 0)
		count, ok2 := getU32(p, 4)
		if !ok1 || !ok2 || count < 1 || int(start+count) > st.n {
			errFrame(c, "%s: range [%d, %d) out of bounds for N=%d",
				tag, start, start+count, st.n)
			return false
		}
		var data []float32
		var udata []uint32
		var cdata []uint64
		var err error
		reply := "VOLT"
		switch tag {
		case "READ":
			var v []float32
			v, err = st.be.readVoltages()
			if err == nil {
				data = v[start : start+count]
			}
		case "SPIK":
			var s []uint32
			s, err = st.be.readSpikes()
			if err == nil {
				udata = s[start : start+count]
			}
			reply = "SPIK"
		case "SPKC":
			cdata, err = st.be.spikeCounts(int(start), int(count))
			reply = "SPKC"
		}
		if err != nil {
			errFrame(c, "%s readback: %v", tag, err)
			return false
		}
		elem := 4
		if tag == "SPKC" {
			elem = 8
		}
		out := make([]byte, 8+int(count)*elem)
		binary.LittleEndian.PutUint32(out[0:], start)
		binary.LittleEndian.PutUint32(out[4:], count)
		switch tag {
		case "READ":
			putF32s(out, 8, data)
		case "SPIK":
			for i, v := range udata {
				binary.LittleEndian.PutUint32(out[8+i*4:], v)
			}
		case "SPKC":
			for i, v := range cdata {
				binary.LittleEndian.PutUint64(out[8+i*8:], v)
			}
		}
		_ = writeFrame(c, reply, out)

	case "STAT":
		tot, err := st.be.totalSpikes()
		if err != nil {
			errFrame(c, "STAT readback: %v", err)
			return false
		}
		out := make([]byte, 32)
		binary.LittleEndian.PutUint64(out[0:], st.be.simSteps())
		binary.LittleEndian.PutUint64(out[8:], math.Float64bits(float64(st.be.simSteps())*st.dtMs))
		binary.LittleEndian.PutUint64(out[16:], tot)
		binary.LittleEndian.PutUint64(out[24:], math.Float64bits(st.lastStepWall))
		_ = writeFrame(c, "STAT", out)

	case "QUIT":
		_ = writeFrame(c, "BYE ", nil)
		return true

	default:
		errFrame(c, "unknown tag %q", tag)
	}
	return false
}

// runServer builds the brain (GPU preferred, CPU fallback), settles it,
// applies the experiment (if any), and serves it on addr until killed.
func runServer(cfg *Config, g *Graph, exp *experimentSpec, addr string) error {
	var be backend
	beName := "webgpu"
	if gs, err := newGPU(cfg, g); err != nil {
		fmt.Printf("WebGPU unavailable (%v); server falling back to CPU backend\n", err)
		be, beName = newCPUSim(cfg, g), "cpu"
	} else {
		be = gs
	}
	defer be.close()

	// Settle + zero counters so clients start from a clean t=0.
	if err := be.stepN(50); err != nil {
		return err
	}
	if err := be.resetStats(); err != nil {
		return err
	}
	if err := applyExperiment(be, exp); err != nil {
		return err
	}

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", addr, err)
	}
	fmt.Printf("flybrain server (%s backend): N=%d dt=%.2fms listening on %s\n",
		beName, g.N, cfg.DtMs, ln.Addr())
	st := &srvState{be: be, n: g.N, dtMs: cfg.DtMs}
	for {
		c, err := ln.Accept()
		if err != nil {
			return fmt.Errorf("accept: %w", err)
		}
		go st.handle(c)
	}
}
