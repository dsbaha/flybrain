package main

// runClient exercises the server protocol end to end and reports PASS/FAIL.
// It is the executable contract a Webots controller (or any other client)
// can rely on: handshake, SENS injection, STEP, READ/SPIK, STAT, error
// handling, QUIT.
//
// The check runs the server quiet: start the server with
// --bg-rate 0 --inject-frac 0 so background drive doesn't mask the
// injected currents.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"net"
	"strings"
)

type testClient struct {
	conn net.Conn
	br   *bufio.Reader
	n    int
	dtMs float64
}

func dialServer(addr string) (*testClient, error) {
	c, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, err
	}
	tc := &testClient{conn: c, br: bufio.NewReader(c)}
	line, err := tc.br.ReadString('\n')
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("handshake: %w", err)
	}
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "FLYBRAIN/1 ") {
		c.Close()
		return nil, fmt.Errorf("bad handshake: %q", line)
	}
	for _, f := range strings.Fields(line[11:]) {
		var v float64
		if _, err := fmt.Sscanf(f, "N=%d", &tc.n); err == nil {
			continue
		}
		if _, err := fmt.Sscanf(f, "DT=%f", &v); err == nil {
			tc.dtMs = v
		}
	}
	if tc.n <= 0 || tc.dtMs <= 0 {
		c.Close()
		return nil, fmt.Errorf("bad handshake values: %q", line)
	}
	fmt.Printf("handshake: %s\n", line)
	return tc, nil
}

func (tc *testClient) req(tag string, payload []byte) (string, []byte, error) {
	if err := writeFrame(tc.conn, tag, payload); err != nil {
		return "", nil, err
	}
	return readFrame(tc.br)
}

func (tc *testClient) sens(offset int, vals []float32) error {
	p := make([]byte, 4+len(vals)*4)
	binary.LittleEndian.PutUint32(p[0:], uint32(offset))
	putF32s(p, 4, vals)
	tag, _, err := tc.req("SENS", p)
	if err != nil {
		return err
	}
	if tag != "OKAY" {
		return fmt.Errorf("SENS: got %q, want OKAY", tag)
	}
	return nil
}

func (tc *testClient) step(n int) (totSpikes, simSteps uint64, err error) {
	p := make([]byte, 4)
	binary.LittleEndian.PutUint32(p, uint32(n))
	tag, out, err := tc.req("STEP", p)
	if err != nil {
		return 0, 0, err
	}
	if tag != "DONE" || len(out) != 16 {
		return 0, 0, fmt.Errorf("STEP: got %q len %d", tag, len(out))
	}
	return binary.LittleEndian.Uint64(out[0:]),
		binary.LittleEndian.Uint64(out[8:]), nil
}

func (tc *testClient) read(start, count int) ([]float32, error) {
	p := make([]byte, 8)
	binary.LittleEndian.PutUint32(p[0:], uint32(start))
	binary.LittleEndian.PutUint32(p[4:], uint32(count))
	tag, out, err := tc.req("READ", p)
	if err != nil {
		return nil, err
	}
	if tag != "VOLT" || len(out) != 8+count*4 {
		return nil, fmt.Errorf("READ: got %q len %d", tag, len(out))
	}
	vals := make([]float32, count)
	for i := range vals {
		vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(out[8+i*4:]))
	}
	return vals, nil
}

func (tc *testClient) spik(start, count int) ([]uint32, error) {
	p := make([]byte, 8)
	binary.LittleEndian.PutUint32(p[0:], uint32(start))
	binary.LittleEndian.PutUint32(p[4:], uint32(count))
	tag, out, err := tc.req("SPIK", p)
	if err != nil {
		return nil, err
	}
	if tag != "SPIK" || len(out) != 8+count*4 {
		return nil, fmt.Errorf("SPIK: got %q len %d", tag, len(out))
	}
	vals := make([]uint32, count)
	for i := range vals {
		vals[i] = binary.LittleEndian.Uint32(out[8+i*4:])
		if vals[i] > 1 {
			return nil, fmt.Errorf("SPIK: flag %d out of range", vals[i])
		}
	}
	return vals, nil
}

func (tc *testClient) spkc(start, count int) ([]uint64, error) {
	p := make([]byte, 8)
	binary.LittleEndian.PutUint32(p[0:], uint32(start))
	binary.LittleEndian.PutUint32(p[4:], uint32(count))
	tag, out, err := tc.req("SPKC", p)
	if err != nil {
		return nil, err
	}
	if tag != "SPKC" || len(out) != 8+count*8 {
		return nil, fmt.Errorf("SPKC: got %q len %d", tag, len(out))
	}
	vals := make([]uint64, count)
	for i := range vals {
		vals[i] = binary.LittleEndian.Uint64(out[8+i*8:])
	}
	return vals, nil
}

func sumU32(xs []uint32) uint64 {
	var s uint64
	for _, x := range xs {
		s += uint64(x)
	}
	return s
}

func mean(xs []float32) float64 {
	var s float64
	for _, x := range xs {
		s += float64(x)
	}
	return s / float64(len(xs))
}

func minF(xs []float32) float32 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func maxF(xs []float32) float32 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func runClient(addr string) error {
	tc, err := dialServer(addr)
	if err != nil {
		return err
	}
	defer tc.conn.Close()
	fail := func(format string, args ...any) error {
		return fmt.Errorf("FAIL: "+format, args...)
	}

	// 0. Baseline STAT: the server settles the network before serving, so
	// learn the starting step counter instead of assuming zero.
	tag0, out0, err := tc.req("STAT", nil)
	if err != nil || tag0 != "STAT" || len(out0) != 32 {
		return fail("initial STAT: %v %q", err, tag0)
	}
	baseSteps := binary.LittleEndian.Uint64(out0[0:])
	baseTot := binary.LittleEndian.Uint64(out0[16:])
	fmt.Printf("baseline: simSteps=%d totalSpikes=%d\n", baseSteps, baseTot)

	// 1. Inject a strong depolarizing current into neurons [0,64).
	const injN = 64
	vals := make([]float32, injN)
	for i := range vals {
		vals[i] = 25.0 // mV/step: well above threshold from rest
	}
	if err := tc.sens(0, vals); err != nil {
		return fail("SENS: %v", err)
	}
	fmt.Println("SENS [0,64) <- 25mV: OKAY")

	// 2. Step and confirm the network responds.
	tot, steps, err := tc.step(200)
	if err != nil {
		return fail("STEP: %v", err)
	}
	if steps != baseSteps+200 {
		return fail("STEP: simSteps=%d, want %d", steps, baseSteps+200)
	}
	if tot == baseTot {
		return fail("STEP: zero new spikes with 25mV injection (network silent)")
	}
	fmt.Printf("STEP 200: DONE totalSpikes=%d simSteps=%d\n", tot, steps)

	// 3. The injected current must drive spikes in the injected region, not
	// elsewhere. (Voltage alone can't show this: a 25mV drive pins neurons
	// at reset between spikes, so last-step spike flags are the signal.
	// Identical noiseless neurons also lockstep, so sample a few steps.)
	var injSpk, ctlSpk uint64
	extraSteps := 0
	for i := 0; i < 4; i++ {
		var st uint64
		if st, _, err = tc.step(1); err != nil {
			return fail("STEP 1: %v", err)
		}
		tot = st
		extraSteps++
		sp, err := tc.spik(0, 128)
		if err != nil {
			return fail("SPIK: %v", err)
		}
		injSpk, ctlSpk = sumU32(sp[:injN]), sumU32(sp[injN:])
		fmt.Printf("SPIK [0,128): inj spikes=%d  ctl spikes=%d (last step)\n", injSpk, ctlSpk)
		if injSpk > 0 && injSpk > ctlSpk {
			break
		}
	}
	if injSpk == 0 || injSpk <= ctlSpk {
		return fail("injected region not driven (inj=%d ctl=%d)", injSpk, ctlSpk)
	}

	// 4. READ returns sane voltages with the right framing.
	v, err := tc.read(0, 128)
	if err != nil {
		return fail("READ: %v", err)
	}
	for i, x := range v {
		if math.IsNaN(float64(x)) || x < -90 || x > 60 {
			return fail("READ: neuron %d voltage insane: %f", i, x)
		}
	}
	fmt.Printf("READ [0,128): voltages sane (range %.1f..%.1f mV)\n",
		minF(v), maxF(v))

	// 4. Out-of-range READ must be rejected cleanly, not crash the server.
	p := make([]byte, 8)
	binary.LittleEndian.PutUint32(p[0:], uint32(tc.n-10))
	binary.LittleEndian.PutUint32(p[4:], 64)
	tag, _, err := tc.req("READ", p)
	if err != nil {
		return fail("bad READ: %v", err)
	}
	if tag != "ERR " {
		return fail("bad READ: got %q, want ERR", tag)
	}
	fmt.Println("READ out-of-range: ERR (correct)")

	// 4b. SENV scatter-write: drive 8 scattered neurons, verify OKAY and
	// that an out-of-range index is rejected.
	sp := make([]byte, 4+8*8)
	binary.LittleEndian.PutUint32(sp[0:], 8)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(sp[4+i*8:], uint32(i*1000))
		binary.LittleEndian.PutUint32(sp[8+i*8:], math.Float32bits(25.0))
	}
	tag, _, err = tc.req("SENV", sp)
	if err != nil || tag != "OKAY" {
		return fail("SENV: %v %q", err, tag)
	}
	fmt.Println("SENV 8 scattered <- 25mV: OKAY")
	bad := make([]byte, 4+8)
	binary.LittleEndian.PutUint32(bad[0:], 1)
	binary.LittleEndian.PutUint32(bad[4:], uint32(tc.n+5))
	binary.LittleEndian.PutUint32(bad[8:], math.Float32bits(1.0))
	tag, _, err = tc.req("SENV", bad)
	if err != nil || tag != "ERR " {
		return fail("SENV out-of-range: got %q %v, want ERR", tag, err)
	}
	fmt.Println("SENV out-of-range: ERR (correct)")

	// 4c. SPKC cumulative counts: injected region must have nonzero counts
	// that never decrease across steps.
	c1, err := tc.spkc(0, 128)
	if err != nil {
		return fail("SPKC: %v", err)
	}
	var st10 uint64
	if st10, _, err = tc.step(10); err != nil {
		return fail("STEP 10: %v", err)
	}
	tot = st10
	extraSteps += 10
	c2, err := tc.spkc(0, 128)
	if err != nil {
		return fail("SPKC: %v", err)
	}
	for i := range c1 {
		if c2[i] < c1[i] {
			return fail("SPKC: count decreased at neuron %d", i)
		}
	}
	var cinj uint64
	for _, x := range c2[:injN] {
		cinj += x
	}
	if cinj == 0 {
		return fail("SPKC: injected region has zero cumulative spikes")
	}
	fmt.Printf("SPKC [0,128): cumulative, monotonic, inj region=%d spikes\n", cinj)

	// 5. STAT sanity: time base and counters consistent.
	tag, out, err := tc.req("STAT", nil)
	if err != nil {
		return fail("STAT: %v", err)
	}
	if tag != "STAT" || len(out) != 32 {
		return fail("STAT: got %q len %d", tag, len(out))
	}
	stSteps := binary.LittleEndian.Uint64(out[0:])
	stMs := math.Float64frombits(binary.LittleEndian.Uint64(out[8:]))
	stTot := binary.LittleEndian.Uint64(out[16:])
	wantSteps := baseSteps + 200 + uint64(extraSteps)
	if stSteps != wantSteps || math.Abs(stMs-float64(wantSteps)*tc.dtMs) > 1e-6 || stTot != tot {
		return fail("STAT inconsistent: steps=%d ms=%.3f tot=%d", stSteps, stMs, stTot)
	}
	fmt.Printf("STAT: steps=%d simMs=%.1f totalSpikes=%d\n", stSteps, stMs, stTot)

	// 6. Clean shutdown.
	tag, _, err = tc.req("QUIT", nil)
	if err != nil {
		return fail("QUIT: %v", err)
	}
	if tag != "BYE " {
		return fail("QUIT: got %q, want BYE", tag)
	}
	fmt.Println("QUIT: BYE")
	fmt.Println("CLIENT PASS: protocol verified end to end")
	return nil
}
