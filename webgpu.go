package main

// WebGPU backend via github.com/townsendmerino/wgpu — a minimal CGO
// binding over wgpu-native v29 (Vulkan). The native static library is
// linked into the binary; no runtime .so needed.

import (
	_ "embed"
	"fmt"
	"math"
	"time"
	"unsafe"

	"github.com/townsendmerino/wgpu"
)

//go:embed shaders.wgsl
var shaderWGSL string

// gpuSim holds WebGPU state for the LIF network.
type gpuSim struct {
	device   *wgpu.Device
	queue    *wgpu.Queue
	tickPipe *wgpu.ComputePipeline
	intPipe  *wgpu.ComputePipeline
	propPipe *wgpu.ComputePipeline
	tickBG   *wgpu.BindGroup
	intBG    *wgpu.BindGroup
	propBG   *wgpu.BindGroup
	staging  *wgpu.Buffer
	totalB   *wgpu.Buffer
	vB       *wgpu.Buffer
	spikeB   *wgpu.Buffer
	sensB    *wgpu.Buffer
	n        int
	steps    uint64
}

func b2b[T any](s []T, elem int) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*elem)
}

func backendName(b wgpu.BackendType) string {
	switch b {
	case wgpu.BackendTypeVulkan:
		return "Vulkan"
	case wgpu.BackendTypeMetal:
		return "Metal"
	case wgpu.BackendTypeD3D12:
		return "D3D12"
	case wgpu.BackendTypeNull:
		return "Null"
	default:
		return fmt.Sprintf("backend#%d", uint32(b))
	}
}

func newGPU(cfg *Config, g *Graph) (*gpuSim, error) {
	instance := wgpu.CreateInstance(nil)
	defer instance.Release()

	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
		BackendType:     wgpu.BackendTypeVulkan,
	})
	if err != nil {
		return nil, fmt.Errorf("no Vulkan adapter found: %w", err)
	}
	defer adapter.Release()

	info := adapter.GetInfo()
	fmt.Printf("adapter: %s | backend: %s | vendor %s\n",
		info.Name, backendName(info.BackendType), info.VendorName)

	n := g.N
	e := uint64(len(g.Col))
	dm := deriveModel(cfg)
	d64 := uint64(dm.delayD)
	need := e * 4 // col and w buffers, the big ones
	alim := adapter.GetLimits().Limits
	fmt.Printf("limits: maxStorageBufferBindingSize=%d MiB maxBufferSize=%d MiB\n",
		alim.MaxStorageBufferBindingSize>>20, alim.MaxBufferSize>>20)
	if alim.MaxStorageBufferBindingSize < need {
		return nil, fmt.Errorf("adapter maxStorageBufferBindingSize (%d) < needed (%d); reduce --synapses",
			alim.MaxStorageBufferBindingSize, need)
	}
	reqLim := alim
	if need > reqLim.MaxStorageBufferBindingSize {
		reqLim.MaxStorageBufferBindingSize = need
	}
	if need > reqLim.MaxBufferSize {
		reqLim.MaxBufferSize = need
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{Limits: reqLim},
	})
	if err != nil {
		return nil, fmt.Errorf("request device: %w", err)
	}
	queue := device.GetQueue()

	mkBuf := func(size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return device.CreateBuffer(&wgpu.BufferDescriptor{
			Usage: usage,
			Size:  size,
		})
	}
	storage := wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc

	n64 := uint64(n)
	newB := func(sz uint64, u wgpu.BufferUsage) *wgpu.Buffer {
		b, err := mkBuf(sz, u)
		if err != nil {
			panic(fmt.Sprintf("create buffer: %v", err))
		}
		return b
	}
	vB := newB(n64*4, storage)
	refrB := newB(n64*4, storage)
	rngB := newB(n64*4, storage)
	accB := newB(n64*d64*4, storage) // delay ring: D slots of n int32
	gsynB := newB(n64*4, storage)    // synaptic conductance, zero-init
	spikeB := newB(n64*4, storage)
	totalB := newB(n64*4, storage)
	uniB := newB(80, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	stepCtrB := newB(4, storage) // delay-ring slot counter, zero-init
	sensB := newB(n64*4, storage) // host sensory channel, zero-init
	rowB := newB((n64+1)*4, storage)
	colB := newB(e*4, storage)
	wB := newB(e*4, storage)
	staging, err := mkBuf(n64*4, wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	if err != nil {
		return nil, err
	}

	// Upload static data.
	v0 := make([]float32, n)
	for i := range v0 {
		v0[i] = dm.vRest
	}
	rng0 := make([]uint32, n)
	for i := range rng0 {
		rng0[i] = uint32(splitmix64(uint64(cfg.Seed)*0x9E3779B9 + uint64(i) + 1))
	}
	uni := make([]float32, 20)
	uni[0] = float32(cfg.DtMs)
	uni[1] = float32(cfg.BgRate * cfg.DtMs)
	uni[2] = float32(cfg.BgKickMv)
	uni[3] = float32(cfg.InjRateHz / 1000.0 * cfg.DtMs)
	uni[4] = float32(cfg.InjKickMv)
	uni[5] = float32(int(cfg.InjFrac * float64(n)))
	uni[6] = float32(n)
	uni[7] = dm.vRest
	uni[8] = dm.vTh
	uni[9] = dm.vReset
	uni[10] = dm.tauM
	uni[11] = float32(dm.refrSteps)
	uni[12] = dm.synDecay
	uni[13] = dm.synRate
	if dm.gReset {
		uni[14] = 1
	}
	uni[15] = float32(dm.delayD)
	if dm.sensPoisson {
		uni[16] = 1
	}
	uni[17] = dm.poissonKick

	tUp := time.Now()
	put := func(b *wgpu.Buffer, d []byte) error { return queue.WriteBuffer(b, 0, d) }
	for _, up := range []struct {
		b *wgpu.Buffer
		d []byte
	}{
		{vB, b2b(v0, 4)}, {rngB, b2b(rng0, 4)},
		{uniB, b2b(uni, 4)}, {rowB, b2b(g.RowPtr, 4)},
		{colB, b2b(g.Col, 4)}, {wB, b2b(g.W, 4)},
	} {
		if err := put(up.b, up.d); err != nil {
			return nil, err
		}
	}
	upBytes := n64*4*4 + 80 + (n64+1)*4 + e*4 + e*4
	fmt.Printf("uploaded %.1f MiB in %.1fs\n",
		float64(upBytes>>20), time.Since(tUp).Seconds())

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderWGSL},
	})
	if err != nil {
		return nil, fmt.Errorf("shader module: %w", err)
	}
	defer shader.Release()

	tickPipe, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{Module: shader, EntryPoint: "tick"},
	})
	if err != nil {
		return nil, fmt.Errorf("tick pipeline: %w", err)
	}
	intPipe, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{Module: shader, EntryPoint: "integrate"},
	})
	if err != nil {
		return nil, fmt.Errorf("integrate pipeline: %w", err)
	}
	propPipe, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{Module: shader, EntryPoint: "propagate"},
	})
	if err != nil {
		return nil, fmt.Errorf("propagate pipeline: %w", err)
	}

	mkBG := func(pipe *wgpu.ComputePipeline, bufs ...*wgpu.Buffer) (*wgpu.BindGroup, error) {
		layout := pipe.GetBindGroupLayout(0)
		if layout == nil {
			return nil, fmt.Errorf("bind group layout")
		}
		defer layout.Release()
		entries := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			entries[i] = wgpu.BindGroupEntry{
				Binding: uint32(i), Buffer: b, Offset: 0, Size: b.GetSize(),
			}
		}
		return device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Layout: layout, Entries: entries,
		})
	}
	// tick bindings: step counter
	tickBG, err := mkBG(tickPipe, stepCtrB)
	if err != nil {
		return nil, fmt.Errorf("tick bindgroup: %w", err)
	}
	// integrate bindings: uniforms, v, refr, rng, acc, gsyn, spike, spike_total, sens, step_ctr
	intBG, err := mkBG(intPipe, uniB, vB, refrB, rngB, accB, gsynB, spikeB, totalB, sensB, stepCtrB)
	if err != nil {
		return nil, fmt.Errorf("integrate bindgroup: %w", err)
	}
	// propagate bindings: uniforms, row_ptr, col, w, spike, acc, step_ctr
	propBG, err := mkBG(propPipe, uniB, rowB, colB, wB, spikeB, accB, stepCtrB)
	if err != nil {
		return nil, fmt.Errorf("propagate bindgroup: %w", err)
	}

	return &gpuSim{
		device: device, queue: queue,
		tickPipe: tickPipe, intPipe: intPipe, propPipe: propPipe,
		tickBG: tickBG, intBG: intBG, propBG: propBG,
		staging: staging, totalB: totalB, vB: vB,
		spikeB: spikeB, sensB: sensB, n: n,
	}, nil
}

// runSteps advances the simulation k steps, batching dispatches.
func (s *gpuSim) runSteps(k int) error {
	groups := uint32((s.n + WorkSize - 1) / WorkSize)
	enc, err := s.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	for i := 0; i < k; i++ {
		pass.SetPipeline(s.tickPipe)
		pass.SetBindGroup(0, s.tickBG, nil)
		pass.DispatchWorkgroups(1, 1, 1)
		pass.SetPipeline(s.intPipe)
		pass.SetBindGroup(0, s.intBG, nil)
		pass.DispatchWorkgroups(groups, 1, 1)
		pass.SetPipeline(s.propPipe)
		pass.SetBindGroup(0, s.propBG, nil)
		pass.DispatchWorkgroups(groups, 1, 1)
	}
	if err := pass.End(); err != nil {
		return err
	}
	pass.Release()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	enc.Release()
	s.queue.Submit(cmd)
	cmd.Release()
	return nil
}

// readbackU32 copies src (n uint32s) to CPU via the staging buffer.
func (s *gpuSim) readbackU32(src *wgpu.Buffer) ([]uint32, error) {
	return s.readbackU32Range(src, 0, s.n)
}

// readbackU32Range copies src[start:start+count] to CPU via the staging buffer.
func (s *gpuSim) readbackU32Range(src *wgpu.Buffer, start, count int) ([]uint32, error) {
	size := uint64(count) * 4
	off := uint64(start) * 4
	enc, err := s.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	if err := enc.CopyBufferToBuffer(src, off, s.staging, 0, size); err != nil {
		return nil, err
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	enc.Release()
	s.queue.Submit(cmd)
	cmd.Release()

	done := make(chan wgpu.BufferMapAsyncStatus, 1)
	if err := s.staging.MapAsync(wgpu.MapModeRead, 0, size,
		func(st wgpu.BufferMapAsyncStatus) { done <- st }); err != nil {
		return nil, err
	}
	for {
		select {
		case st := <-done:
			if st != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("map async status %d", uint32(st))
			}
			goto mapped
		default:
			s.device.Poll(true, nil)
		}
	}
mapped:
	raw := s.staging.GetMappedRange(0, uint(size))
	out := make([]uint32, count)
	copy(b2b(out, 4), raw)
	if err := s.staging.Unmap(); err != nil {
		return nil, err
	}
	return out, nil
}

// readbackF32 copies src (n float32s) to CPU via the staging buffer.
func (s *gpuSim) readbackF32(src *wgpu.Buffer) ([]float32, error) {
	u, err := s.readbackU32(src)
	if err != nil {
		return nil, err
	}
	out := make([]float32, s.n)
	for i, x := range u {
		out[i] = math.Float32frombits(x)
	}
	return out, nil
}

func (s *gpuSim) close() {
	if s.staging != nil {
		s.staging.Release()
	}
	if s.tickBG != nil {
		s.tickBG.Release()
	}
	if s.intBG != nil {
		s.intBG.Release()
	}
	if s.propBG != nil {
		s.propBG.Release()
	}
	if s.tickPipe != nil {
		s.tickPipe.Release()
	}
	if s.intPipe != nil {
		s.intPipe.Release()
	}
	if s.propPipe != nil {
		s.propPipe.Release()
	}
	if s.queue != nil {
		s.queue.Release()
	}
	if s.device != nil {
		s.device.Release()
	}
}

// ---- backend interface (interactive/server use) ----

// stepN advances the network n steps, chunking submissions so no single
// command encoder grows unbounded.
func (s *gpuSim) stepN(n int) error {
	for done := 0; done < n; {
		k := n - done
		if k > 10000 {
			k = 10000
		}
		if err := s.runSteps(k); err != nil {
			return err
		}
		done += k
	}
	s.steps += uint64(n)
	return nil
}

func (s *gpuSim) setSens(offset int, vals []float32) error {
	if offset < 0 || len(vals) == 0 || offset+len(vals) > s.n {
		return errOutOfRange("setSens", offset, len(vals), s.n)
	}
	return s.queue.WriteBuffer(s.sensB, uint64(offset)*4, b2b(vals, 4))
}

func (s *gpuSim) readVoltages() ([]float32, error) {
	return s.readbackF32(s.vB)
}

func (s *gpuSim) readSpikes() ([]uint32, error) {
	return s.readbackU32(s.spikeB)
}

func (s *gpuSim) spikeCounts(start, count int) ([]uint64, error) {
	if start < 0 || count < 0 || start+count > s.n {
		return nil, errOutOfRange("spikeCounts", start, count, s.n)
	}
	u, err := s.readbackU32Range(s.totalB, start, count)
	if err != nil {
		return nil, err
	}
	out := make([]uint64, count)
	for i, x := range u {
		out[i] = uint64(x)
	}
	return out, nil
}

func (s *gpuSim) totalSpikes() (uint64, error) {
	totals, err := s.readbackU32(s.totalB)
	if err != nil {
		return 0, err
	}
	var tot uint64
	for _, c := range totals {
		tot += uint64(c)
	}
	return tot, nil
}

func (s *gpuSim) simSteps() uint64 { return s.steps }

// resetStats zeroes the per-neuron cumulative spike counters on the GPU.
func (s *gpuSim) resetStats() error {
	return s.queue.WriteBuffer(s.totalB, 0, make([]byte, uint64(s.n)*4))
}
