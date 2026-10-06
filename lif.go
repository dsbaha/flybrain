package main

import (
	"strconv"
	"time"
)

// xorshift32 matches the WGSL rng exactly.
func xorshift32(s uint32) uint32 {
	s ^= s << 13
	s ^= s >> 17
	s ^= s << 5
	return s
}

func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	z := x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// cpuSim is the pure-Go reference LIF simulator. Math mirrors shaders.wgsl
// operation-for-operation (float32) so --selftest can bit-compare.
//
// Synapse model (Shiu et al.): spikes add their weight to a per-neuron
// conductance g that decays exponentially (tau_syn) and is delivered after
// a delay ring of D steps; g resets to 0 when the neuron spikes. The
// classic model is the delta-synapse limit (decay 0, D=1, no g-reset,
// synRate=1/dt), which is bit-identical to the old code.
type cpuSim struct {
	n        int
	dt       float32
	pBg      float32
	kickMv   float32
	pInj     float32
	kickInj  float32
	nInject  int
	dm       derivedModel

	v     []float32
	gsyn  []float32 // synaptic conductance (mV), decays with synDecay
	refr  []uint32
	rng   []uint32
	acc   []int32 // delay ring: D slots of n int32 accumulators
	spike []uint32
	total []uint64
	sens  []float32 // host sensory channel: mV/step (current) or Hz (poisson)
	steps   uint64
	stepIdx uint64 // ring slot counter

	g *Graph
}

func newCPUSim(cfg *Config, g *Graph) *cpuSim {
	dm := deriveModel(cfg)
	s := &cpuSim{
		n:        g.N,
		dt:       float32(cfg.DtMs),
		pBg:      float32(cfg.BgRate * cfg.DtMs),
		kickMv:   float32(cfg.BgKickMv),
		pInj:     float32(cfg.InjRateHz / 1000.0 * cfg.DtMs),
		kickInj:  float32(cfg.InjKickMv),
		nInject:  int(cfg.InjFrac * float64(g.N)),
		dm:       dm,
		v:        make([]float32, g.N),
		gsyn:     make([]float32, g.N),
		refr:     make([]uint32, g.N),
		rng:      make([]uint32, g.N),
		acc:      make([]int32, g.N*dm.delayD),
		spike:    make([]uint32, g.N),
		total:    make([]uint64, g.N),
		sens:     make([]float32, g.N),
		g:        g,
	}
	for i := range s.v {
		s.v[i] = dm.vRest
		s.rng[i] = uint32(splitmix64(uint64(cfg.Seed)*0x9E3779B9 + uint64(i) + 1))
	}
	return s
}

const u32maxF = float32(4294967295.0)

// step advances the network by one dt. Order mirrors the WGSL kernels:
// tick (implicit via s.step), integrate (drain slot, rng, update) then
// propagate (scatter spikes into the same slot: it is drained D steps
// later, giving the axonal delay).
func (s *cpuSim) step() {
	g := s.g
	dm := s.dm
	slot := int(s.stepIdx % uint64(dm.delayD))
	base := slot * s.n
	for i := 0; i < s.n; i++ {
		a := s.acc[base+i]
		s.acc[base+i] = 0
		gs := s.gsyn[i]*dm.synDecay + float32(a)/FPScale
		s.gsyn[i] = gs

		s.rng[i] = xorshift32(s.rng[i])
		var kick float32
		if float32(s.rng[i])/u32maxF < s.pBg {
			kick += s.kickMv
		}
		if i < s.nInject {
			s.rng[i] = xorshift32(s.rng[i])
			if float32(s.rng[i])/u32maxF < s.pInj {
				kick += s.kickInj
			}
		}
		if dm.sensPoisson {
			if s.sens[i] > 0 {
				s.rng[i] = xorshift32(s.rng[i])
				if float32(s.rng[i])/u32maxF < s.sens[i]*s.dt/1000 {
					kick += dm.poissonKick
				}
			}
		} else {
			kick += s.sens[i]
		}

		if s.refr[i] > 0 {
			s.refr[i]--
			s.spike[i] = 0
		} else {
			v := s.v[i] + kick
			v += s.dt * ((dm.vRest-v)/dm.tauM + gs*dm.synRate)
			if v >= dm.vTh {
				s.spike[i] = 1
				s.total[i]++
				s.v[i] = dm.vReset
				if dm.gReset {
					s.gsyn[i] = 0
				}
				s.refr[i] = dm.refrSteps
			} else {
				s.spike[i] = 0
				// Reversal-potential clamp (mirrors shaders.wgsl).
				if v < -95 {
					v = -95
				} else if v > 70 {
					v = 70
				}
				s.v[i] = v
			}
		}
	}
	for i := 0; i < s.n; i++ {
		if s.spike[i] == 0 {
			continue
		}
		for k := g.RowPtr[i]; k < g.RowPtr[i+1]; k++ {
			s.acc[base+int(g.Col[k])] += g.W[k]
		}
	}
	s.stepIdx++
}

func runCPU(cfg *Config, g *Graph, exp *experimentSpec) (*RunResult, error) {
	s := newCPUSim(cfg, g)
	if exp != nil {
		rate := float32(exp.rateHz)
		for _, i := range exp.indices {
			s.sens[i] = rate
		}
	}
	// Warmup (excluded from reported stats).
	for i := 0; i < 50 && i < cfg.Steps; i++ {
		s.step()
	}
	s.resetStats()
	t0 := time.Now()
	for i := 0; i < cfg.Steps; i++ {
		s.step()
		if cfg.StatsEveryMs > 0 && (i+1)%int(float64(cfg.StatsEveryMs)/cfg.DtMs) == 0 {
			printCPUStats(i+1, s)
		}
	}
	var tot uint64
	active := 0
	for _, c := range s.total {
		tot += c
		if c > 0 {
			active++
		}
	}
	return &RunResult{WallSec: time.Since(t0).Seconds(), TotalSpikes: tot, ActiveNeurons: active}, nil
}

func printCPUStats(step int, s *cpuSim) {
	var tot uint64
	for _, c := range s.total {
		tot += c
	}
	simSec := float64(step) * float64(s.dt) / 1000.0
	print("\r")
	print("\x1b[2K")
	print("step ", step, "  spikes ", tot, "  meanHz ",
		strconv.FormatFloat(float64(tot)/float64(s.n)/simSec, 'f', 2, 64))
}

// ---- backend interface (interactive/server use) ----

func (s *cpuSim) stepN(n int) error {
	for i := 0; i < n; i++ {
		s.step()
	}
	s.steps += uint64(n)
	return nil
}

func (s *cpuSim) setSens(offset int, vals []float32) error {
	if offset < 0 || len(vals) == 0 || offset+len(vals) > s.n {
		return errOutOfRange("setSens", offset, len(vals), s.n)
	}
	copy(s.sens[offset:], vals)
	return nil
}

func (s *cpuSim) readVoltages() ([]float32, error) {
	out := make([]float32, s.n)
	copy(out, s.v)
	return out, nil
}

func (s *cpuSim) readSpikes() ([]uint32, error) {
	out := make([]uint32, s.n)
	copy(out, s.spike)
	return out, nil
}

func (s *cpuSim) spikeCounts(start, count int) ([]uint64, error) {
	if start < 0 || count < 0 || start+count > s.n {
		return nil, errOutOfRange("spikeCounts", start, count, s.n)
	}
	out := make([]uint64, count)
	copy(out, s.total[start:start+count])
	return out, nil
}

func (s *cpuSim) totalSpikes() (uint64, error) {
	var tot uint64
	for _, c := range s.total {
		tot += c
	}
	return tot, nil
}

func (s *cpuSim) simSteps() uint64 { return s.steps }

func (s *cpuSim) resetStats() error {
	for i := range s.total {
		s.total[i] = 0
	}
	return nil
}

func (s *cpuSim) close() {}
