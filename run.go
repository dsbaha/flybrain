package main

import (
	"fmt"
	"math"
	"time"
)

// runWebGPU executes the benchmark on the GPU backend.
func runWebGPU(cfg *Config, g *Graph, exp *experimentSpec) (*RunResult, error) {
	sim, err := newGPU(cfg, g)
	if err != nil {
		return nil, err
	}
	defer sim.close()

	stepsPerDisp := cfg.StepsPerDisp
	if stepsPerDisp < 1 {
		stepsPerDisp = 1
	}
	statsEvery := 0
	if cfg.StatsEveryMs > 0 {
		statsEvery = int(float64(cfg.StatsEveryMs) / cfg.DtMs)
	}

	// Warmup (excluded from reported stats).
	if err := sim.runSteps(50); err != nil {
		return nil, fmt.Errorf("warmup: %w", err)
	}
	if err := sim.resetStats(); err != nil {
		return nil, fmt.Errorf("reset stats: %w", err)
	}
	if err := applyExperiment(sim, exp); err != nil {
		return nil, err
	}

	var prevTot uint64
	t0 := time.Now()
	done := 0
	for done < cfg.Steps {
		k := stepsPerDisp
		if done+k > cfg.Steps {
			k = cfg.Steps - done
		}
		if err := sim.runSteps(k); err != nil {
			return nil, err
		}
		done += k
		if statsEvery > 0 && done%statsEvery < k {
			tot, meanHz, err := gpuProgress(sim, cfg, done, prevTot)
			if err != nil {
				return nil, err
			}
			prevTot = tot
			fmt.Printf("\r\x1b[2Kstep %d/%d  spikes %d  meanHz %.2f",
				done, cfg.Steps, tot, meanHz)
		}
	}
	fmt.Println()
	wall := time.Since(t0).Seconds()

	totals, err := sim.readbackU32(sim.totalB)
	if err != nil {
		return nil, fmt.Errorf("final readback: %w", err)
	}
	var tot uint64
	active := 0
	for _, c := range totals {
		tot += uint64(c)
		if c > 0 {
			active++
		}
	}
	return &RunResult{WallSec: wall, TotalSpikes: tot, ActiveNeurons: active}, nil
}

func gpuProgress(sim *gpuSim, cfg *Config, done int, prev uint64) (uint64, float64, error) {
	totals, err := sim.readbackU32(sim.totalB)
	if err != nil {
		return 0, 0, err
	}
	var tot uint64
	for _, c := range totals {
		tot += uint64(c)
	}
	simSec := float64(done) * cfg.DtMs / 1000.0
	return tot, float64(tot) / float64(sim.n) / simSec, nil
}

// runSelfTest bit-compares the CPU and WebGPU backends on a small network
// with identical seeds. Without a GPU adapter it falls back to a CPU
// determinism check (two identical runs must match exactly).
func runSelfTest() error {
	fmt.Println("=== flybrain selftest: CPU vs WebGPU ===")
	cfg := &Config{
		Model: "shiu",
		Neurons: 4096, Synapses: 400000, InhibFrac: 0.19,
		Steps: 400, DtMs: 1.0, BgRate: 0.05, BgKickMv: 2.5,
		InjFrac: 0, Seed: 777, SensMode: "poisson", PoissonKickMv: 68.75,
	}
	g, err := genSynthetic(cfg.Neurons, cfg.Synapses, cfg.InhibFrac, cfg.Seed)
	if err != nil {
		return err
	}
	cpuA := newCPUSim(cfg, g)
	cpuB := newCPUSim(cfg, g)
	// Exercise the Poisson sensory path on both (must stay identical).
	for _, c := range []*cpuSim{cpuA, cpuB} {
		for i := 0; i < 64; i++ {
			c.sens[i*7%cfg.Neurons] = 150.0 // Hz
		}
	}
	for i := 0; i < cfg.Steps; i++ {
		cpuA.step()
	}
	for i := 0; i < cfg.Steps; i++ {
		cpuB.step()
	}
	for i := range cpuA.total {
		if cpuA.total[i] != cpuB.total[i] {
			return fmt.Errorf("CPU non-determinism at neuron %d", i)
		}
	}
	fmt.Println("CPU determinism: OK (two identical runs match exactly)")

	sim, err := newGPU(cfg, g)
	if err != nil {
		fmt.Printf("WebGPU unavailable here (%v)\n", err)
		fmt.Println("SELFTEST PARTIAL: CPU core validated; rerun --selftest on Vulkan hardware for the GPU comparison.")
		return nil
	}
	defer sim.close()
	for i := 0; i < 64; i++ {
		if err := sim.setSens(i*7%cfg.Neurons, []float32{150.0}); err != nil {
			return err
		}
	}
	if err := sim.runSteps(cfg.Steps); err != nil {
		return err
	}
	gpuTot, err := sim.readbackU32(sim.totalB)
	if err != nil {
		return err
	}
	gpuV, err := sim.readbackF32(sim.vB)
	if err != nil {
		return err
	}
	badSpikes := 0
	for i := range cpuA.total {
		if uint64(gpuTot[i]) != cpuA.total[i] {
			badSpikes++
		}
	}
	maxVDiff := float32(0)
	for i := range cpuA.v {
		if d := float32(math.Abs(float64(gpuV[i] - cpuA.v[i]))); d > maxVDiff {
			maxVDiff = d
		}
	}
	fmt.Printf("spike-count mismatches: %d / %d neurons\n", badSpikes, cfg.Neurons)
	fmt.Printf("max |v_gpu - v_cpu|  : %g mV\n", maxVDiff)
	if badSpikes == 0 && maxVDiff < 1e-3 {
		fmt.Println("SELFTEST PASS: GPU matches CPU reference.")
		return nil
	}
	return fmt.Errorf("SELFTEST FAIL")
}
