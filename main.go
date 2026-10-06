// Package main implements flybrain: a Go + WebGPU (Vulkan) leaky
// integrate-and-fire spiking network simulator sized like the adult fruit
// fly brain connectome (FlyWire: ~139,255 neurons, ~50M synapses).
//
// Two backends:
//   - webgpu : GPU compute shaders via wgpu-native (Vulkan on Linux).
//   - cpu    : pure-Go reference with bit-identical math, used for validation.
//
// The default network is a synthetic connectome-shaped random graph with
// Dale's-law E/I split (19% inhibitory, after measured neurotransmitter
// ratios). A real edge list can be loaded with --edges (TSV: src dst w_mV).
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"time"
)

// Config holds all run parameters.
type Config struct {
	Backend      string
	Model        string  // neuron/synapse model: "shiu" (default, Shiu et al.) | "classic" (legacy -70/-50 delta synapses)
	TauSynMs     float64 // synaptic conductance decay tau (ms); 0 = instantaneous (delta) synapses
	SynDelayMs   float64 // axonal+synaptic delay (ms) before a spike affects its targets
	SensMode     string  // sensory channel meaning: "current" (mV/step) | "poisson" (Hz Poisson drive)
	PoissonKickMv float64 // voltage kick (mV) per Poisson sensory event
	Experiment   string  // named experiment: "" | "sugar" | "p9" (needs --graph; forces sens-poisson)
	Neurons      int
	Synapses     int64
	InhibFrac    float64
	Steps        int
	DtMs         float64
	StepsPerDisp int
	BgRate       float64 // background Poisson kick probability per neuron per ms
	BgKickMv     float64 // background kick size (mV)
	InjFrac      float64 // fraction of neurons receiving host "sensory" Poisson drive
	InjRateHz    float64 // sensory drive rate (Hz)
	InjKickMv    float64 // sensory kick size (mV)
	EdgesPath    string
	Seed         int64
	StatsEveryMs int
	SelfTest     bool
	ServerAddr   string
	ClientAddr   string
	Fetch        bool
	FetchDir     string
	WScale       float64
	WCap         float64
	GraphPath    string
}

// Model constants shared by both backends (must stay identical).
// The default "shiu" model follows Shiu et al. (fly-brain_sim reference):
// v_rest=v_reset=-52mV, v_th=-45mV, tau_m=20ms, refractory 2.2ms,
// exponential synapses (tau_syn=5ms, 1.8ms delay, conductance reset on
// spike). "classic" preserves the legacy -70/-50mV delta-synapse behavior.
const (
	VRest    = -70.0 // mV (classic model)
	VReset   = -70.0 // mV (classic model)
	VTh      = -50.0 // mV (classic model)
	TauM     = 20.0  // ms
	TRefMs   = 2.0   // ms absolute refractory (classic model)
	FPScale  = 1024  // fixed-point scale for synaptic weights
	WExcMv   = 0.25  // single excitatory PSP size (mV)
	WInhMv   = 1.0   // single inhibitory PSP size (mV)
	WorkSize = 256   // WGSL workgroup size

	// Shiu et al. reference model (default).
	ShiuVRest  = -52.0 // mV
	ShiuVReset = -52.0 // mV
	ShiuVTh    = -45.0 // mV
	ShiuTRefMs = 2.2   // ms absolute refractory
	ShiuTauSyn = 5.0   // ms synaptic conductance decay
	ShiuSynDly = 1.8   // ms axonal+synaptic delay
	ShiuWScale = 0.275 // mV per synapse (reference w_syn)
)

// derivedModel is the fully resolved per-run model both backends consume.
type derivedModel struct {
	vRest, vTh, vReset, tauM float32
	refrSteps                uint32
	synDecay                 float32 // exp(-dt/tauSyn); 0 for delta synapses
	synRate                  float32 // 1/tauM for conductance synapses; 1/dt for delta
	gReset                   bool    // zero synaptic conductance on spike
	delayD                   int     // delay ring depth in steps (>=1)
	sensPoisson              bool
	poissonKick              float32
}

// deriveModel resolves cfg into the concrete numbers both backends use.
func deriveModel(cfg *Config) derivedModel {
	var dm derivedModel
	if cfg.Model == "classic" {
		dm.vRest, dm.vTh, dm.vReset, dm.tauM = VRest, VTh, VReset, TauM
		dm.refrSteps = uint32(math.Round(TRefMs / cfg.DtMs))
		dm.synDecay = 0
		dm.synRate = float32(1.0 / cfg.DtMs) // delta: full PSP in one step
		dm.gReset = false
		dm.delayD = 1
	} else {
		dm.vRest, dm.vTh, dm.vReset, dm.tauM = ShiuVRest, ShiuVTh, ShiuVReset, TauM
		dm.refrSteps = uint32(math.Round(ShiuTRefMs / cfg.DtMs))
		if cfg.TauSynMs > 0 {
			dm.synDecay = float32(math.Exp(-cfg.DtMs / cfg.TauSynMs))
			dm.synRate = float32(1.0 / TauM)
		} else {
			dm.synDecay = 0
			dm.synRate = float32(1.0 / cfg.DtMs)
		}
		dm.gReset = true
		dm.delayD = int(math.Round(cfg.SynDelayMs / cfg.DtMs))
		if dm.delayD < 1 {
			dm.delayD = 1
		}
	}
	dm.sensPoisson = cfg.SensMode == "poisson"
	dm.poissonKick = float32(cfg.PoissonKickMv)
	return dm
}

func main() {
	cfg := Config{}
	flag.StringVar(&cfg.Backend, "backend", "webgpu", "backend: webgpu | cpu")
	flag.StringVar(&cfg.Model, "model", "shiu", "neuron/synapse model: shiu (Shiu et al. reference) | classic (legacy -70/-50mV delta synapses)")
	flag.Float64Var(&cfg.TauSynMs, "tau-syn-ms", ShiuTauSyn, "synaptic conductance decay tau in ms (0 = instantaneous delta synapses)")
	flag.Float64Var(&cfg.SynDelayMs, "syn-delay-ms", ShiuSynDly, "axonal+synaptic delay in ms before a spike affects its targets")
	flag.StringVar(&cfg.SensMode, "sens-mode", "current", "sensory channel meaning: current (mV/step) | poisson (Hz Poisson drive)")
	flag.Float64Var(&cfg.PoissonKickMv, "poisson-kick-mv", ShiuWScale*250, "voltage kick (mV) per Poisson sensory event (reference: w_syn * f_poi)")
	flag.StringVar(&cfg.Experiment, "experiment", "", "named experiment (needs --graph): sugar (21 GRNs @200Hz) | p9 (P9 DNs @100Hz, forward walking)")
	flag.IntVar(&cfg.Neurons, "neurons", 139255, "neuron count (FlyWire adult brain: 139255)")
	flag.Int64Var(&cfg.Synapses, "synapses", 50000000, "synapse count (FlyWire: ~50M+)")
	flag.Float64Var(&cfg.InhibFrac, "inhibitory-frac", 0.19, "fraction of inhibitory source neurons")
	flag.IntVar(&cfg.Steps, "steps", 2000, "simulation steps to run")
	flag.Float64Var(&cfg.DtMs, "dt-ms", 1.0, "timestep in ms")
	flag.IntVar(&cfg.StepsPerDisp, "steps-per-dispatch", 10, "sim steps batched per GPU submission")
	flag.Float64Var(&cfg.BgRate, "bg-rate", 0.1, "background Poisson kick probability per neuron per ms")
	flag.Float64Var(&cfg.BgKickMv, "bg-kick-mv", 8.0, "background kick size in mV")
	flag.Float64Var(&cfg.InjFrac, "inject-frac", 0.02, "fraction of neurons with sensory Poisson drive")
	flag.Float64Var(&cfg.InjRateHz, "inject-rate-hz", 40.0, "sensory drive rate in Hz")
	flag.Float64Var(&cfg.InjKickMv, "inject-kick-mv", 4.0, "sensory kick size in mV")
	flag.StringVar(&cfg.EdgesPath, "edges", "", "TSV edge list (src dst weight_mV, 0-based) overriding synthetic graph")
	flag.Int64Var(&cfg.Seed, "seed", 1234, "RNG seed")
	flag.IntVar(&cfg.StatsEveryMs, "stats-every-ms", 500, "sim-ms between progress reports (0=off)")
	flag.BoolVar(&cfg.SelfTest, "selftest", false, "CPU vs WebGPU bit-comparison on a small network, then exit")
	flag.StringVar(&cfg.ServerAddr, "server", "", "serve the brain on TCP addr (e.g. :5555) for Webots/clients, instead of benchmarking")
	flag.StringVar(&cfg.ClientAddr, "client", "", "run the protocol verification client against a server addr (e.g. localhost:5555), then exit")
	flag.BoolVar(&cfg.Fetch, "fetch-connectome", false, "download the FlyWire proofread connectivity file from Zenodo and convert it to a .fbc dump, then exit")
	flag.StringVar(&cfg.FetchDir, "fetch-dir", "flywire-data", "directory for the downloaded feather file and the .fbc dump")
	flag.Float64Var(&cfg.WScale, "w-scale", ShiuWScale, "mV per synapse when converting the connectome (sign from transmitter predictions)")
	flag.Float64Var(&cfg.WCap, "w-cap", 25.0, "max |weight| in mV per edge when converting (point-LIF sanity cap)")
	flag.StringVar(&cfg.GraphPath, "graph", "", "load a .fbc connectome dump (see --fetch-connectome) instead of the synthetic graph / --edges")
	flag.Parse()

	// Model-aware drive defaults: the Shiu reference model has no global
	// background drive (activity is stimulus-evoked and sparse); the legacy
	// classic defaults would be a seizure in its 7mV operating range.
	// Explicit flags always win.
	setFlags := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { setFlags[f.Name] = true })
	if cfg.Model == "shiu" {
		if !setFlags["bg-rate"] {
			cfg.BgRate = 0
		}
		if !setFlags["bg-kick-mv"] {
			cfg.BgKickMv = 2.5
		}
		if !setFlags["inject-frac"] {
			cfg.InjFrac = 0
		}
	}

	if cfg.SelfTest {
		if err := runSelfTest(); err != nil {
			log.Fatalf("selftest: %v", err)
		}
		return
	}

	if cfg.ClientAddr != "" {
		if err := runClient(cfg.ClientAddr); err != nil {
			log.Fatalf("client: %v", err)
		}
		return
	}

	if cfg.Fetch {
		if err := fetchConnectome(cfg.FetchDir, cfg.WScale, cfg.WCap); err != nil {
			log.Fatalf("fetch-connectome: %v", err)
		}
		return
	}

	t0 := time.Now()
	var g *Graph
	var rootIDs []uint64
	var err error
	switch {
	case cfg.GraphPath != "":
		fmt.Printf("loading connectome dump %s ...\n", cfg.GraphPath)
		g, rootIDs, err = loadDump(cfg.GraphPath)
		if err == nil {
			fmt.Printf("dump: %d neurons, %d synapses, %d root IDs mapped\n",
				g.N, len(g.Col), len(rootIDs))
		}
	case cfg.EdgesPath != "":
		fmt.Printf("loading edge list %s ...\n", cfg.EdgesPath)
		g, err = loadEdgeList(cfg.EdgesPath, cfg.Neurons)
	default:
		fmt.Printf("generating synthetic connectome: N=%d E=%d seed=%d ...\n",
			cfg.Neurons, cfg.Synapses, cfg.Seed)
		g, err = genSynthetic(cfg.Neurons, cfg.Synapses, cfg.InhibFrac, cfg.Seed)
	}
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("graph ready: %d neurons, %d synapses (%.1fs)\n",
		g.N, len(g.Col), time.Since(t0).Seconds())

	var exp *experimentSpec
	if cfg.Experiment != "" {
		if cfg.GraphPath == "" {
			log.Fatal("--experiment needs --graph (root IDs come from the .fbc dump)")
		}
		exp, err = resolveExperiment(cfg.Experiment, rootIDs)
		if err != nil {
			log.Fatal(err)
		}
		cfg.SensMode = "poisson" // experiments drive named neurons with Poisson spike trains
		fmt.Printf("experiment %q: %d neurons @ %.0f Hz Poisson\n",
			exp.name, len(exp.indices), exp.rateHz)
	}
	dm := deriveModel(&cfg)
	tRefMs := ShiuTRefMs
	if cfg.Model == "classic" {
		tRefMs = TRefMs
	}
	fmt.Printf("model %s: v_rest=%.0f v_th=%.0f tau_m=%.0fms t_ref=%.1fms tau_syn=%.1fms delay=%.1fms greset=%v sens=%s\n",
		cfg.Model, dm.vRest, dm.vTh, dm.tauM,
		tRefMs, cfg.TauSynMs, cfg.SynDelayMs, dm.gReset, cfg.SensMode)

	if cfg.ServerAddr != "" {
		if err := runServer(&cfg, g, exp, cfg.ServerAddr); err != nil {
			log.Fatal(err)
		}
		return
	}

	var res *RunResult
	switch cfg.Backend {
	case "cpu":
		res, err = runCPU(&cfg, g, exp)
	case "webgpu":
		res, err = runWebGPU(&cfg, g, exp)
	default:
		log.Fatalf("unknown backend %q", cfg.Backend)
	}
	if err != nil {
		log.Fatal(err)
	}

	simSec := float64(cfg.Steps) * cfg.DtMs / 1000.0
	fmt.Printf("\n=== result ===\n")
	fmt.Printf("steps        : %d x %.2f ms = %.2f sim-seconds\n", cfg.Steps, cfg.DtMs, simSec)
	fmt.Printf("wall time    : %.2fs (sim loop only)\n", res.WallSec)
	fmt.Printf("throughput   : %.0f steps/s  (%.2fx real-time)\n",
		float64(cfg.Steps)/res.WallSec, simSec/res.WallSec)
	fmt.Printf("total spikes : %d\n", res.TotalSpikes)
	fmt.Printf("mean rate    : %.2f Hz/neuron\n",
		float64(res.TotalSpikes)/float64(cfg.Neurons)/simSec)
	fmt.Printf("active neurons: %d / %d (%.3f%%)\n",
		res.ActiveNeurons, g.N, 100*float64(res.ActiveNeurons)/float64(g.N))
}

// RunResult summarizes a benchmark run.
type RunResult struct {
	WallSec       float64
	TotalSpikes   uint64
	ActiveNeurons int
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "flybrain: "+format+"\n", args...)
	os.Exit(1)
}
