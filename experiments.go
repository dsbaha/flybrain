package main

import "fmt"

// Named-neuron experiments replicated from the fly-brain_sim reference
// (nalin-dhiman/fly-brain_sim, Shiu et al. model). Root IDs are FlyWire
// proofread IDs; they are resolved to local indices through the .fbc
// dump's embedded root-ID table, so --experiment requires --graph.

// 21 sugar gustatory receptor neurons, stimulated at 200 Hz (Poisson)
// in the reference "sugar" experiment.
var sugarGRNRootIDs = []uint64{
	720575940624963786, 720575940630233916, 720575940637568838,
	720575940638202345, 720575940617000768, 720575940630797113,
	720575940632889389, 720575940621754367, 720575940621502051,
	720575940640649691, 720575940639332736, 720575940616885538,
	720575940639198653, 720575940639259967, 720575940617937543,
	720575940632425919, 720575940633143833, 720575940612670570,
	720575940628853239, 720575940629176663, 720575940611875570,
}

// P9 descending neurons (left/right). Driving both at 100 Hz Poisson
// evokes forward walking in the reference "p9" experiment — these are the
// biologically grounded motor readout for embodied use.
var p9RootIDs = []uint64{
	720575940627652358, // P9 left
	720575940635872101, // P9 right
}

// experimentSpec is a resolved experiment: local neuron indices plus the
// Poisson drive rate in Hz applied through the sensory channel.
type experimentSpec struct {
	name    string
	indices []int
	rateHz  float64
}

// resolveExperiment maps an experiment name to local neuron indices using
// the dump's root-ID table.
func resolveExperiment(name string, rootIDs []uint64) (*experimentSpec, error) {
	var roots []uint64
	var rate float64
	switch name {
	case "sugar":
		roots, rate = sugarGRNRootIDs, 200.0
	case "p9":
		roots, rate = p9RootIDs, 100.0
	default:
		return nil, fmt.Errorf("unknown experiment %q (want sugar|p9)", name)
	}
	idx := make(map[uint64]int, len(rootIDs))
	for i, r := range rootIDs {
		idx[r] = i
	}
	spec := &experimentSpec{name: name, rateHz: rate}
	for _, r := range roots {
		i, ok := idx[r]
		if !ok {
			return nil, fmt.Errorf("experiment %q: root ID %d not in this graph", name, r)
		}
		spec.indices = append(spec.indices, i)
	}
	return spec, nil
}

// applyExperiment drives the experiment's neurons through the backend's
// sensory channel (caller must have set sens-poisson mode in cfg).
func applyExperiment(be backend, spec *experimentSpec) error {
	if spec == nil {
		return nil
	}
	rate := float32(spec.rateHz)
	for _, i := range spec.indices {
		if err := be.setSens(i, []float32{rate}); err != nil {
			return fmt.Errorf("experiment %q: %w", spec.name, err)
		}
	}
	return nil
}
