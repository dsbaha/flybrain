package main

import "fmt"

// backend is the interactive simulation interface shared by the CPU and
// WebGPU simulators. The TCP server (server.go) drives the brain through
// this interface; the batch benchmark paths use the concrete types directly.
type backend interface {
	// stepN advances the network n steps (n >= 1).
	stepN(n int) error
	// setSens writes host sensory currents (mV/step, persistent until
	// overwritten) into neurons [offset, offset+len(vals)).
	setSens(offset int, vals []float32) error
	// readVoltages returns a copy of current membrane voltages (mV).
	readVoltages() ([]float32, error)
	// readSpikes returns the 0/1 spike flags of the most recent step.
	readSpikes() ([]uint32, error)
	// spikeCounts returns cumulative per-neuron spike counts for
	// [start, start+count) since the last resetStats.
	spikeCounts(start, count int) ([]uint64, error)
	// totalSpikes returns cumulative spikes since the last resetStats.
	totalSpikes() (uint64, error)
	// simSteps returns steps advanced since construction (incl. warmup).
	simSteps() uint64
	// resetStats zeroes cumulative spike counters (not voltages or time).
	resetStats() error
	close()
}

// errOutOfRange formats a bounds error for interactive accessors.
func errOutOfRange(op string, offset, count, n int) error {
	return fmt.Errorf("%s: range [%d, %d) out of bounds for N=%d",
		op, offset, offset+count, n)
}
