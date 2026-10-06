// Fly brain LIF kernels (Shiu et al. model). Math mirrors lif.go
// operation-for-operation (float32) so --selftest can bit-compare.
//
// Synapse model: spikes add their fixed-point weight to a per-neuron
// conductance g that decays exponentially (syn_decay = exp(-dt/tau_syn))
// and is delivered after a delay ring of delay_d steps; g resets to 0 on
// spike when g_reset is set. The classic model is the delta-synapse limit
// (syn_decay=0, delay_d=1, g_reset=0, syn_rate=1/dt).
//
// Delay ring: step s drains slot (s mod D) and propagate writes the same
// slot; it is next drained at step s+D. No step counter is needed in the
// shaders except to pick the slot, which the tick kernel maintains.

struct Uniforms {
    dt: f32,
    p_bg: f32,
    kick_mv: f32,
    p_inj: f32,
    kick_inj: f32,
    n_inject: f32,
    n: f32,
    v_rest: f32,
    v_th: f32,
    v_reset: f32,
    tau_m: f32,
    refr_steps: f32,
    syn_decay: f32,
    syn_rate: f32,
    g_reset: f32,
    delay_d: f32,
    sens_poisson: f32,
    poisson_kick: f32,
    _pad0: f32,
    _pad1: f32,
};

fn xorshift(s: u32) -> u32 {
    var x = s;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

// ---------- tick: advance the delay-ring slot counter (1 thread) ----------
// The counter is atomic: across many dispatches batched in one compute
// pass, plain loads are not guaranteed to observe prior dispatch writes
// on all drivers.

@group(0) @binding(0) var<storage, read_write> tick_ctr: array<atomic<u32>>;

@compute @workgroup_size(1)
fn tick(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) {
        atomicAdd(&tick_ctr[0], 1u);
    }
}

// ---------- integrate ----------

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> refr: array<u32>;
@group(0) @binding(3) var<storage, read_write> rng: array<u32>;
@group(0) @binding(4) var<storage, read_write> acc: array<atomic<i32>>; // delay_d rings of n
@group(0) @binding(5) var<storage, read_write> gsyn: array<f32>; // synaptic conductance (mV)
@group(0) @binding(6) var<storage, read_write> spike: array<u32>;
@group(0) @binding(7) var<storage, read_write> spike_total: array<u32>;
@group(0) @binding(8) var<storage, read_write> sens: array<f32>; // host sensory channel
@group(0) @binding(9) var<storage, read_write> step_ctr: array<atomic<u32>>;

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (f32(i) >= u.n) { return; }
    let nn = u32(u.n);
    let d = u32(u.delay_d);
    let base = (atomicLoad(&step_ctr[0]) % d) * nn;

    // Drain this step's delay slot; decay the synaptic conductance.
    let a = atomicExchange(&acc[base + i], 0);
    let gs = gsyn[i] * u.syn_decay + f32(a) / 1024.0;
    gsyn[i] = gs;

    // Background Poisson kick.
    let r1 = xorshift(rng[i]);
    rng[i] = r1;
    var kick = 0.0;
    if (f32(r1) / 4294967295.0 < u.p_bg) {
        kick += u.kick_mv;
    }
    // Sensory ("host injection") Poisson drive on first n_inject neurons.
    if (f32(i) < u.n_inject) {
        let r2 = xorshift(rng[i]);
        rng[i] = r2;
        if (f32(r2) / 4294967295.0 < u.p_inj) {
            kick += u.kick_inj;
        }
    }
    // Host sensory channel: persistent current, or Poisson rate (Hz).
    if (u.sens_poisson > 0.5) {
        if (sens[i] > 0.0) {
            let r3 = xorshift(rng[i]);
            rng[i] = r3;
            if (f32(r3) / 4294967295.0 < sens[i] * u.dt / 1000.0) {
                kick += u.poisson_kick;
            }
        }
    } else {
        kick += sens[i];
    }

    if (refr[i] > 0u) {
        refr[i] -= 1u;
        spike[i] = 0u;
    } else {
        var vv = v[i] + kick;
        vv += u.dt * ((u.v_rest - vv) / u.tau_m + gs * u.syn_rate);
        if (vv >= u.v_th) {
            spike[i] = 1u;
            spike_total[i] += 1u;
            v[i] = u.v_reset;
            if (u.g_reset > 0.5) {
                gsyn[i] = 0.0;
            }
            refr[i] = u32(u.refr_steps);
        } else {
            spike[i] = 0u;
            // Reversal-potential clamp: real membranes cannot leave
            // roughly [-95, +70] mV no matter how strong the input.
            v[i] = clamp(vv, -95.0, 70.0);
        }
    }
}

// ---------- propagate ----------

@group(0) @binding(0) var<uniform> pu: Uniforms;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col: array<u32>;
@group(0) @binding(3) var<storage, read> w: array<i32>;
@group(0) @binding(4) var<storage, read> p_spike: array<u32>;
@group(0) @binding(5) var<storage, read_write> p_acc: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read> p_step_ctr: array<atomic<u32>>;

@compute @workgroup_size(256)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&p_spike)) { return; }
    if (p_spike[i] == 0u) { return; }
    // Same slot integrate just drained: it is next read delay_d steps out.
    let base = (atomicLoad(&p_step_ctr[0]) % u32(pu.delay_d)) * u32(pu.n);
    let start = row_ptr[i];
    let end = row_ptr[i + 1u];
    for (var k = start; k < end; k++) {
        atomicAdd(&p_acc[base + col[k]], w[k]);
    }
}
