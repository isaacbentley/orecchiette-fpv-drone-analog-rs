// ddc_decimate.wgsl
//
// The batched sliding-DDC sweep's mixer + 63-tap Blackman-sinc FIR +
// decimate, across every probe AND every output sample in one dispatch
// (`n_probes * out_len` threads) — this is the per-input-sample mixer
// pass that dominates `detect_from_iq`'s cost on the CPU (see
// `detector.rs`'s `ddc_and_decimate`, run once per probe, sequentially,
// over the full un-decimated input).
//
// Each thread reconstructs its `num_taps`-sample mixed window by
// stepping the anchor phasor in `phase_table` (built on the CPU host in
// f64 — see `gpu.rs`'s module doc comment for why: computing the mixer
// phase from a raw f32 `phase_adv * i0` product, or accumulating it via
// a long GPU-side recursion, both lose enough precision over a ~1e6-
// sample buffer to measurably corrupt the output) BACKWARDS a few taps
// at a time. Because this walk is at most `num_taps` (63) steps of
// bounded magnitude, it carries negligible additional error on top of
// the host-computed anchor.
//
// Tap convention: `output[out_idx] = sum_{k=0}^{num_taps-1} taps[k] *
// mixed[i0 - k]`, `i0 = out_idx * decimation_factor` — i.e. `taps[k]` (in
// natural, non-reversed order) multiplies the sample `k` steps before the
// anchor. This matches `StreamingDDC::process_into_decimated`'s doubled-
// delay-line convolution exactly (derived from its `taps_for_conv`
// pre-reversal — see that function's doc comment).

struct Config {
    sample_rate: f32,
    decimation_factor: u32,
    num_taps: u32,
    total_iq_len: u32,
    out_len: u32,
    n_probes: u32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> taps: array<f32>;
@group(0) @binding(2) var<storage, read> offsets_hz: array<f32>;
@group(0) @binding(3) var<storage, read> phase_table: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> input_iq: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> output_iq: array<vec2<f32>>;

const TWO_PI: f32 = 6.283185307179586;
// Upper bound on the tap loop's trip count. The crate's wideband sweep
// always uses `StreamingDDC::new`'s DEFAULT_FIR_TAPS (63); this headroom
// is a guard, not a tuned value — the host clamps its `num_taps` to the
// mirrored `SHADER_MAX_TAPS` before dispatch, and the `min` below bounds
// the loop even if the two ever disagree.
const MAX_TAPS: u32 = 96u;

// ── A note on workgroup-memory tiling ───────────────────────────────
//
// A tiled variant (each workgroup cooperatively mixing its shared input
// span into workgroup memory once, instead of every thread re-reading
// and re-mixing ~num_taps/D overlapping samples from global memory) was
// implemented and measured at 213 ms/sweep vs 14.7 ms/sweep for THIS
// kernel on an Apple-silicon GPU (1 M samples x 11 probes, D = 5) — a
// 14x REGRESSION, despite passing the CPU-equivalence tests. On a UMA
// TBDR GPU the "redundant" overlapping reads are served from cache
// nearly free, while the tile path pays a workgroupBarrier plus long
// sequential per-thread phasor walks before any useful work. Don't
// re-attempt tiling here without profiling on the actual target GPU.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_idx = global_id.x;
    let probe = global_idx / config.out_len;
    let out_idx = global_idx % config.out_len;
    if (probe >= config.n_probes) {
        return;
    }

    let offset_hz = offsets_hz[probe];
    let phase_adv = -TWO_PI * offset_hz / config.sample_rate;
    let step_re = cos(phase_adv);
    let step_im = sin(phase_adv);

    let i0 = i32(out_idx * config.decimation_factor);
    let anchor = phase_table[probe * config.out_len + out_idx];
    var cur_re = anchor.x;
    var cur_im = anchor.y;

    var sum_re = 0.0;
    var sum_im = 0.0;

    let n_taps = min(config.num_taps, MAX_TAPS);
    for (var k = 0u; k < n_taps; k++) {
        let j = i0 - i32(k);
        if (j >= 0 && u32(j) < config.total_iq_len) {
            let s = input_iq[u32(j)];
            let mixed_re = s.x * cur_re - s.y * cur_im;
            let mixed_im = s.x * cur_im + s.y * cur_re;
            let tap = taps[k];
            sum_re += mixed_re * tap;
            sum_im += mixed_im * tap;
        }
        // Step the phasor BACKWARDS by one sample (multiply by the
        // conjugate of the forward per-sample step — valid since the
        // phasor stays unit-magnitude) to reach tap k+1's sample.
        let new_re = cur_re * step_re + cur_im * step_im;
        let new_im = cur_im * step_re - cur_re * step_im;
        cur_re = new_re;
        cur_im = new_im;
        let mag_sq = cur_re * cur_re + cur_im * cur_im;
        let inv = 0.5 * (3.0 - mag_sq);
        cur_re *= inv;
        cur_im *= inv;
    }

    output_iq[probe * config.out_len + out_idx] = vec2<f32>(sum_re, sum_im);
}
