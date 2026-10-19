"""
FRET Simulation - ELASTIC PENDULUM + KAPPA DYNAMICS (ULTRA-OPTIMIZED)
======================================================================

All physics identical to the EPK speed-optimized version.
Additional speed improvements on top of the previous optimizations:

  1. BATCHED RNG     - Pre-generate random numbers in blocks of 200K.
                       Eliminates ~20 individual np.random.randn() calls
                       per step, replacing them with array indexing.
                       Expected speedup: 1.5–3×.

  2. PARALLEL TRAJ.  - Run N_WORKERS independent trajectories via Numba
                       prange, each targeting TARGET_BURSTS/N_WORKERS bursts.
                       Results concatenated post-hoc.
                       Expected speedup: ~linear with CPU cores.

  3. INTEGER MOTOR   - Replace float time comparison for motor updates with
                       an integer countdown (motor_dt/cargo_dt is a fixed
                       ratio). Avoids float64 subtract+compare every step.

  4. EARLY REJECTION - Move the 97.5% no-detection check BEFORE calling
                       simulate_photon_fast, avoiding function-call overhead
                       for the vast majority of steps.

  5. INLINED HELPERS - Volume check, intensity, reflect, wrap_2pi are
                       manually inlined in the hot loop to guarantee no
                       call overhead.

Outputs:
  - photon_events_nanosecond_epk.csv
  - photon_bursts_nanosecond_epk.csv
"""

import numpy as np
import pandas as pd
from numba import jit, prange, config
import time
import os

# Use all available cores
N_WORKERS = max(1, os.cpu_count())

# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

kT = 4.1e-12  # pN·µm (thermal energy at room temperature)

DONOR_RADIUS    = 0.5   # nm
ACCEPTOR_RADIUS = 0.5   # nm
VISCOSITY       = 8.9e-12  # kg/(s·nm)

DONOR_FRICTION    = 6 * np.pi * DONOR_RADIUS    * VISCOSITY
ACCEPTOR_FRICTION = 6 * np.pi * ACCEPTOR_RADIUS * VISCOSITY

DONOR_DIFFUSION    = kT / DONOR_FRICTION    # nm²/s
ACCEPTOR_DIFFUSION = kT / ACCEPTOR_FRICTION # nm²/s

MOTOR_DIFFUSION = 1e8  # nm²/s
MOTOR_DIFFUSION_INSIDE = 10000

DONOR_DIPOLE_DIFF    = kT / (8 * np.pi * VISCOSITY * DONOR_RADIUS**3)
ACCEPTOR_DIPOLE_DIFF = kT / (8 * np.pi * VISCOSITY * ACCEPTOR_RADIUS**3)

DONOR_NUM_LINKS    = 3
ACCEPTOR_NUM_LINKS = 3
DONOR_SPRING_CONSTANT    = (1010 / DONOR_NUM_LINKS)    * 1e-9
ACCEPTOR_SPRING_CONSTANT = (1010 / ACCEPTOR_NUM_LINKS) * 1e-9

DONOR_K_HAT    = DONOR_SPRING_CONSTANT    / DONOR_FRICTION
ACCEPTOR_K_HAT = ACCEPTOR_SPRING_CONSTANT / ACCEPTOR_FRICTION

DONOR_RADIUS_EQ    = 0.1255 * DONOR_NUM_LINKS
ACCEPTOR_RADIUS_EQ = 0.1255 * ACCEPTOR_NUM_LINKS

WD_THETA = 20.0;  WA_THETA = 20.0
WD_PHI   = 200.0; WA_PHI   = 200.0

DONOR_THETA_DIFF    = WD_THETA * DONOR_DIFFUSION
ACCEPTOR_THETA_DIFF = WA_THETA * ACCEPTOR_DIFFUSION
DONOR_PHI_DIFF      = WD_PHI   * DONOR_DIFFUSION
ACCEPTOR_PHI_DIFF   = WA_PHI   * ACCEPTOR_DIFFUSION

CARGO_SPRING_CONSTANT = (1010 / 12) * 1e-9
CARGO_K_HAT           = CARGO_SPRING_CONSTANT / DONOR_FRICTION

DONOR_FLUORESCENCE_RATE    = (3/2) / (4e-5)
ACCEPTOR_FLUORESCENCE_RATE = (3/2) / (4e-5)

FORSTER_RADIUS = 5.0
I_EQ           = 0.7 * FORSTER_RADIUS

PROB_DONOR_NONRADIATIVE    = 0.15
PROB_ACCEPTOR_NONRADIATIVE = 0.21
PROB_ACCEPTOR_NOT_DETECTED = 0.25
PROB_DONOR_NOT_DETECTED    = 0.15
PROB_NO_DETECTION          = 0.975

# =============================================================================
# ADAPTIVE TIME STEP PARAMETERS
# =============================================================================

MOTOR_TIME_STEP_INSIDE  = 10e-6
CARGO_TIME_STEP_INSIDE  = 100e-9

MOTOR_TIME_STEP_OUTSIDE = 10e-4
CARGO_TIME_STEP_OUTSIDE = 10e-6

# =============================================================================
# BURST TARGET
# =============================================================================

TARGET_BURSTS = 10000
BURST_MIN_GAP = 1e-5

# =============================================================================
# CONFOCAL MICROSCOPY PARAMETERS
# =============================================================================

ELLIPSOID_SEMI_AXIS_X = 250
ELLIPSOID_SEMI_AXIS_Y = 250
ELLIPSOID_SEMI_AXIS_Z = 1000

REFRACTIVE_INDEX    = 1.3
NUMERICAL_APERTURE  = 1.2
WAVE_VECTOR         = 2.33 * REFRACTIVE_INDEX / NUMERICAL_APERTURE
BEAM_WAIST          = (1 / (np.pi**(3/2) * WAVE_VECTOR))**(1/3)
DETECTION_THRESHOLD = 1 / np.exp(2)

# =============================================================================
# MOTOR CONFINEMENT
# =============================================================================

CONFINEMENT_FACTOR  = 4
MOTOR_CONFINEMENT_X = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_X
MOTOR_CONFINEMENT_Y = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Y
MOTOR_CONFINEMENT_Z = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Z

# =============================================================================
# MOTOR COUNTDOWN RATIOS (OPT 3: integer motor countdown)
# =============================================================================

MOTOR_RATIO_INSIDE  = int(round(MOTOR_TIME_STEP_INSIDE  / CARGO_TIME_STEP_INSIDE))
MOTOR_RATIO_OUTSIDE = int(round(MOTOR_TIME_STEP_OUTSIDE / CARGO_TIME_STEP_OUTSIDE))

# =============================================================================
# DISPLAY SETUP
# =============================================================================

print("\n" + "="*70)
print("FRET - ELASTIC PENDULUM + KAPPA (ULTRA-OPTIMIZED)")
print("="*70)
print(f"Target bursts:          {TARGET_BURSTS:,}")
print(f"Motor diffusion:        {MOTOR_DIFFUSION:.2e} nm²/s")
print(f"Confinement box:  ±{MOTOR_CONFINEMENT_X:.0f} × ±{MOTOR_CONFINEMENT_Y:.0f} × ±{MOTOR_CONFINEMENT_Z:.0f} nm")
print(f"  Time step          INSIDE            OUTSIDE")
print(f"  Motor              {MOTOR_TIME_STEP_INSIDE*1e6:6.1f} µs         {MOTOR_TIME_STEP_OUTSIDE*1e3:6.1f} ms")
print(f"  Cargo              {CARGO_TIME_STEP_INSIDE*1e9:6.1f} ns         {CARGO_TIME_STEP_OUTSIDE*1e6:6.1f} µs")
print(f"Förster radius:         {FORSTER_RADIUS} nm   i_eq: {I_EQ:.2f} nm")
print(f"Linker eq. length:      {DONOR_RADIUS_EQ:.4f} nm ({DONOR_NUM_LINKS} links)")
print(f"Dipole D_rot:           {DONOR_DIPOLE_DIFF:.4e} rad²/s  (Cartesian OU)")
print(f"Motor countdown ratio:  IN={MOTOR_RATIO_INSIDE}  OUT={MOTOR_RATIO_OUTSIDE}")
print(f"Parallel workers:       {N_WORKERS}")
print()
print("Speed optimizations active (inherited):")
print("  + Linker sin/cos cached (4 trig/dye instead of 6)")
print("  + Dipole Cartesian OU + normalize (replaces 6 trig/dye)")
print("  + Pre-computed theta drift coefficient and centrifugal dt")
print("  + Pre-computed dipole noise amplitudes for IN/OUT regimes")
print("NEW optimizations:")
print("  + Batched RNG: blocks of 200K randn() per refill")
print("  + Parallel trajectories via Numba prange")
print("  + Integer motor countdown (no float time compare)")
print("  + Early 97.5% no-detection rejection before photon function")
print("  + Inlined volume/intensity/reflect/wrap helpers")
print("="*70 + "\n")


# =============================================================================
# PRE-COMPUTE CONSTANTS (FLOAT32)
# =============================================================================

TWO_PI_F32 = np.float32(2 * np.pi)

SQRT_2_DONOR_DIFF    = np.float32(np.sqrt(2 * DONOR_DIFFUSION))
SQRT_2_ACCEPTOR_DIFF = np.float32(np.sqrt(2 * ACCEPTOR_DIFFUSION))

SQRT_2_DONOR_THETA_DIFF    = np.float32(np.sqrt(2 * DONOR_THETA_DIFF))
SQRT_2_ACCEPTOR_THETA_DIFF = np.float32(np.sqrt(2 * ACCEPTOR_THETA_DIFF))

THETA_DRIFT_K_D = np.float32(0.5 * DONOR_THETA_DIFF)
THETA_DRIFT_K_A = np.float32(0.5 * ACCEPTOR_THETA_DIFF)

DONOR_PHI_DIFF_F32    = np.float32(DONOR_PHI_DIFF)
ACCEPTOR_PHI_DIFF_F32 = np.float32(ACCEPTOR_PHI_DIFF)

# INSIDE
CARGO_DT_IN_F32         = np.float32(CARGO_TIME_STEP_INSIDE)
MOTOR_DT_IN_F32         = np.float32(MOTOR_TIME_STEP_INSIDE)
SQRT_CARGO_DT_IN        = np.float32(np.sqrt(CARGO_TIME_STEP_INSIDE))
SQRT_2_MOTOR_DIFF_IN    = np.float32(np.sqrt(2 * MOTOR_DIFFUSION_INSIDE * MOTOR_TIME_STEP_INSIDE))
CARGO_SPRING_IN         = np.float32(CARGO_K_HAT         * CARGO_TIME_STEP_INSIDE)
CARGO_SPRING_HALF_IN    = np.float32((CARGO_K_HAT / 2)   * CARGO_TIME_STEP_INSIDE)
DONOR_RADIAL_TERM_IN    = np.float32(DONOR_K_HAT    * CARGO_TIME_STEP_INSIDE)
ACCEPTOR_RADIAL_TERM_IN = np.float32(ACCEPTOR_K_HAT * CARGO_TIME_STEP_INSIDE)
TWO_DT_IN               = np.float32(2.0 * CARGO_TIME_STEP_INSIDE)
SQRT_2_DIP_D_DT_IN      = np.float32(np.sqrt(2 * DONOR_DIPOLE_DIFF    * CARGO_TIME_STEP_INSIDE))
SQRT_2_DIP_A_DT_IN      = np.float32(np.sqrt(2 * ACCEPTOR_DIPOLE_DIFF * CARGO_TIME_STEP_INSIDE))

# OUTSIDE
CARGO_DT_OUT_F32         = np.float32(CARGO_TIME_STEP_OUTSIDE)
MOTOR_DT_OUT_F32         = np.float32(MOTOR_TIME_STEP_OUTSIDE)
SQRT_CARGO_DT_OUT        = np.float32(np.sqrt(CARGO_TIME_STEP_OUTSIDE))
SQRT_2_MOTOR_DIFF_OUT    = np.float32(np.sqrt(2 * MOTOR_DIFFUSION * MOTOR_TIME_STEP_OUTSIDE))
CARGO_SPRING_OUT         = np.float32(CARGO_K_HAT         * CARGO_TIME_STEP_OUTSIDE)
CARGO_SPRING_HALF_OUT    = np.float32((CARGO_K_HAT / 2)   * CARGO_TIME_STEP_OUTSIDE)
DONOR_RADIAL_TERM_OUT    = np.float32(DONOR_K_HAT    * CARGO_TIME_STEP_OUTSIDE)
ACCEPTOR_RADIAL_TERM_OUT = np.float32(ACCEPTOR_K_HAT * CARGO_TIME_STEP_OUTSIDE)
TWO_DT_OUT               = np.float32(2.0 * CARGO_TIME_STEP_OUTSIDE)
SQRT_2_DIP_D_DT_OUT      = np.float32(np.sqrt(2 * DONOR_DIPOLE_DIFF    * CARGO_TIME_STEP_OUTSIDE))
SQRT_2_DIP_A_DT_OUT      = np.float32(np.sqrt(2 * ACCEPTOR_DIPOLE_DIFF * CARGO_TIME_STEP_OUTSIDE))

# Ellipsoid / intensity
ELLIPSOID_A_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_X**2)
ELLIPSOID_B_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_Y**2)
ELLIPSOID_C_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_Z**2)
ELLIPSOID_C_K_SQ        = np.float32((ELLIPSOID_SEMI_AXIS_Z * WAVE_VECTOR)**2)
MINUS_2_OVER_WAIST_SQ   = np.float32(-1 / (BEAM_WAIST**2))
DETECTION_THRESHOLD_F32 = np.float32(DETECTION_THRESHOLD)

# FRET
FORSTER_RADIUS_F32 = np.float32(FORSTER_RADIUS)
I_EQ_F32           = np.float32(I_EQ)
DONOR_RATE_F32     = np.float32(DONOR_FLUORESCENCE_RATE)
ACCEPTOR_RATE_F32  = np.float32(ACCEPTOR_FLUORESCENCE_RATE)
RD_EQ_F32          = np.float32(DONOR_RADIUS_EQ)
RA_EQ_F32          = np.float32(ACCEPTOR_RADIUS_EQ)

PROB_QD = np.float32(PROB_DONOR_NONRADIATIVE)
PROB_QA = np.float32(PROB_ACCEPTOR_NONRADIATIVE)
PROB_AS = np.float32(PROB_ACCEPTOR_NOT_DETECTED)
PROB_DS = np.float32(PROB_DONOR_NOT_DETECTED)
PROB_ND = np.float32(PROB_NO_DETECTION)

BURST_MIN_GAP_F32 = np.float32(BURST_MIN_GAP)
MOTOR_CONF_X_F32  = np.float32(MOTOR_CONFINEMENT_X)
MOTOR_CONF_Y_F32  = np.float32(MOTOR_CONFINEMENT_Y)
MOTOR_CONF_Z_F32  = np.float32(MOTOR_CONFINEMENT_Z)


# =============================================================================
# BATCHED RNG CONSTANTS
# =============================================================================

RNG_BATCH_SIZE = 200_000  # Number of random numbers per refill


# =============================================================================
# SINGLE-WORKER JIT SIMULATION (all helpers inlined)
# =============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def run_simulation_worker(
    max_steps, target_bursts, burst_min_gap,
    init_motor_x, init_motor_y, init_motor_z,
    seed,
    # ── Translational noise (time-step independent) ──
    sqrt_2_donor_diff, sqrt_2_acceptor_diff,
    # ── Linker angular noise (time-step independent) ──
    sqrt_2_donor_theta, sqrt_2_acceptor_theta,
    theta_drift_k_d, theta_drift_k_a,
    donor_phi_diff, acceptor_phi_diff,
    # ── INSIDE constants ──
    cargo_dt_in, motor_dt_in, sqrt_cdt_in, sqrt_2_motor_in,
    cargo_spring_in, cargo_spring_half_in,
    donor_rad_in, acceptor_rad_in,
    two_dt_in,
    sqrt_2_dip_d_dt_in, sqrt_2_dip_a_dt_in,
    # ── OUTSIDE constants ──
    cargo_dt_out, motor_dt_out, sqrt_cdt_out, sqrt_2_motor_out,
    cargo_spring_out, cargo_spring_half_out,
    donor_rad_out, acceptor_rad_out,
    two_dt_out,
    sqrt_2_dip_d_dt_out, sqrt_2_dip_a_dt_out,
    # ── Dye equilibria ──
    rd_eq, ra_eq, i_eq,
    # ── Volume / intensity ──
    a_sq, b_sq, c_sq, c_k_sq, minus_2_w_sq, detection_threshold,
    # ── FRET ──
    forster_r, donor_rate, acceptor_rate,
    # ── Probabilities ──
    prob_qd, prob_qa, prob_as, prob_ds, prob_nd,
    # ── Confinement ──
    conf_x, conf_y, conf_z,
    # ── Motor countdown ratios ──
    motor_ratio_in, motor_ratio_out,
    # ── RNG batch size ──
    rng_batch):
    """
    Single-worker ultra-optimized EPK simulation.

    Optimizations over previous version:
      - Batched RNG (randn block pre-generated, indexed into)
      - Integer motor countdown (no float time comparison)
      - Early 97.5% no-detection rejection
      - All helpers inlined (no function-call overhead)
    """

    # Seed the per-worker RNG
    np.random.seed(seed)

    max_events = target_bursts * 200

    event_times         = np.zeros(max_events, dtype=np.float32)
    event_colors        = np.zeros(max_events, dtype=np.int32)
    event_donor_x       = np.zeros(max_events, dtype=np.float32)
    event_donor_y       = np.zeros(max_events, dtype=np.float32)
    event_donor_z       = np.zeros(max_events, dtype=np.float32)
    event_acceptor_x    = np.zeros(max_events, dtype=np.float32)
    event_acceptor_y    = np.zeros(max_events, dtype=np.float32)
    event_acceptor_z    = np.zeros(max_events, dtype=np.float32)
    event_interdye_dist = np.zeros(max_events, dtype=np.float32)
    event_fret_eff      = np.zeros(max_events, dtype=np.float32)
    event_donor_int     = np.zeros(max_events, dtype=np.float32)
    event_acceptor_int  = np.zeros(max_events, dtype=np.float32)

    f32_1e_n6 = np.float32(1e-6)
    f32_0pt01 = np.float32(0.01)
    f32_0     = np.float32(0.0)
    f32_1     = np.float32(1.0)
    f32_2     = np.float32(2.0)
    f32_3     = np.float32(3.0)
    f32_1pt5  = np.float32(1.5)
    eps_sin   = np.float32(1e-6)
    eps_tan   = np.float32(1e-3)
    two_pi    = np.float32(6.2831853)

    # ── Batched RNG buffers (OPT 1) ───────────────────────────────────────────
    randn_buf = np.random.randn(rng_batch).astype(np.float32)
    randu_buf = np.random.random(rng_batch).astype(np.float32)
    ri = np.int64(0)   # randn index
    ui = np.int64(0)   # randu index

    # ── Motor + cargo ──────────────────────────────────────────────────────────
    motor_x = np.float32(init_motor_x)
    motor_y = np.float32(init_motor_y)
    motor_z = np.float32(init_motor_z)
    cargo_x = np.float32(init_motor_x)
    cargo_y = np.float32(init_motor_y)
    cargo_z = np.float32(init_motor_z)

    # ── Linker spherical state ─────────────────────────────────────────────────
    r_d = np.float32(rd_eq) + np.float32(np.sqrt(0.5 * sqrt_2_donor_diff * sqrt_2_donor_diff)) * np.float32(np.random.randn())
    r_a = np.float32(ra_eq) + np.float32(np.sqrt(0.5 * sqrt_2_acceptor_diff * sqrt_2_acceptor_diff)) * np.float32(np.random.randn())
    if r_d < f32_1e_n6: r_d = f32_1e_n6
    if r_a < f32_1e_n6: r_a = f32_1e_n6

    th_d = np.float32(np.abs(np.random.normal(f32_0, np.float32(0.01))))
    th_a = np.float32(np.abs(np.random.normal(f32_0, np.float32(0.01))))
    ph_d = np.float32(np.random.random() * two_pi)
    ph_a = np.float32(np.random.random() * two_pi)

    # ── Dipole state: Cartesian unit vectors ───────────────────────────────────
    ddx = np.float32(np.random.randn())
    ddy = np.float32(np.random.randn())
    ddz = np.float32(np.random.randn())
    dn  = np.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
    if dn == f32_0: dn = f32_1
    ddx /= dn; ddy /= dn; ddz /= dn

    adx = np.float32(np.random.randn())
    ady = np.float32(np.random.randn())
    adz = np.float32(np.random.randn())
    an  = np.sqrt(adx*adx + ady*ady + adz*adz)
    if an == f32_0: an = f32_1
    adx /= an; ady /= an; adz /= an

    # ── Timers / counters ──────────────────────────────────────────────────────
    current_time     = np.float64(0.0)
    last_photon_time = np.float64(-1.0)

    total_photons   = np.int64(0)
    total_bursts    = np.int64(0)
    event_count     = np.int64(0)
    cargo_step      = np.int64(0)
    in_volume_count = np.int64(0)
    total_steps     = np.int64(0)

    was_in_volume = False

    # ── Integer motor countdown (OPT 3) ───────────────────────────────────────
    motor_countdown = np.int64(0)  # fire on first step

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    while cargo_step < max_steps and total_bursts < target_bursts:

        # ── Refill RNG buffers if needed (OPT 1) ─────────────────────────────
        # We need at most ~25 randn and ~8 randu per step (including photon sim)
        if ri + 30 >= rng_batch:
            randn_buf = np.random.randn(rng_batch).astype(np.float32)
            ri = np.int64(0)
        if ui + 10 >= rng_batch:
            randu_buf = np.random.random(rng_batch).astype(np.float32)
            ui = np.int64(0)

        # ── Select time-step regime ───────────────────────────────────────────
        if was_in_volume:
            cargo_dt   = cargo_dt_in
            sqrt_cdt   = sqrt_cdt_in
            sqrt2_m    = sqrt_2_motor_in
            c_spr      = cargo_spring_in
            c_spr_half = cargo_spring_half_in
            d_rad      = donor_rad_in
            a_rad      = acceptor_rad_in
            two_dt     = two_dt_in
            dip_d_amp  = sqrt_2_dip_d_dt_in
            dip_a_amp  = sqrt_2_dip_a_dt_in
            motor_ratio = motor_ratio_in
        else:
            cargo_dt   = cargo_dt_out
            sqrt_cdt   = sqrt_cdt_out
            sqrt2_m    = sqrt_2_motor_out
            c_spr      = cargo_spring_out
            c_spr_half = cargo_spring_half_out
            d_rad      = donor_rad_out
            a_rad      = acceptor_rad_out
            two_dt     = two_dt_out
            dip_d_amp  = sqrt_2_dip_d_dt_out
            dip_a_amp  = sqrt_2_dip_a_dt_out
            motor_ratio = motor_ratio_out

        # ── Motor update via integer countdown (OPT 3) ────────────────────────
        motor_countdown -= 1
        if motor_countdown <= 0:
            motor_x += sqrt2_m * randn_buf[ri];     ri += 1
            motor_y += sqrt2_m * randn_buf[ri];     ri += 1
            motor_z += sqrt2_m * randn_buf[ri];     ri += 1
            motor_countdown = motor_ratio

            # INLINED reflect (OPT 5) for motor_x
            span = f32_2 * conf_x
            v = motor_x + conf_x
            v = v - span * np.floor(v / span)
            if v > conf_x:
                v = span - v
            motor_x = v - conf_x

            # reflect motor_y
            span = f32_2 * conf_y
            v = motor_y + conf_y
            v = v - span * np.floor(v / span)
            if v > conf_y:
                v = span - v
            motor_y = v - conf_y

            # reflect motor_z
            span = f32_2 * conf_z
            v = motor_z + conf_z
            v = v - span * np.floor(v / span)
            if v > conf_z:
                v = span - v
            motor_z = v - conf_z

        # ── Cargo anchor (OU toward motor) ────────────────────────────────────
        noise_c = sqrt_2_donor_diff * sqrt_cdt
        cargo_x += c_spr      * (motor_x - cargo_x) + noise_c * randn_buf[ri]; ri += 1
        cargo_y += c_spr_half * (motor_y - cargo_y) + noise_c * randn_buf[ri]; ri += 1
        cargo_z += c_spr      * (motor_z - cargo_z) + noise_c * randn_buf[ri]; ri += 1

        # ── Linker radial dynamics ────────────────────────────────────────────
        if r_d < f32_1e_n6: r_d = f32_1e_n6
        if r_a < f32_1e_n6: r_a = f32_1e_n6

        r_d += d_rad * (rd_eq - r_d) + two_dt / r_d \
               + sqrt_2_donor_diff    * randn_buf[ri] * sqrt_cdt; ri += 1
        r_a += a_rad * (ra_eq - r_a) + two_dt / r_a \
               + sqrt_2_acceptor_diff * randn_buf[ri] * sqrt_cdt; ri += 1

        if r_d < f32_1e_n6: r_d = f32_1e_n6
        if r_a < f32_1e_n6: r_a = f32_1e_n6

        # ── Linker theta (cached sin/cos, derive tan) ─────────────────────────
        # DONOR
        sin_th_d = np.sin(th_d)
        cos_th_d = np.cos(th_d)
        tan_d    = sin_th_d / (cos_th_d + eps_tan)
        denom_d  = r_d * r_d * f32_2 * tan_d + f32_0pt01
        drift_d  = -f32_2 * sin_th_d + theta_drift_k_d / denom_d
        th_d    += drift_d * cargo_dt \
                   + (sqrt_2_donor_theta / r_d) * randn_buf[ri] * sqrt_cdt; ri += 1

        sin_phi_denom_d = sin_th_d if np.abs(sin_th_d) > eps_sin else eps_sin

        # INLINED wrap_2pi for ph_d (OPT 5)
        ph_d_new = ph_d + (donor_phi_diff / (r_d * r_d * sin_phi_denom_d)) \
                   * randn_buf[ri] * sqrt_cdt; ri += 1
        if ph_d_new >= two_pi:
            ph_d_new -= two_pi * np.floor(ph_d_new / two_pi)
        elif ph_d_new < f32_0:
            ph_d_new += two_pi * (f32_1 + np.floor((-ph_d_new) / two_pi))
        ph_d = ph_d_new

        # ACCEPTOR
        sin_th_a = np.sin(th_a)
        cos_th_a = np.cos(th_a)
        tan_a    = sin_th_a / (cos_th_a + eps_tan)
        denom_a  = r_a * r_a * f32_2 * tan_a + f32_0pt01
        drift_a  = -f32_2 * sin_th_a + theta_drift_k_a / denom_a
        th_a    += drift_a * cargo_dt \
                   + (sqrt_2_acceptor_theta / r_a) * randn_buf[ri] * sqrt_cdt; ri += 1

        sin_phi_denom_a = sin_th_a if np.abs(sin_th_a) > eps_sin else eps_sin

        ph_a_new = ph_a + (acceptor_phi_diff / (r_a * r_a * sin_phi_denom_a)) \
                   * randn_buf[ri] * sqrt_cdt; ri += 1
        if ph_a_new >= two_pi:
            ph_a_new -= two_pi * np.floor(ph_a_new / two_pi)
        elif ph_a_new < f32_0:
            ph_a_new += two_pi * (f32_1 + np.floor((-ph_a_new) / two_pi))
        ph_a = ph_a_new

        # ── Linker spherical → Cartesian (reuse cached sin/cos) ───────────────
        cos_ph_d = np.cos(ph_d);  sin_ph_d = np.sin(ph_d)
        cos_ph_a = np.cos(ph_a);  sin_ph_a = np.sin(ph_a)

        dx_local = r_d * sin_th_d * cos_ph_d
        dy_local = r_d * sin_th_d * sin_ph_d
        dz_local = r_d * cos_th_d

        ax_local = r_a * sin_th_a * cos_ph_a
        ay_local = r_a * sin_th_a * sin_ph_a
        az_local = r_a * cos_th_a

        # ── Absolute dye positions ─────────────────────────────────────────────
        donor_x    = cargo_x + dx_local + i_eq
        donor_y    = cargo_y + dy_local
        donor_z    = cargo_z + dz_local
        acceptor_x = cargo_x + ax_local
        acceptor_y = cargo_y + ay_local
        acceptor_z = cargo_z + az_local

        # ── INLINED confocal volume check (OPT 5) ─────────────────────────────
        donor_in = (donor_x*donor_x)/a_sq + (donor_y*donor_y)/b_sq + (donor_z*donor_z)/c_sq <= f32_1
        acceptor_in = (acceptor_x*acceptor_x)/a_sq + (acceptor_y*acceptor_y)/b_sq + (acceptor_z*acceptor_z)/c_sq <= f32_1

        total_steps += 1

        donor_intensity    = f32_0
        acceptor_intensity = f32_0

        if donor_in:
            # INLINED intensity calculation (OPT 5)
            exponent = minus_2_w_sq * ((donor_x*donor_x)/a_sq + (donor_y*donor_y)/b_sq + (donor_z*donor_z)/c_k_sq)
            donor_intensity = np.exp(exponent)
            donor_in = donor_intensity >= detection_threshold

        if acceptor_in:
            exponent = minus_2_w_sq * ((acceptor_x*acceptor_x)/a_sq + (acceptor_y*acceptor_y)/b_sq + (acceptor_z*acceptor_z)/c_k_sq)
            acceptor_intensity = np.exp(exponent)
            acceptor_in = acceptor_intensity >= detection_threshold

        currently_in_volume = donor_in or acceptor_in

        # ── Photon simulation (only inside volume) ─────────────────────────────
        if currently_in_volume:
            in_volume_count += 1

            ix = (dx_local + i_eq) - ax_local
            iy = dy_local - ay_local
            iz = dz_local - az_local
            dist_sq = ix*ix + iy*iy + iz*iz
            if dist_sq <= np.float32(1e-12):
                dist_sq = np.float32(1e-12)
            dist     = np.sqrt(dist_sq)
            fret_eff = f32_1 / (f32_1 + (dist / forster_r) ** np.float32(6))

            # ── Evolve dipole: Cartesian OU + normalise ──────────────────────
            ddx += dip_d_amp * randn_buf[ri]; ri += 1
            ddy += dip_d_amp * randn_buf[ri]; ri += 1
            ddz += dip_d_amp * randn_buf[ri]; ri += 1
            dn   = np.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
            if dn == f32_0: dn = f32_1
            ddx /= dn;  ddy /= dn;  ddz /= dn

            adx += dip_a_amp * randn_buf[ri]; ri += 1
            ady += dip_a_amp * randn_buf[ri]; ri += 1
            adz += dip_a_amp * randn_buf[ri]; ri += 1
            an   = np.sqrt(adx*adx + ady*ady + adz*adz)
            if an == f32_0: an = f32_1
            adx /= an;  ady /= an;  adz /= an

            # ── kappa² ──────────────────────────────────────────────────────
            inv_dist = f32_1 / dist
            rx = ix * inv_dist;  ry = iy * inv_dist;  rz = iz * inv_dist
            term1    = ddx*adx + ddy*ady + ddz*adz
            term2    = (ddx*rx + ddy*ry + ddz*rz) * (adx*rx + ady*ry + adz*rz)
            kappa_sq = (term1 - f32_3 * term2) ** np.float32(2)

            da_rate = donor_rate * f32_1pt5 * kappa_sq * (forster_r * inv_dist) ** np.float32(6)

            # ── Photon emission with EARLY REJECTION (OPT 4) ─────────────────
            total_int = donor_intensity + acceptor_intensity
            if total_int > f32_0:
                # 97.5% of the time, no detection — skip the expensive path
                if randu_buf[ui] >= prob_nd:
                    ui += 1
                    # Not rejected by no-detection — run photon simulation
                    if randu_buf[ui] >= prob_qd:
                        ui += 1

                        # Exponential race: FRET vs donor decay
                        # Use randn_buf entries reinterpreted as exponential via -log(uniform)
                        inv_da = f32_1 / da_rate if da_rate > f32_0 else np.float32(1e9)
                        inv_donor = f32_1 / donor_rate

                        t_fret  = -inv_da    * np.log(randu_buf[ui]  + np.float32(1e-30)); ui += 1
                        t_donor = -inv_donor * np.log(randu_buf[ui]  + np.float32(1e-30)); ui += 1

                        photon_state = np.int32(3)
                        is_detected  = np.int32(0)

                        if t_fret < t_donor:
                            if t_fret <= cargo_dt:
                                if randu_buf[ui] >= prob_qa:
                                    ui += 1
                                    inv_acc = f32_1 / acceptor_rate
                                    t_acc_em = -inv_acc * np.log(randu_buf[ui] + np.float32(1e-30)); ui += 1
                                    if t_fret + t_acc_em <= cargo_dt:
                                        if randu_buf[ui] >= prob_as:
                                            ui += 1
                                            photon_state = np.int32(5)
                                            is_detected  = np.int32(1)
                                        else:
                                            ui += 1
                                    else:
                                        pass  # acceptor emission too late
                                else:
                                    ui += 1
                            # else: FRET too late
                        else:
                            if t_donor <= cargo_dt:
                                if randu_buf[ui] >= prob_ds:
                                    ui += 1
                                    photon_state = np.int32(4)
                                    is_detected  = np.int32(1)
                                else:
                                    ui += 1
                            # else: donor emission too late

                        if is_detected == 1 and event_count < max_events:
                            if last_photon_time < 0.0 or \
                               (current_time - last_photon_time) >= burst_min_gap:
                                total_bursts += np.int64(1)
                            last_photon_time = current_time

                            event_times[event_count]         = np.float32(current_time)
                            event_colors[event_count]        = photon_state
                            event_donor_x[event_count]       = donor_x
                            event_donor_y[event_count]       = donor_y
                            event_donor_z[event_count]       = donor_z
                            event_acceptor_x[event_count]    = acceptor_x
                            event_acceptor_y[event_count]    = acceptor_y
                            event_acceptor_z[event_count]    = acceptor_z
                            event_interdye_dist[event_count] = dist
                            event_fret_eff[event_count]      = fret_eff
                            event_donor_int[event_count]     = donor_intensity
                            event_acceptor_int[event_count]  = acceptor_intensity
                            event_count   += np.int64(1)
                            total_photons += np.int64(1)
                    else:
                        ui += 1  # consumed qd uniform
                else:
                    ui += 1  # consumed nd uniform

        was_in_volume = currently_in_volume
        current_time += cargo_dt
        cargo_step   += 1

        if event_count >= max_events:
            break

    occupancy = np.float32(in_volume_count) / np.float32(total_steps) \
                if total_steps > 0 else np.float32(0.0)

    return (cargo_step, total_photons, total_bursts, event_count, occupancy,
            np.float32(current_time),
            event_times, event_colors,
            event_donor_x, event_donor_y, event_donor_z,
            event_acceptor_x, event_acceptor_y, event_acceptor_z,
            event_interdye_dist, event_fret_eff,
            event_donor_int, event_acceptor_int)


# =============================================================================
# PARALLEL DISPATCH (OPT 2)
# =============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def run_parallel(
    n_workers, max_steps_per_worker, bursts_per_worker, burst_min_gap,
    seeds,
    # ── all physics constants passed through ──
    sqrt_2_donor_diff, sqrt_2_acceptor_diff,
    sqrt_2_donor_theta, sqrt_2_acceptor_theta,
    theta_drift_k_d, theta_drift_k_a,
    donor_phi_diff, acceptor_phi_diff,
    cargo_dt_in, motor_dt_in, sqrt_cdt_in, sqrt_2_motor_in,
    cargo_spring_in, cargo_spring_half_in,
    donor_rad_in, acceptor_rad_in,
    two_dt_in, sqrt_2_dip_d_dt_in, sqrt_2_dip_a_dt_in,
    cargo_dt_out, motor_dt_out, sqrt_cdt_out, sqrt_2_motor_out,
    cargo_spring_out, cargo_spring_half_out,
    donor_rad_out, acceptor_rad_out,
    two_dt_out, sqrt_2_dip_d_dt_out, sqrt_2_dip_a_dt_out,
    rd_eq, ra_eq, i_eq,
    a_sq, b_sq, c_sq, c_k_sq, minus_2_w_sq, detection_threshold,
    forster_r, donor_rate, acceptor_rate,
    prob_qd, prob_qa, prob_as, prob_ds, prob_nd,
    conf_x, conf_y, conf_z,
    motor_ratio_in, motor_ratio_out,
    rng_batch):
    """
    Dispatch n_workers parallel trajectories.
    Each worker contributes roughly bursts_per_worker bursts.
    Results are stored in per-worker slots of shared arrays.
    """

    max_events_per = bursts_per_worker * 200

    # Per-worker result accumulators
    all_cargo_steps = np.zeros(n_workers, dtype=np.int64)
    all_photons     = np.zeros(n_workers, dtype=np.int64)
    all_bursts      = np.zeros(n_workers, dtype=np.int64)
    all_events      = np.zeros(n_workers, dtype=np.int64)
    all_occupancy   = np.zeros(n_workers, dtype=np.float32)
    all_sim_time    = np.zeros(n_workers, dtype=np.float32)

    # Concatenated event storage
    total_max_events = n_workers * max_events_per
    c_times         = np.zeros(total_max_events, dtype=np.float32)
    c_colors        = np.zeros(total_max_events, dtype=np.int32)
    c_donor_x       = np.zeros(total_max_events, dtype=np.float32)
    c_donor_y       = np.zeros(total_max_events, dtype=np.float32)
    c_donor_z       = np.zeros(total_max_events, dtype=np.float32)
    c_acceptor_x    = np.zeros(total_max_events, dtype=np.float32)
    c_acceptor_y    = np.zeros(total_max_events, dtype=np.float32)
    c_acceptor_z    = np.zeros(total_max_events, dtype=np.float32)
    c_interdye      = np.zeros(total_max_events, dtype=np.float32)
    c_fret          = np.zeros(total_max_events, dtype=np.float32)
    c_d_int         = np.zeros(total_max_events, dtype=np.float32)
    c_a_int         = np.zeros(total_max_events, dtype=np.float32)

    for w in prange(n_workers):
        res = run_simulation_worker(
            max_steps_per_worker, bursts_per_worker, burst_min_gap,
            np.float32(0.0), np.float32(0.0), np.float32(0.0),
            seeds[w],
            sqrt_2_donor_diff, sqrt_2_acceptor_diff,
            sqrt_2_donor_theta, sqrt_2_acceptor_theta,
            theta_drift_k_d, theta_drift_k_a,
            donor_phi_diff, acceptor_phi_diff,
            cargo_dt_in, motor_dt_in, sqrt_cdt_in, sqrt_2_motor_in,
            cargo_spring_in, cargo_spring_half_in,
            donor_rad_in, acceptor_rad_in,
            two_dt_in, sqrt_2_dip_d_dt_in, sqrt_2_dip_a_dt_in,
            cargo_dt_out, motor_dt_out, sqrt_cdt_out, sqrt_2_motor_out,
            cargo_spring_out, cargo_spring_half_out,
            donor_rad_out, acceptor_rad_out,
            two_dt_out, sqrt_2_dip_d_dt_out, sqrt_2_dip_a_dt_out,
            rd_eq, ra_eq, i_eq,
            a_sq, b_sq, c_sq, c_k_sq, minus_2_w_sq, detection_threshold,
            forster_r, donor_rate, acceptor_rate,
            prob_qd, prob_qa, prob_as, prob_ds, prob_nd,
            conf_x, conf_y, conf_z,
            motor_ratio_in, motor_ratio_out,
            rng_batch)

        (cs, tp, tb, ne, occ, st,
         e_t, e_c, e_dx, e_dy, e_dz,
         e_ax, e_ay, e_az,
         e_id, e_fe, e_di, e_ai) = res

        all_cargo_steps[w] = cs
        all_photons[w]     = tp
        all_bursts[w]      = tb
        all_events[w]      = ne
        all_occupancy[w]   = occ
        all_sim_time[w]    = st

        offset = w * max_events_per
        for j in range(ne):
            c_times[offset + j]      = e_t[j]
            c_colors[offset + j]     = e_c[j]
            c_donor_x[offset + j]    = e_dx[j]
            c_donor_y[offset + j]    = e_dy[j]
            c_donor_z[offset + j]    = e_dz[j]
            c_acceptor_x[offset + j] = e_ax[j]
            c_acceptor_y[offset + j] = e_ay[j]
            c_acceptor_z[offset + j] = e_az[j]
            c_interdye[offset + j]   = e_id[j]
            c_fret[offset + j]       = e_fe[j]
            c_d_int[offset + j]      = e_di[j]
            c_a_int[offset + j]      = e_ai[j]

    return (all_cargo_steps, all_photons, all_bursts, all_events,
            all_occupancy, all_sim_time,
            c_times, c_colors,
            c_donor_x, c_donor_y, c_donor_z,
            c_acceptor_x, c_acceptor_y, c_acceptor_z,
            c_interdye, c_fret, c_d_int, c_a_int,
            max_events_per)


# =============================================================================
# INITIALIZE AND RUN
# =============================================================================

print("Initializing simulation...")
print(f"Using {N_WORKERS} parallel workers, each targeting ~{TARGET_BURSTS // N_WORKERS} bursts")
print("Note: First run includes JIT compilation (~1-2 min)\n")

max_simulation_steps = 10_000_000_000
bursts_per_worker = (TARGET_BURSTS + N_WORKERS - 1) // N_WORKERS

# Generate unique seeds for each worker
master_rng = np.random.default_rng(42)
seeds = master_rng.integers(0, 2**31, size=N_WORKERS).astype(np.int64)

print(f"Maximum steps per worker (safety cap): {max_simulation_steps:,}")
print(f"Bursts per worker:                     {bursts_per_worker:,}")
print("Starting simulation...\n")
start_time = time.time()

par_results = run_parallel(
    N_WORKERS, max_simulation_steps, bursts_per_worker, BURST_MIN_GAP_F32,
    seeds,
    SQRT_2_DONOR_DIFF, SQRT_2_ACCEPTOR_DIFF,
    SQRT_2_DONOR_THETA_DIFF, SQRT_2_ACCEPTOR_THETA_DIFF,
    THETA_DRIFT_K_D, THETA_DRIFT_K_A,
    DONOR_PHI_DIFF_F32, ACCEPTOR_PHI_DIFF_F32,
    CARGO_DT_IN_F32,  MOTOR_DT_IN_F32,  SQRT_CARGO_DT_IN,  SQRT_2_MOTOR_DIFF_IN,
    CARGO_SPRING_IN,  CARGO_SPRING_HALF_IN,
    DONOR_RADIAL_TERM_IN, ACCEPTOR_RADIAL_TERM_IN,
    TWO_DT_IN, SQRT_2_DIP_D_DT_IN, SQRT_2_DIP_A_DT_IN,
    CARGO_DT_OUT_F32, MOTOR_DT_OUT_F32, SQRT_CARGO_DT_OUT, SQRT_2_MOTOR_DIFF_OUT,
    CARGO_SPRING_OUT, CARGO_SPRING_HALF_OUT,
    DONOR_RADIAL_TERM_OUT, ACCEPTOR_RADIAL_TERM_OUT,
    TWO_DT_OUT, SQRT_2_DIP_D_DT_OUT, SQRT_2_DIP_A_DT_OUT,
    RD_EQ_F32, RA_EQ_F32, I_EQ_F32,
    ELLIPSOID_A_SQ, ELLIPSOID_B_SQ, ELLIPSOID_C_SQ,
    ELLIPSOID_C_K_SQ, MINUS_2_OVER_WAIST_SQ, DETECTION_THRESHOLD_F32,
    FORSTER_RADIUS_F32, DONOR_RATE_F32, ACCEPTOR_RATE_F32,
    PROB_QD, PROB_QA, PROB_AS, PROB_DS, PROB_ND,
    MOTOR_CONF_X_F32, MOTOR_CONF_Y_F32, MOTOR_CONF_Z_F32,
    np.int64(MOTOR_RATIO_INSIDE), np.int64(MOTOR_RATIO_OUTSIDE),
    np.int64(RNG_BATCH_SIZE)
)

elapsed_time = time.time() - start_time


# =============================================================================
# UNPACK, MERGE, AND TRIM RESULTS FROM ALL WORKERS
# =============================================================================

(all_cargo_steps, all_photons, all_bursts, all_events,
 all_occupancy, all_sim_time,
 c_times, c_colors,
 c_donor_x, c_donor_y, c_donor_z,
 c_acceptor_x, c_acceptor_y, c_acceptor_z,
 c_interdye, c_fret, c_d_int, c_a_int,
 max_events_per) = par_results

# Aggregate per-worker stats
num_cargo_steps         = int(np.sum(all_cargo_steps))
total_photons           = int(np.sum(all_photons))
total_bursts_collected  = int(np.sum(all_bursts))
total_events            = int(np.sum(all_events))
occupancy               = float(np.mean(all_occupancy))
total_simulation_time   = float(np.max(all_sim_time))

# Merge events from all workers into contiguous arrays
merge_indices = []
for w in range(N_WORKERS):
    ne = int(all_events[w])
    offset = w * max_events_per
    for j in range(ne):
        merge_indices.append(offset + j)

merge_idx = np.array(merge_indices, dtype=np.int64)

event_times         = c_times[merge_idx]
event_colors        = c_colors[merge_idx]
event_donor_x       = c_donor_x[merge_idx]
event_donor_y       = c_donor_y[merge_idx]
event_donor_z       = c_donor_z[merge_idx]
event_acceptor_x    = c_acceptor_x[merge_idx]
event_acceptor_y    = c_acceptor_y[merge_idx]
event_acceptor_z    = c_acceptor_z[merge_idx]
event_interdye_dist = c_interdye[merge_idx]
event_fret_eff      = c_fret[merge_idx]
event_donor_intensity    = c_d_int[merge_idx]
event_acceptor_intensity = c_a_int[merge_idx]

n = len(merge_idx)

# Sort by time (events come from independent workers with their own timelines)
sort_idx = np.argsort(event_times)
event_times         = event_times[sort_idx]
event_colors        = event_colors[sort_idx]
event_donor_x       = event_donor_x[sort_idx]
event_donor_y       = event_donor_y[sort_idx]
event_donor_z       = event_donor_z[sort_idx]
event_acceptor_x    = event_acceptor_x[sort_idx]
event_acceptor_y    = event_acceptor_y[sort_idx]
event_acceptor_z    = event_acceptor_z[sort_idx]
event_interdye_dist = event_interdye_dist[sort_idx]
event_fret_eff      = event_fret_eff[sort_idx]
event_donor_intensity    = event_donor_intensity[sort_idx]
event_acceptor_intensity = event_acceptor_intensity[sort_idx]


print(f"\n{'='*70}")
print("SIMULATION COMPLETE!")
print(f"{'='*70}")
print(f"Workers used:               {N_WORKERS}")
print(f"Execution time:             {elapsed_time:.1f} s  ({elapsed_time/60:.1f} min)")
for w in range(N_WORKERS):
    print(f"  Worker {w}: {int(all_bursts[w]):,} bursts, "
          f"{int(all_events[w]):,} events, "
          f"{int(all_cargo_steps[w]):,} steps, "
          f"occ={all_occupancy[w]*100:.3f}%")
print(f"Total bursts collected:     {total_bursts_collected:,}  (target: {TARGET_BURSTS:,})")
print(f"Total photons collected:    {total_photons:,}")
print(f"Simulated physical time:    {total_simulation_time:.4f} s (longest worker)")
print(f"Total cargo steps:          {num_cargo_steps:,}")
print(f"Mean volume occupancy:      {occupancy*100:.3f}%")
print(f"Data reduction factor:      {num_cargo_steps/max(n, 1):.1f}x")
print(f"Simulation speed:           {num_cargo_steps/elapsed_time:,.0f} steps/s")
print(f"{'='*70}\n")


# =============================================================================
# ANALYZE PHOTON EVENTS
# =============================================================================

print("Analyzing photon events...")
donor_mask     = event_colors == 4
acceptor_mask  = event_colors == 5
donor_count    = int(np.sum(donor_mask))
acceptor_count = int(np.sum(acceptor_mask))

print(f"\nPhoton statistics:")
print(f"  Donor photons:          {donor_count:,}")
print(f"  Acceptor photons:       {acceptor_count:,}")
print(f"  Acceptor/Total ratio:   {acceptor_count/max(total_photons, 1):.3f}")
if n > 0:
    print(f"  Mean FRET efficiency:   {np.mean(event_fret_eff):.3f}")
    print(f"  Mean interdye distance: {np.mean(event_interdye_dist):.3f} nm")
    print(f"  Std  interdye distance: {np.std(event_interdye_dist):.3f} nm")


# =============================================================================
# POST-HOC BURST ANALYSIS
# =============================================================================

print("\nCreating photon burst analysis...")


def identify_bursts_combined(times, colors, min_gap=0.001):
    if len(times) == 0:
        return []
    bursts = []
    idx = [0]
    for i in range(1, len(times)):
        if times[i] - times[idx[-1]] < min_gap:
            idx.append(i)
        else:
            bc = colors[idx]
            bursts.append({
                'entry_time':       float(times[idx[0]]),
                'exit_time':        float(times[idx[-1]]),
                'duration':         float(times[idx[-1]] - times[idx[0]]),
                'total_photons':    len(idx),
                'donor_photons':    int(np.sum(bc == 4)),
                'acceptor_photons': int(np.sum(bc == 5))
            })
            idx = [i]
    bc = colors[idx]
    bursts.append({
        'entry_time':       float(times[idx[0]]),
        'exit_time':        float(times[idx[-1]]),
        'duration':         float(times[idx[-1]] - times[idx[0]]),
        'total_photons':    len(idx),
        'donor_photons':    int(np.sum(bc == 4)),
        'acceptor_photons': int(np.sum(bc == 5))
    })
    return bursts


posthoc_gap = BURST_MIN_GAP if BURST_MIN_GAP > 0 else 1e-6
all_bursts = identify_bursts_combined(event_times, event_colors, min_gap=posthoc_gap)

print(f"  Bursts tracked in-loop:     {total_bursts_collected:,}")
print(f"  Bursts identified post-hoc: {len(all_bursts):,}")
if len(all_bursts) > 0:
    print(f"  Avg photons/burst:          {np.mean([b['total_photons']    for b in all_bursts]):.1f}")
    print(f"  Avg donor photons/burst:    {np.mean([b['donor_photons']    for b in all_bursts]):.1f}")
    print(f"  Avg acceptor photons/burst: {np.mean([b['acceptor_photons'] for b in all_bursts]):.1f}")


# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\nSaving results to CSV...")

event_df = pd.DataFrame({
    'time':               event_times,
    'photon_type':        ['donor' if c == 4 else 'acceptor' for c in event_colors],
    'donor_x':            event_donor_x,
    'donor_y':            event_donor_y,
    'donor_z':            event_donor_z,
    'acceptor_x':         event_acceptor_x,
    'acceptor_y':         event_acceptor_y,
    'acceptor_z':         event_acceptor_z,
    'interdye_distance':  event_interdye_dist,
    'fret_efficiency':    event_fret_eff,
    'donor_intensity':    event_donor_intensity,
    'acceptor_intensity': event_acceptor_intensity
})
event_df.to_csv('photon_events_nanosecond_epk.csv', index=False)
print(f"  Saved {len(event_df):,} photon events to 'photon_events_nanosecond_epk.csv'")

burst_df = pd.DataFrame(all_bursts)
summary_rows = [
    f'total_photons={total_photons}',
    f'donor_photons={donor_count}',
    f'acceptor_photons={acceptor_count}',
    f'total_bursts={total_bursts_collected}',
    f'target_bursts={TARGET_BURSTS}',
    f'simulation_time={total_simulation_time:.4f}s',
    f'volume_occupancy={occupancy:.4f}',
    f'cargo_dt_in={CARGO_TIME_STEP_INSIDE}s',
    f'cargo_dt_out={CARGO_TIME_STEP_OUTSIDE}s',
    f'confinement={MOTOR_CONFINEMENT_X:.0f}x{MOTOR_CONFINEMENT_Y:.0f}x{MOTOR_CONFINEMENT_Z:.0f}nm',
    f'forster_radius={FORSTER_RADIUS}nm',
    f'i_eq={I_EQ}nm',
    f'donor_dipole_diff={DONOR_DIPOLE_DIFF:.4e}',
    f'execution_time={elapsed_time:.1f}s',
    f'n_workers={N_WORKERS}',
    f'optimizations=batched_rng+parallel+int_motor+early_reject+inlined',
]
summary_df = pd.DataFrame({
    'entry_time':       ['SUMMARY'] * len(summary_rows),
    'exit_time':        summary_rows,
    'duration':         [''] * len(summary_rows),
    'total_photons':    [''] * len(summary_rows),
    'donor_photons':    [''] * len(summary_rows),
    'acceptor_photons': [''] * len(summary_rows),
})
if len(all_bursts) > 0:
    burst_df = burst_df[['entry_time', 'exit_time', 'duration',
                          'total_photons', 'donor_photons', 'acceptor_photons']]
final_burst_df = pd.concat([summary_df, burst_df], ignore_index=True)
final_burst_df.to_csv('photon_bursts_nanosecond_epk.csv', index=False)
print(f"  Saved burst analysis to 'photon_bursts_nanosecond_epk.csv'\n")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"{'='*70}")
print("ULTRA-OPTIMIZED EPK SIMULATION COMPLETE!")
print(f"{'='*70}")
print(f"  Workers:          {N_WORKERS}")
print(f"  Execution time:   {elapsed_time:.1f} s ({elapsed_time/60:.1f} min)")
print(f"  Steps/second:     {num_cargo_steps/elapsed_time:,.0f}")
print(f"  Photons/second:   {total_photons/elapsed_time:,.0f}")
print(f"  Bursts/second:    {total_bursts_collected/elapsed_time:,.1f}")
print(f"  Memory (events):  {n*12*4/1e6:.1f} MB  "
      f"(vs {num_cargo_steps*13*4/1e6:.1f} MB if storing all steps)")
print(f"\nFiles saved:")
print(f"  - photon_events_nanosecond_epk.csv  ({len(event_df):,} events)")
print(f"  - photon_bursts_nanosecond_epk.csv  ({len(all_bursts):,} bursts)")
print(f"{'='*70}\n")
