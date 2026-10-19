"""
FRET Simulation - ELASTIC PENDULUM + KAPPA DYNAMICS (SPEED-OPTIMIZED)
======================================================================

All physics identical to the EPK version. Speed improvements:

  1. LINKER  - cache sin(theta)/cos(theta) once per dye per step.
               tan θ = sin/cos (no separate tan call).
               The same cached values feed both the SDE drift AND the
               spherical→Cartesian conversion, so each dye costs only
               4 trig ops (sin_th, cos_th, sin_ph, cos_ph) instead of 6.
               Saves 4 trig calls per step always (8 total, 2 dyes).

  2. DIPOLE  - replace (theta_dip, phi_dip) spherical SDE with a 3-component
               Cartesian OU vector that is normalised each step.
               Dipole rotational diffusion is isotropic (single D_rot for
               both axes), so normalised 3-D BM gives the same uniform
               stationary distribution on S² as the spherical Ito SDE — no
               Ito correction drift needed in Cartesian coordinates.
               Saves 6 trig calls per dye per inside step (12 total, 2 dyes).

  3. CONSTANTS - pre-compute the following loop-invariant values:
               THETA_DRIFT_K_{D,A}  = sqrt_2_theta² × 0.25
                                     = 0.5 × theta_diff  (time-step independent)
               TWO_DT_{IN,OUT}      = 2.0 × cargo_dt     (for centrifugal term)
               SQRT_2_DIP_DT_{IN,OUT} = sqrt(2 × D_dip × dt)  (dipole noise amp)

  Combined, these save ≈ 8 trig calls per step outside the volume and
  ≈ 20 trig calls per step inside the volume.

Outputs:
  - photon_events_nanosecond.csv
  - photon_bursts_nanosecond.csv
"""

import numpy as np
import pandas as pd
from numba import jit
import time


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

# Motor diffusion: 100 µm²/s = 1e8 nm²/s
MOTOR_DIFFUSION = 1e8  # nm²/s

# Rotational diffusion of dipole (Stokes: kT / (8π η r³))
DONOR_DIPOLE_DIFF    = kT / (8 * np.pi * VISCOSITY * DONOR_RADIUS**3)
ACCEPTOR_DIPOLE_DIFF = kT / (8 * np.pi * VISCOSITY * ACCEPTOR_RADIUS**3)

# Linker spring constants
DONOR_NUM_LINKS    = 3
ACCEPTOR_NUM_LINKS = 3
DONOR_SPRING_CONSTANT    = (1010 / DONOR_NUM_LINKS)    * 1e-9  # N/nm
ACCEPTOR_SPRING_CONSTANT = (1010 / ACCEPTOR_NUM_LINKS) * 1e-9  # N/nm

DONOR_K_HAT    = DONOR_SPRING_CONSTANT    / DONOR_FRICTION
ACCEPTOR_K_HAT = ACCEPTOR_SPRING_CONSTANT / ACCEPTOR_FRICTION

DONOR_RADIUS_EQ    = 0.1255 * DONOR_NUM_LINKS     # nm
ACCEPTOR_RADIUS_EQ = 0.1255 * ACCEPTOR_NUM_LINKS  # nm

# Angular diffusion scale factors for linker theta and phi
WD_THETA = 20.0;  WA_THETA = 20.0
WD_PHI   = 200.0; WA_PHI   = 200.0

DONOR_THETA_DIFF    = WD_THETA * DONOR_DIFFUSION
ACCEPTOR_THETA_DIFF = WA_THETA * ACCEPTOR_DIFFUSION
DONOR_PHI_DIFF      = WD_PHI   * DONOR_DIFFUSION
ACCEPTOR_PHI_DIFF   = WA_PHI   * ACCEPTOR_DIFFUSION

# Cargo anchor spring
CARGO_SPRING_CONSTANT = (1010 / 12) * 1e-9  # N/nm
CARGO_K_HAT           = CARGO_SPRING_CONSTANT / DONOR_FRICTION

# Fluorescence rates
DONOR_FLUORESCENCE_RATE    = (3/2) / (4e-5)  # s⁻¹
ACCEPTOR_FLUORESCENCE_RATE = (3/2) / (4e-5)  # s⁻¹

# FRET parameters
FORSTER_RADIUS = 5.0              # nm
I_EQ           = 0.7 * FORSTER_RADIUS  # nm

# Detection probabilities
PROB_DONOR_NONRADIATIVE    = 0.15
PROB_ACCEPTOR_NONRADIATIVE = 0.21
PROB_ACCEPTOR_NOT_DETECTED = 0.25
PROB_DONOR_NOT_DETECTED    = 0.15
PROB_NO_DETECTION          = 0.975


# =============================================================================
# ADAPTIVE TIME STEP PARAMETERS
# =============================================================================

MOTOR_TIME_STEP_INSIDE  = 10e-6    # 10 µs
CARGO_TIME_STEP_INSIDE  = 100e-9   # 100 ns

MOTOR_TIME_STEP_OUTSIDE = 1000e-6  # 1 ms
CARGO_TIME_STEP_OUTSIDE = 10e-6    # 10 µs


# =============================================================================
# BURST TARGET
# =============================================================================

TARGET_BURSTS = 10000
BURST_MIN_GAP = 0.0  # seconds


# =============================================================================
# CONFOCAL MICROSCOPY PARAMETERS
# =============================================================================

ELLIPSOID_SEMI_AXIS_X = 250    # nm
ELLIPSOID_SEMI_AXIS_Y = 250    # nm
ELLIPSOID_SEMI_AXIS_Z = 1000   # nm

REFRACTIVE_INDEX    = 1.3
NUMERICAL_APERTURE  = 1.2
WAVE_VECTOR         = 2.33 * REFRACTIVE_INDEX / NUMERICAL_APERTURE
BEAM_WAIST          = (1 / (np.pi**(3/2) * WAVE_VECTOR))**(1/3)
DETECTION_THRESHOLD = 1 / np.exp(2)


# =============================================================================
# MOTOR CONFINEMENT
# =============================================================================

CONFINEMENT_FACTOR  = 4
MOTOR_CONFINEMENT_X = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_X  # ±1000 nm
MOTOR_CONFINEMENT_Y = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Y  # ±1000 nm
MOTOR_CONFINEMENT_Z = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Z  # ±4000 nm


# =============================================================================
# DISPLAY SETUP
# =============================================================================

print("\n" + "="*70)
print("FRET - ELASTIC PENDULUM + KAPPA (SPEED-OPTIMIZED)")
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
print()
print("Speed optimizations active:")
print("  + Linker sin/cos cached (4 trig/dye instead of 6)")
print("  + Dipole Cartesian OU + normalize (replaces 6 trig/dye)")
print("  + Pre-computed theta drift coefficient and centrifugal dt")
print("  + Pre-computed dipole noise amplitudes for IN/OUT regimes")
print("="*70 + "\n")


# =============================================================================
# PRE-COMPUTE CONSTANTS (FLOAT32)
# =============================================================================

TWO_PI_F32 = np.float32(2 * np.pi)

# ── Translational noise (time-step independent) ──────────────────────────────
SQRT_2_DONOR_DIFF    = np.float32(np.sqrt(2 * DONOR_DIFFUSION))
SQRT_2_ACCEPTOR_DIFF = np.float32(np.sqrt(2 * ACCEPTOR_DIFFUSION))

# ── Linker theta noise (time-step independent) ───────────────────────────────
SQRT_2_DONOR_THETA_DIFF    = np.float32(np.sqrt(2 * DONOR_THETA_DIFF))
SQRT_2_ACCEPTOR_THETA_DIFF = np.float32(np.sqrt(2 * ACCEPTOR_THETA_DIFF))

# OPT 3: theta drift coefficient = sqrt_2_theta² × 0.25 = 0.5 × theta_diff
# Pre-computed so it is not re-evaluated every step.
THETA_DRIFT_K_D = np.float32(0.5 * DONOR_THETA_DIFF)
THETA_DRIFT_K_A = np.float32(0.5 * ACCEPTOR_THETA_DIFF)

DONOR_PHI_DIFF_F32    = np.float32(DONOR_PHI_DIFF)
ACCEPTOR_PHI_DIFF_F32 = np.float32(ACCEPTOR_PHI_DIFF)

# ── INSIDE time-step dependent constants ─────────────────────────────────────
CARGO_DT_IN_F32         = np.float32(CARGO_TIME_STEP_INSIDE)
MOTOR_DT_IN_F32         = np.float32(MOTOR_TIME_STEP_INSIDE)
SQRT_CARGO_DT_IN        = np.float32(np.sqrt(CARGO_TIME_STEP_INSIDE))
SQRT_2_MOTOR_DIFF_IN    = np.float32(np.sqrt(2 * MOTOR_DIFFUSION * MOTOR_TIME_STEP_INSIDE))
CARGO_SPRING_IN         = np.float32(CARGO_K_HAT         * CARGO_TIME_STEP_INSIDE)
CARGO_SPRING_HALF_IN    = np.float32((CARGO_K_HAT / 2)   * CARGO_TIME_STEP_INSIDE)
DONOR_RADIAL_TERM_IN    = np.float32(DONOR_K_HAT    * CARGO_TIME_STEP_INSIDE)
ACCEPTOR_RADIAL_TERM_IN = np.float32(ACCEPTOR_K_HAT * CARGO_TIME_STEP_INSIDE)
# OPT 3: centrifugal pre-factor (2 × dt) avoids per-step multiply-by-2
TWO_DT_IN               = np.float32(2.0 * CARGO_TIME_STEP_INSIDE)
# OPT 3: dipole Cartesian noise amplitude = sqrt(2 × D_dip × dt)
SQRT_2_DIP_D_DT_IN      = np.float32(np.sqrt(2 * DONOR_DIPOLE_DIFF    * CARGO_TIME_STEP_INSIDE))
SQRT_2_DIP_A_DT_IN      = np.float32(np.sqrt(2 * ACCEPTOR_DIPOLE_DIFF * CARGO_TIME_STEP_INSIDE))

# ── OUTSIDE time-step dependent constants ────────────────────────────────────
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

# ── Ellipsoid / intensity ─────────────────────────────────────────────────────
ELLIPSOID_A_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_X**2)
ELLIPSOID_B_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_Y**2)
ELLIPSOID_C_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_Z**2)
ELLIPSOID_C_K_SQ        = np.float32((ELLIPSOID_SEMI_AXIS_Z * WAVE_VECTOR)**2)
MINUS_2_OVER_WAIST_SQ   = np.float32(-1 / (BEAM_WAIST**2))
DETECTION_THRESHOLD_F32 = np.float32(DETECTION_THRESHOLD)

# ── FRET ──────────────────────────────────────────────────────────────────────
FORSTER_RADIUS_F32 = np.float32(FORSTER_RADIUS)
I_EQ_F32           = np.float32(I_EQ)
DONOR_RATE_F32     = np.float32(DONOR_FLUORESCENCE_RATE)
ACCEPTOR_RATE_F32  = np.float32(ACCEPTOR_FLUORESCENCE_RATE)
RD_EQ_F32          = np.float32(DONOR_RADIUS_EQ)
RA_EQ_F32          = np.float32(ACCEPTOR_RADIUS_EQ)

# ── Detection probabilities ───────────────────────────────────────────────────
PROB_QD = np.float32(PROB_DONOR_NONRADIATIVE)
PROB_QA = np.float32(PROB_ACCEPTOR_NONRADIATIVE)
PROB_AS = np.float32(PROB_ACCEPTOR_NOT_DETECTED)
PROB_DS = np.float32(PROB_DONOR_NOT_DETECTED)
PROB_ND = np.float32(PROB_NO_DETECTION)

# ── Burst gap / confinement ───────────────────────────────────────────────────
BURST_MIN_GAP_F32 = np.float32(BURST_MIN_GAP)
MOTOR_CONF_X_F32  = np.float32(MOTOR_CONFINEMENT_X)
MOTOR_CONF_Y_F32  = np.float32(MOTOR_CONFINEMENT_Y)
MOTOR_CONF_Z_F32  = np.float32(MOTOR_CONFINEMENT_Z)


# =============================================================================
# JIT-COMPILED HELPER FUNCTIONS
# =============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def is_inside_volume(x, y, z, a_sq, b_sq, c_sq):
    return (x*x)/a_sq + (y*y)/b_sq + (z*z)/c_sq <= 1.0


@jit(nopython=True, fastmath=True, cache=True)
def calculate_intensity_fast(x, y, z, a_sq, b_sq, c_k_sq, minus_2_over_w_sq):
    exponent = minus_2_over_w_sq * ((x*x)/a_sq + (y*y)/b_sq + (z*z)/c_k_sq)
    return np.exp(exponent)


@jit(nopython=True, fastmath=True, cache=True)
def reflect(value, bound):
    """Fold value back into [-bound, +bound] via modular reflection."""
    span = np.float32(2.0) * bound
    v = value + bound
    v = v - span * np.floor(v / span)
    if v > bound:
        v = span - v
    return v - bound


@jit(nopython=True, fastmath=True, cache=True)
def wrap_2pi(angle):
    """Wrap angle into [0, 2π)."""
    two_pi = np.float32(6.2831853)
    if angle >= two_pi:
        angle -= two_pi * np.floor(angle / two_pi)
    elif angle < np.float32(0.0):
        angle += two_pi * (np.float32(1.0) + np.floor((-angle) / two_pi))
    return angle


@jit(nopython=True, fastmath=True, cache=True)
def simulate_photon_fast(delta_t, da_rate, donor_rate, acceptor_rate,
                         prob_qd, prob_qa, prob_as, prob_ds, prob_nd):
    """Exponential race photon emission within time window delta_t."""
    if np.random.random() < prob_nd:
        return np.int32(3), np.int32(0)
    if np.random.random() < prob_qd:
        return np.int32(3), np.int32(0)

    t_fret  = np.random.exponential(np.float32(1.0) / da_rate) \
              if da_rate > np.float32(0.0) else np.float32(1e9)
    t_donor = np.random.exponential(np.float32(1.0) / donor_rate)

    if t_fret < t_donor:
        if t_fret > delta_t:
            return np.int32(3), np.int32(0)
        if np.random.random() < prob_qa:
            return np.int32(3), np.int32(0)
        if t_fret + np.random.exponential(np.float32(1.0) / acceptor_rate) > delta_t:
            return np.int32(3), np.int32(0)
        if np.random.random() < prob_as:
            return np.int32(3), np.int32(0)
        return np.int32(5), np.int32(1)

    if t_donor > delta_t:
        return np.int32(3), np.int32(0)
    if np.random.random() < prob_ds:
        return np.int32(3), np.int32(0)
    return np.int32(4), np.int32(1)


# =============================================================================
# MAIN JIT SIMULATION FUNCTION
# =============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def run_simulation(
    max_steps, target_bursts, burst_min_gap,
    init_motor_x, init_motor_y, init_motor_z,
    # ── Translational noise (time-step independent) ──
    sqrt_2_donor_diff, sqrt_2_acceptor_diff,
    # ── Linker angular noise (time-step independent) ──
    sqrt_2_donor_theta, sqrt_2_acceptor_theta,
    theta_drift_k_d, theta_drift_k_a,          # OPT 3: pre-computed drift coeff
    donor_phi_diff, acceptor_phi_diff,
    # ── INSIDE constants ──
    cargo_dt_in, motor_dt_in, sqrt_cdt_in, sqrt_2_motor_in,
    cargo_spring_in, cargo_spring_half_in,
    donor_rad_in, acceptor_rad_in,
    two_dt_in,                                  # OPT 3: 2*cargo_dt_in
    sqrt_2_dip_d_dt_in, sqrt_2_dip_a_dt_in,    # OPT 3: dipole noise amplitudes
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
    conf_x, conf_y, conf_z):
    """
    Speed-optimised EPK simulation.

    Linker state : (r, theta, phi) spherical — but sin/cos cached per step so
                   the tan denominator and Cartesian conversion share one
                   evaluation.  Per dye: 4 trig calls (sin_th, cos_th,
                   sin_ph, cos_ph) instead of 6.

    Dipole state : (ddx, ddy, ddz) Cartesian unit vector, evolved as 3
                   independent OU components and normalised each inside step.
                   Isotropic rotational Brownian motion on S² with D_rot =
                   kT/(8π η r³).  Physically identical to the spherical Ito
                   SDE but requires only 1 sqrt + 3 divides instead of 6 trig
                   calls per dye.  Dipole updated only inside the volume.
    """
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

    # ── Motor + cargo ──────────────────────────────────────────────────────────
    motor_x = np.float32(init_motor_x)
    motor_y = np.float32(init_motor_y)
    motor_z = np.float32(init_motor_z)
    cargo_x = np.float32(init_motor_x)
    cargo_y = np.float32(init_motor_y)
    cargo_z = np.float32(init_motor_z)

    # ── Linker spherical state ─────────────────────────────────────────────────
    r_init_std_d = np.float32(np.sqrt(np.abs(np.float32(0.5) * sqrt_2_donor_diff    * sqrt_2_donor_diff)))
    r_init_std_a = np.float32(np.sqrt(np.abs(np.float32(0.5) * sqrt_2_acceptor_diff * sqrt_2_acceptor_diff)))
    r_d = np.float32(rd_eq) + r_init_std_d * np.float32(np.random.randn())
    r_a = np.float32(ra_eq) + r_init_std_a * np.float32(np.random.randn())
    if r_d < f32_1e_n6: r_d = f32_1e_n6
    if r_a < f32_1e_n6: r_a = f32_1e_n6

    th_d = np.float32(np.abs(np.random.normal(f32_0, np.float32(0.01))))
    th_a = np.float32(np.abs(np.random.normal(f32_0, np.float32(0.01))))
    two_pi = np.float32(6.2831853)
    ph_d = np.float32(np.random.random() * two_pi)
    ph_a = np.float32(np.random.random() * two_pi)

    # ── Dipole state: Cartesian unit vectors (OPT 2) ───────────────────────────
    # Initialise near (0, 0, 1) with small random perturbation
    ddx = np.float32(np.random.normal(f32_0, np.float32(0.1)))
    ddy = np.float32(np.random.normal(f32_0, np.float32(0.1)))
    ddz = np.float32(np.random.normal(f32_1, np.float32(0.1)))
    dn  = np.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
    if dn == f32_0: dn = f32_1
    ddx /= dn; ddy /= dn; ddz /= dn

    adx = np.float32(np.random.normal(f32_0, np.float32(0.1)))
    ady = np.float32(np.random.normal(f32_0, np.float32(0.1)))
    adz = np.float32(np.random.normal(f32_1, np.float32(0.1)))
    an  = np.sqrt(adx*adx + ady*ady + adz*adz)
    if an == f32_0: an = f32_1
    adx /= an; ady /= an; adz /= an

    # ── Timers / counters ──────────────────────────────────────────────────────
    current_time     = np.float64(0.0)
    last_motor_time  = np.float64(0.0)
    last_photon_time = np.float64(-1.0)

    total_photons = np.int64(0)
    total_bursts  = np.int64(0)
    event_count   = np.int64(0)
    cargo_step    = np.int64(0)
    in_volume_count = np.int64(0)
    total_steps     = np.int64(0)

    was_in_volume = False

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    while cargo_step < max_steps and total_bursts < target_bursts:

        # ── Select time-step regime ───────────────────────────────────────────
        if was_in_volume:
            cargo_dt   = cargo_dt_in
            motor_dt   = motor_dt_in
            sqrt_cdt   = sqrt_cdt_in
            sqrt2_m    = sqrt_2_motor_in
            c_spr      = cargo_spring_in
            c_spr_half = cargo_spring_half_in
            d_rad      = donor_rad_in
            a_rad      = acceptor_rad_in
            two_dt     = two_dt_in
            dip_d_amp  = sqrt_2_dip_d_dt_in   # OPT 3: pre-computed dipole noise amp
            dip_a_amp  = sqrt_2_dip_a_dt_in
        else:
            cargo_dt   = cargo_dt_out
            motor_dt   = motor_dt_out
            sqrt_cdt   = sqrt_cdt_out
            sqrt2_m    = sqrt_2_motor_out
            c_spr      = cargo_spring_out
            c_spr_half = cargo_spring_half_out
            d_rad      = donor_rad_out
            a_rad      = acceptor_rad_out
            two_dt     = two_dt_out
            dip_d_amp  = sqrt_2_dip_d_dt_out
            dip_a_amp  = sqrt_2_dip_a_dt_out

        # ── Motor (time-triggered + reflecting boundaries) ────────────────────
        if current_time - last_motor_time >= motor_dt:
            motor_x += sqrt2_m * np.random.randn()
            motor_y += sqrt2_m * np.random.randn()
            motor_z += sqrt2_m * np.random.randn()
            last_motor_time = current_time
            motor_x = reflect(motor_x, conf_x)
            motor_y = reflect(motor_y, conf_y)
            motor_z = reflect(motor_z, conf_z)

        # ── Cargo anchor (OU toward motor) ────────────────────────────────────
        noise_c = sqrt_2_donor_diff * sqrt_cdt
        cargo_x += c_spr      * (motor_x - cargo_x) + noise_c * np.random.randn()
        cargo_y += c_spr_half * (motor_y - cargo_y) + noise_c * np.random.randn()
        cargo_z += c_spr      * (motor_z - cargo_z) + noise_c * np.random.randn()

        # ── Linker radial dynamics ────────────────────────────────────────────
        if r_d < f32_1e_n6: r_d = f32_1e_n6
        if r_a < f32_1e_n6: r_a = f32_1e_n6

        r_d += d_rad * (rd_eq - r_d) + two_dt / r_d \
               + sqrt_2_donor_diff    * np.random.randn() * sqrt_cdt
        r_a += a_rad * (ra_eq - r_a) + two_dt / r_a \
               + sqrt_2_acceptor_diff * np.random.randn() * sqrt_cdt

        if r_d < f32_1e_n6: r_d = f32_1e_n6
        if r_a < f32_1e_n6: r_a = f32_1e_n6

        # ── Linker theta (OPT 1: cache sin/cos, derive tan = sin/cos) ─────────
        #
        # DONOR
        sin_th_d = np.sin(th_d)
        cos_th_d = np.cos(th_d)
        # tan with regularisation: sin/(cos+eps) avoids separate tan call
        tan_d    = sin_th_d / (cos_th_d + eps_tan)
        denom_d  = r_d * r_d * f32_2 * tan_d + f32_0pt01
        # theta_drift_k_d = 0.5 * DONOR_THETA_DIFF (pre-computed, OPT 3)
        drift_d  = -f32_2 * sin_th_d + theta_drift_k_d / denom_d
        th_d    += drift_d * cargo_dt \
                   + (sqrt_2_donor_theta / r_d) * np.random.randn() * sqrt_cdt
        # phi: reuse sin_th_d for denominator (sin(th + eps) ≈ sin(th))
        sin_phi_denom_d = sin_th_d if np.abs(sin_th_d) > eps_sin else eps_sin
        ph_d = wrap_2pi(ph_d + (donor_phi_diff / (r_d * r_d * sin_phi_denom_d))
                        * np.random.randn() * sqrt_cdt)

        # ACCEPTOR
        sin_th_a = np.sin(th_a)
        cos_th_a = np.cos(th_a)
        tan_a    = sin_th_a / (cos_th_a + eps_tan)
        denom_a  = r_a * r_a * f32_2 * tan_a + f32_0pt01
        drift_a  = -f32_2 * sin_th_a + theta_drift_k_a / denom_a
        th_a    += drift_a * cargo_dt \
                   + (sqrt_2_acceptor_theta / r_a) * np.random.randn() * sqrt_cdt
        sin_phi_denom_a = sin_th_a if np.abs(sin_th_a) > eps_sin else eps_sin
        ph_a = wrap_2pi(ph_a + (acceptor_phi_diff / (r_a * r_a * sin_phi_denom_a))
                        * np.random.randn() * sqrt_cdt)

        # ── Linker spherical → Cartesian (OPT 1: reuse cached sin/cos) ────────
        # sin_th_d / cos_th_d / sin(ph) / cos(ph) — only 2 NEW trig calls (ph)
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

        # ── Confocal volume check ──────────────────────────────────────────────
        donor_in    = is_inside_volume(donor_x,    donor_y,    donor_z,    a_sq, b_sq, c_sq)
        acceptor_in = is_inside_volume(acceptor_x, acceptor_y, acceptor_z, a_sq, b_sq, c_sq)
        total_steps += 1

        donor_intensity    = f32_0
        acceptor_intensity = f32_0

        if donor_in:
            donor_intensity = calculate_intensity_fast(
                donor_x, donor_y, donor_z, a_sq, b_sq, c_k_sq, minus_2_w_sq)
            donor_in = donor_intensity >= detection_threshold

        if acceptor_in:
            acceptor_intensity = calculate_intensity_fast(
                acceptor_x, acceptor_y, acceptor_z, a_sq, b_sq, c_k_sq, minus_2_w_sq)
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

            # ── Evolve dipole: Cartesian OU + normalise (OPT 2) ─────────────
            # Pure BM on each component (no restoring force) then normalise.
            # Gives uniform stationary distribution on S² — identical physics
            # to the spherical Ito SDE but with no trig operations.
            ddx += dip_d_amp * np.random.randn()
            ddy += dip_d_amp * np.random.randn()
            ddz += dip_d_amp * np.random.randn()
            dn   = np.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
            if dn == f32_0: dn = f32_1
            ddx /= dn;  ddy /= dn;  ddz /= dn

            adx += dip_a_amp * np.random.randn()
            ady += dip_a_amp * np.random.randn()
            adz += dip_a_amp * np.random.randn()
            an   = np.sqrt(adx*adx + ady*ady + adz*adz)
            if an == f32_0: an = f32_1
            adx /= an;  ady /= an;  adz /= an

            # ── kappa² from current dipole unit vectors ─────────────────────
            rx = ix / dist;  ry = iy / dist;  rz = iz / dist
            term1    = ddx*adx + ddy*ady + ddz*adz
            term2    = (ddx*rx + ddy*ry + ddz*rz) * (adx*rx + ady*ry + adz*rz)
            kappa_sq = (term1 - f32_3 * term2) ** np.float32(2)

            da_rate = donor_rate * f32_1pt5 * kappa_sq * (forster_r / dist) ** np.float32(6)

            # ── Photon emission ─────────────────────────────────────────────
            total_int = donor_intensity + acceptor_intensity
            if total_int > f32_0:
                photon_state, is_detected = simulate_photon_fast(
                    cargo_dt, da_rate, donor_rate, acceptor_rate,
                    prob_qd, prob_qa, prob_as, prob_ds, prob_nd
                )

                if is_detected == 1 and event_count < max_events:
                    if last_photon_time < 0.0 or \
                       (current_time - last_photon_time) >= burst_min_gap:
                        total_bursts += np.int64(1)
                    last_photon_time = current_time

                    event_times[event_count]        = np.float32(current_time)
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
# INITIALIZE AND RUN
# =============================================================================

print("Initializing simulation...")
print("Note: First run includes JIT compilation (~1-2 min)\n")

init_motor_x, init_motor_y, init_motor_z = 0.0, 0.0, 0.0
max_simulation_steps = 10_000_000_000

print(f"Maximum steps (safety cap): {max_simulation_steps:,}")
print("Starting simulation...\n")
start_time = time.time()

results = run_simulation(
    max_simulation_steps, TARGET_BURSTS, BURST_MIN_GAP_F32,
    init_motor_x, init_motor_y, init_motor_z,
    # Translational noise
    SQRT_2_DONOR_DIFF, SQRT_2_ACCEPTOR_DIFF,
    # Linker angular noise + pre-computed drift coeff
    SQRT_2_DONOR_THETA_DIFF, SQRT_2_ACCEPTOR_THETA_DIFF,
    THETA_DRIFT_K_D, THETA_DRIFT_K_A,
    DONOR_PHI_DIFF_F32, ACCEPTOR_PHI_DIFF_F32,
    # INSIDE
    CARGO_DT_IN_F32,  MOTOR_DT_IN_F32,  SQRT_CARGO_DT_IN,  SQRT_2_MOTOR_DIFF_IN,
    CARGO_SPRING_IN,  CARGO_SPRING_HALF_IN,
    DONOR_RADIAL_TERM_IN, ACCEPTOR_RADIAL_TERM_IN,
    TWO_DT_IN, SQRT_2_DIP_D_DT_IN, SQRT_2_DIP_A_DT_IN,
    # OUTSIDE
    CARGO_DT_OUT_F32, MOTOR_DT_OUT_F32, SQRT_CARGO_DT_OUT, SQRT_2_MOTOR_DIFF_OUT,
    CARGO_SPRING_OUT, CARGO_SPRING_HALF_OUT,
    DONOR_RADIAL_TERM_OUT, ACCEPTOR_RADIAL_TERM_OUT,
    TWO_DT_OUT, SQRT_2_DIP_D_DT_OUT, SQRT_2_DIP_A_DT_OUT,
    # Equilibria
    RD_EQ_F32, RA_EQ_F32, I_EQ_F32,
    # Volume
    ELLIPSOID_A_SQ, ELLIPSOID_B_SQ, ELLIPSOID_C_SQ,
    ELLIPSOID_C_K_SQ, MINUS_2_OVER_WAIST_SQ, DETECTION_THRESHOLD_F32,
    # FRET
    FORSTER_RADIUS_F32, DONOR_RATE_F32, ACCEPTOR_RATE_F32,
    # Probabilities
    PROB_QD, PROB_QA, PROB_AS, PROB_DS, PROB_ND,
    # Confinement
    MOTOR_CONF_X_F32, MOTOR_CONF_Y_F32, MOTOR_CONF_Z_F32
)

elapsed_time = time.time() - start_time

# =============================================================================
# UNPACK AND TRIM RESULTS
# =============================================================================

(num_cargo_steps, total_photons, total_bursts_collected, num_events, occupancy,
 total_sim_time_f32,
 event_times, event_colors,
 event_donor_x,    event_donor_y,    event_donor_z,
 event_acceptor_x, event_acceptor_y, event_acceptor_z,
 event_interdye_dist, event_fret_eff,
 event_donor_intensity, event_acceptor_intensity) = results

n = num_events
event_times          = event_times[:n]
event_colors         = event_colors[:n]
event_donor_x        = event_donor_x[:n]
event_donor_y        = event_donor_y[:n]
event_donor_z        = event_donor_z[:n]
event_acceptor_x     = event_acceptor_x[:n]
event_acceptor_y     = event_acceptor_y[:n]
event_acceptor_z     = event_acceptor_z[:n]
event_interdye_dist  = event_interdye_dist[:n]
event_fret_eff       = event_fret_eff[:n]
event_donor_intensity    = event_donor_intensity[:n]
event_acceptor_intensity = event_acceptor_intensity[:n]

total_simulation_time = float(total_sim_time_f32)

print(f"\n{'='*70}")
print("SIMULATION COMPLETE!")
print(f"{'='*70}")
print(f"Execution time:             {elapsed_time:.1f} s  ({elapsed_time/60:.1f} min)")
print(f"Total bursts collected:     {total_bursts_collected:,}  (target: {TARGET_BURSTS:,})")
print(f"Total photons collected:    {total_photons:,}")
print(f"Simulated physical time:    {total_simulation_time:.4f} s")
print(f"Cargo steps taken:          {num_cargo_steps:,}")
print(f"Confocal volume occupancy:  {occupancy*100:.3f}%")
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
event_df.to_csv('photon_events_nanosecond.csv', index=False)
print(f"  Saved {len(event_df):,} photon events to 'photon_events_nanosecond.csv'")

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
final_burst_df.to_csv('photon_bursts_nanosecond.csv', index=False)
print(f"  Saved burst analysis to 'photon_bursts_nanosecond.csv'\n")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"{'='*70}")
print("EPK SPEED-OPTIMIZED SIMULATION COMPLETE!")
print(f"{'='*70}")
print(f"  Execution time:   {elapsed_time:.1f} s ({elapsed_time/60:.1f} min)")
print(f"  Steps/second:     {num_cargo_steps/elapsed_time:,.0f}")
print(f"  Photons/second:   {total_photons/elapsed_time:,.0f}")
print(f"  Bursts/second:    {total_bursts_collected/elapsed_time:,.1f}")
print(f"  Memory (events):  {n*12*4/1e6:.1f} MB  "
      f"(vs {num_cargo_steps*13*4/1e6:.1f} MB if storing all steps)")
print(f"\nFiles saved:")
print(f"  - photon_events_nanosecond.csv  ({len(event_df):,} events)")
print(f"  - photon_bursts_nanosecond.csv  ({len(all_bursts):,} bursts)")
print(f"{'='*70}\n")
