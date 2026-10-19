"""
FRET Simulation - ISOTROPIC DYNAMICS + ULTRA-OPTIMIZED NANOSECOND VERSION
==========================================================================

Combines:
  - Isotropic dye dynamics: radial OU + Cartesian orientation-vector OU
  - Per-step kappa^2 resampling from random dipole orientations
  - CTMC photon emission (continuous-time Markov chain)
  - Cargo anchor following motor via OU process
  - Adaptive time stepping (coarse outside confocal volume, fine inside)
  - Motor confinement via reflecting boundary conditions
  - Burst-based termination
  - Float32 precision, event-only storage, Numba JIT
"""

import numpy as np
import pandas as pd
from numba import jit
import time


# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

kT = 4.1e-12  # pN*microns (thermal energy at room temperature)

# Dye hydrodynamic radii (nm) — isotropic model values
DONOR_RADIUS    = 0.3
ACCEPTOR_RADIUS = 0.3

VISCOSITY = 8.9e-12  # kg/(s*nm)

# Friction coefficients (Stokes drag)
DONOR_FRICTION    = 6 * np.pi * DONOR_RADIUS    * VISCOSITY
ACCEPTOR_FRICTION = 6 * np.pi * ACCEPTOR_RADIUS * VISCOSITY

# Dye diffusion coefficients (Einstein relation: kT/gamma)
DONOR_DIFFUSION    = kT / DONOR_FRICTION    # nm^2/s
ACCEPTOR_DIFFUSION = kT / ACCEPTOR_FRICTION # nm^2/s

# Motor diffusion: 100 um^2/s = 100*(1000 nm)^2/s = 1e8 nm^2/s
MOTOR_DIFFUSION = 1e8  # nm^2/s

# ---- Dye linker spring constants ----
DONOR_NUM_LINKS    = 7
ACCEPTOR_NUM_LINKS = 7
DONOR_SPRING_CONSTANT    = (1010 / DONOR_NUM_LINKS)    * 1e-9  # N/nm
ACCEPTOR_SPRING_CONSTANT = (1010 / ACCEPTOR_NUM_LINKS) * 1e-9  # N/nm

DONOR_K_HAT    = DONOR_SPRING_CONSTANT    / DONOR_FRICTION
ACCEPTOR_K_HAT = ACCEPTOR_SPRING_CONSTANT / ACCEPTOR_FRICTION

# Radial equilibrium lengths (nm)
DONOR_RADIUS_EQ    = 0.08 * DONOR_NUM_LINKS    # 0.56 nm
ACCEPTOR_RADIUS_EQ = 0.08 * ACCEPTOR_NUM_LINKS # 0.56 nm

# ---- Orientation vector spring / diffusion scale factors ----
# Orientation components (d1,d2,d3) are OU processes.
# WD scales diffusion; WS scales spring toward (0,0,1).
WD = 1.0
WS = 1.0
DONOR_ORIENTATION_DIFF = WD * DONOR_DIFFUSION
ACCEPTOR_ORIENTATION_DIFF = WD * ACCEPTOR_DIFFUSION
DONOR_ORIENTATION_K    = WS * DONOR_K_HAT
ACCEPTOR_ORIENTATION_K = WS * ACCEPTOR_K_HAT

# ---- Cargo anchor spring (anchor follows motor) ----
CARGO_SPRING_CONSTANT = (1010 / 12) * 1e-9  # N/nm
CARGO_K_HAT           = CARGO_SPRING_CONSTANT / DONOR_FRICTION

# ---- Fluorescence rates ----
DONOR_FLUORESCENCE_RATE    = 1 / (4e-5)  # s^-1
ACCEPTOR_FLUORESCENCE_RATE = 1 / (4e-5)  # s^-1

# ---- FRET parameters ----
FORSTER_RADIUS = 5.0   # nm
I_EQ           = 0.7 * FORSTER_RADIUS  # nm -- x-offset between dye attachment points on cargo

# ---- Detection probabilities ----
PROB_DONOR_NONRADIATIVE    = 0.15
PROB_ACCEPTOR_NONRADIATIVE = 0.21
PROB_ACCEPTOR_NOT_DETECTED = 0.25
PROB_DONOR_NOT_DETECTED    = 0.15
PROB_NO_DETECTION          = 0.975


# =============================================================================
# SIMULATION TIME STEP PARAMETERS
# =============================================================================
# INSIDE  confocal volume: fine steps for accurate photon statistics
# OUTSIDE confocal volume: coarse steps (no photons recorded)

MOTOR_TIME_STEP_INSIDE  = 10e-6    # 10 us
CARGO_TIME_STEP_INSIDE  = 100e-9   # 100 ns

MOTOR_TIME_STEP_OUTSIDE = 1000e-6  # 1 ms
CARGO_TIME_STEP_OUTSIDE = 10e-6    # 10 us


# =============================================================================
# BURST TARGET
# =============================================================================

TARGET_BURSTS = 10000
BURST_MIN_GAP = 0.0  # seconds — gap >= this opens a new burst


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
# MOTOR CONFINEMENT PARAMETERS
# =============================================================================
# Reflecting box around confocal ellipsoid.
# Box half-widths = CONFINEMENT_FACTOR * ellipsoid semi-axes.

CONFINEMENT_FACTOR  = 4
MOTOR_CONFINEMENT_X = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_X  # ±1000 nm
MOTOR_CONFINEMENT_Y = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Y  # ±1000 nm
MOTOR_CONFINEMENT_Z = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Z  # ±4000 nm


# =============================================================================
# DISPLAY SETUP
# =============================================================================

print("\n" + "="*70)
print("FRET SIMULATION - ISOTROPIC DYNAMICS + NANOSECOND OPTIMIZATION")
print("="*70)
print(f"Target bursts:          {TARGET_BURSTS:,}")
print(f"Burst gap threshold:    {BURST_MIN_GAP*1e3:.1f} ms")
print(f"Motor diffusion:        {MOTOR_DIFFUSION:.2e} nm^2/s  ({MOTOR_DIFFUSION/1e6:.0f} um^2/s)")
print()
print(f"Motor confinement box (reflecting boundaries):")
print(f"  X: +/-{MOTOR_CONFINEMENT_X:.0f} nm  ({CONFINEMENT_FACTOR}x ellipsoid)")
print(f"  Y: +/-{MOTOR_CONFINEMENT_Y:.0f} nm  ({CONFINEMENT_FACTOR}x ellipsoid)")
print(f"  Z: +/-{MOTOR_CONFINEMENT_Z:.0f} nm  ({CONFINEMENT_FACTOR}x ellipsoid)")
print()
print(f"  Time step          INSIDE volume        OUTSIDE volume")
print(f"  ---------------------------------------------------------")
print(f"  Motor (coarse)     {MOTOR_TIME_STEP_INSIDE*1e6:7.1f} us            {MOTOR_TIME_STEP_OUTSIDE*1e3:7.1f} ms")
print(f"  Cargo (Langevin)   {CARGO_TIME_STEP_INSIDE*1e9:7.1f} ns            {CARGO_TIME_STEP_OUTSIDE*1e6:7.1f} us")
print()
print(f"Isotropic Dynamics:")
print(f"  Forster radius:     {FORSTER_RADIUS} nm")
print(f"  i_eq offset:        {I_EQ:.2f} nm")
print(f"  Donor eq. radius:   {DONOR_RADIUS_EQ:.3f} nm ({DONOR_NUM_LINKS} links)")
print(f"  Acceptor eq. radius:{ACCEPTOR_RADIUS_EQ:.3f} nm ({ACCEPTOR_NUM_LINKS} links)")
print(f"  Donor lifetime:     {1/DONOR_FLUORESCENCE_RATE*1e6:.1f} us")
print(f"  kappa^2:            resampled isotropically each step")
print()
print("Optimizations: Float32 | Adaptive dt | Confinement | Event-only | JIT")
print("="*70 + "\n")


# =============================================================================
# PRE-COMPUTE CONSTANTS (FLOAT32)
# =============================================================================

# ---- Noise amplitudes (time-step independent) ----
SQRT_2_DONOR_DIFF         = np.float32(np.sqrt(2 * DONOR_DIFFUSION))
SQRT_2_ACCEPTOR_DIFF      = np.float32(np.sqrt(2 * ACCEPTOR_DIFFUSION))
SQRT_2_DONOR_ORIENT_DIFF  = np.float32(np.sqrt(2 * DONOR_ORIENTATION_DIFF))
SQRT_2_ACCEPTOR_ORIENT_DIFF = np.float32(np.sqrt(2 * ACCEPTOR_ORIENTATION_DIFF))

# ---- INSIDE volume time-step dependent constants ----
CARGO_DT_IN_F32            = np.float32(CARGO_TIME_STEP_INSIDE)
MOTOR_DT_IN_F32            = np.float32(MOTOR_TIME_STEP_INSIDE)
SQRT_CARGO_DT_IN           = np.float32(np.sqrt(CARGO_TIME_STEP_INSIDE))
SQRT_2_MOTOR_DIFF_IN       = np.float32(np.sqrt(2 * MOTOR_DIFFUSION * MOTOR_TIME_STEP_INSIDE))
# Cargo anchor OU spring terms
CARGO_SPRING_IN            = np.float32(CARGO_K_HAT         * CARGO_TIME_STEP_INSIDE)
CARGO_SPRING_HALF_IN       = np.float32((CARGO_K_HAT / 2)   * CARGO_TIME_STEP_INSIDE)
# Dye radial OU spring terms
DONOR_RADIAL_TERM_IN       = np.float32(DONOR_K_HAT         * CARGO_TIME_STEP_INSIDE)
ACCEPTOR_RADIAL_TERM_IN    = np.float32(ACCEPTOR_K_HAT      * CARGO_TIME_STEP_INSIDE)
# Orientation vector OU spring terms
DONOR_ORIENT_TERM_IN       = np.float32(DONOR_ORIENTATION_K    * CARGO_TIME_STEP_INSIDE)
ACCEPTOR_ORIENT_TERM_IN    = np.float32(ACCEPTOR_ORIENTATION_K * CARGO_TIME_STEP_INSIDE)

# ---- OUTSIDE volume time-step dependent constants ----
CARGO_DT_OUT_F32           = np.float32(CARGO_TIME_STEP_OUTSIDE)
MOTOR_DT_OUT_F32           = np.float32(MOTOR_TIME_STEP_OUTSIDE)
SQRT_CARGO_DT_OUT          = np.float32(np.sqrt(CARGO_TIME_STEP_OUTSIDE))
SQRT_2_MOTOR_DIFF_OUT      = np.float32(np.sqrt(2 * MOTOR_DIFFUSION * MOTOR_TIME_STEP_OUTSIDE))
CARGO_SPRING_OUT           = np.float32(CARGO_K_HAT         * CARGO_TIME_STEP_OUTSIDE)
CARGO_SPRING_HALF_OUT      = np.float32((CARGO_K_HAT / 2)   * CARGO_TIME_STEP_OUTSIDE)
DONOR_RADIAL_TERM_OUT      = np.float32(DONOR_K_HAT         * CARGO_TIME_STEP_OUTSIDE)
ACCEPTOR_RADIAL_TERM_OUT   = np.float32(ACCEPTOR_K_HAT      * CARGO_TIME_STEP_OUTSIDE)
DONOR_ORIENT_TERM_OUT      = np.float32(DONOR_ORIENTATION_K    * CARGO_TIME_STEP_OUTSIDE)
ACCEPTOR_ORIENT_TERM_OUT   = np.float32(ACCEPTOR_ORIENTATION_K * CARGO_TIME_STEP_OUTSIDE)

# ---- Ellipsoid / intensity ----
ELLIPSOID_A_SQ             = np.float32(ELLIPSOID_SEMI_AXIS_X**2)
ELLIPSOID_B_SQ             = np.float32(ELLIPSOID_SEMI_AXIS_Y**2)
ELLIPSOID_C_SQ             = np.float32(ELLIPSOID_SEMI_AXIS_Z**2)
ELLIPSOID_C_K_SQ           = np.float32((ELLIPSOID_SEMI_AXIS_Z * WAVE_VECTOR)**2)
MINUS_2_OVER_WAIST_SQ      = np.float32(-1 / (BEAM_WAIST**2))
DETECTION_THRESHOLD_F32    = np.float32(DETECTION_THRESHOLD)

# ---- FRET ----
FORSTER_RADIUS_6           = np.float32(FORSTER_RADIUS**6)
FORSTER_RADIUS_F32         = np.float32(FORSTER_RADIUS)
I_EQ_F32                   = np.float32(I_EQ)
DONOR_RATE_F32             = np.float32(DONOR_FLUORESCENCE_RATE)
ACCEPTOR_RATE_F32          = np.float32(ACCEPTOR_FLUORESCENCE_RATE)
RD_EQ_F32                  = np.float32(DONOR_RADIUS_EQ)
RA_EQ_F32                  = np.float32(ACCEPTOR_RADIUS_EQ)

# ---- Detection probabilities ----
PROB_QD = np.float32(PROB_DONOR_NONRADIATIVE)
PROB_QA = np.float32(PROB_ACCEPTOR_NONRADIATIVE)
PROB_AS = np.float32(PROB_ACCEPTOR_NOT_DETECTED)
PROB_DS = np.float32(PROB_DONOR_NOT_DETECTED)
PROB_ND = np.float32(PROB_NO_DETECTION)

# ---- Burst gap ----
BURST_MIN_GAP_F32 = np.float32(BURST_MIN_GAP)

# ---- Motor confinement ----
MOTOR_CONF_X_F32 = np.float32(MOTOR_CONFINEMENT_X)
MOTOR_CONF_Y_F32 = np.float32(MOTOR_CONFINEMENT_Y)
MOTOR_CONF_Z_F32 = np.float32(MOTOR_CONFINEMENT_Z)


# =============================================================================
# JIT-COMPILED HELPER FUNCTIONS
# =============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def is_inside_volume(x, y, z, a_sq, b_sq, c_sq):
    """True if (x,y,z) is inside the ellipsoid."""
    return (x*x)/a_sq + (y*y)/b_sq + (z*z)/c_sq <= 1.0


@jit(nopython=True, fastmath=True, cache=True)
def calculate_intensity_fast(x, y, z, a_sq, b_sq, c_k_sq, minus_2_over_w_sq):
    """Gaussian beam intensity at (x,y,z). Assumes already inside ellipsoid."""
    exponent = minus_2_over_w_sq * ((x*x)/a_sq + (y*y)/b_sq + (z*z)/c_k_sq)
    return np.exp(exponent)


@jit(nopython=True, fastmath=True, cache=True)
def reflect(value, bound):
    """
    Reflecting boundary: fold value back into [-bound, +bound].
    Handles arbitrarily large steps via modular folding.
    """
    span = np.float32(2.0) * bound
    v = value + bound                    # shift to [0, 2*bound]
    v = v - span * np.floor(v / span)    # wrap
    if v > bound:                        # reflect upper half
        v = span - v
    return v - bound


@jit(nopython=True, fastmath=True, cache=True)
def simulate_photon_ctmc(delta_t, da_rate, donor_rate, acceptor_rate,
                         prob_qd, prob_qa, prob_as, prob_ds, prob_nd):
    """
    CTMC photon emission simulation.

    Simulates one excitation-emission cycle within a window of length delta_t.
    The donor starts excited (state=1). Competing exponential clocks determine
    whether FRET (->state 2) or donor emission (->state 4) wins. Acceptor
    emission is state 5. No-photon outcomes are state 3.

    Returns: (photon_state, is_detected)
      state 4 = donor photon   (is_detected=1 if not lost)
      state 5 = acceptor photon (is_detected=1 if not lost)
      state 3 = no photon      (is_detected=0)
    """
    state = np.int32(1)   # donor excited
    t     = np.float32(0.0)
    nd    = np.int32(0)   # no-detection flag (set once, shared across states)

    while t <= delta_t and state != 3 and state != 4 and state != 5:

        # ---- Donor excited (state 1) ----------------------------------------
        while True:
            if state == np.int32(1):
                kda  = np.random.exponential(np.float32(1.0) / da_rate)
                kd   = np.random.exponential(np.float32(1.0) / donor_rate)
                d_nr = np.int32(1) if np.random.random() < prob_qd else np.int32(0)
                nd   = np.int32(1) if np.random.random() < prob_nd else np.int32(0)

                if kda <= kd and d_nr == 0 and nd == 0:
                    t    += kda
                    state = np.int32(2)   # FRET -> acceptor excited
                elif kd < kda and d_nr == 0 and nd == 0:
                    t    += kd
                    state = np.int32(4)   # donor emission
                else:
                    state = np.int32(3)
                    t     = np.float32(0.0)
                    break

            if t > delta_t:
                break

            # ---- Acceptor excited (state 2) ----------------------------------
            if state == np.int32(2):
                ka   = np.random.exponential(np.float32(1.0) / acceptor_rate)
                a_nr = np.int32(1) if np.random.random() < prob_qa else np.int32(0)

                if a_nr == 0 and nd == 0:
                    t    += ka
                    state = np.int32(5)   # acceptor emission
                elif (t + ka) > delta_t:
                    break                 # emission would fall outside window
                else:
                    state = np.int32(3)
                    t     = np.float32(0.0)
                    break

            # ---- Donor emitted (state 4): check optical detection ------------
            if state == np.int32(4):
                if np.random.random() < prob_as:   # lost in optics
                    state = np.int32(3)
                    t     = np.float32(0.0)
                break

            # ---- Acceptor emitted (state 5): check optical detection ---------
            if state == np.int32(5):
                if np.random.random() < prob_ds:   # lost in optics
                    state = np.int32(3)
                    t     = np.float32(0.0)
                break

        if t > delta_t:
            break

    detected = np.int32(1) if (state == np.int32(4) or state == np.int32(5)) else np.int32(0)
    return state, detected


# =============================================================================
# MAIN JIT SIMULATION FUNCTION
# =============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def run_simulation(
    max_steps, target_bursts, burst_min_gap,
    # --- Initial positions ---
    init_motor_x, init_motor_y, init_motor_z,
    # --- Noise amplitudes (time-step independent) ---
    sqrt_2_donor_diff, sqrt_2_acceptor_diff,
    sqrt_2_donor_orient_diff, sqrt_2_acceptor_orient_diff,
    # --- INSIDE time-step dependent constants ---
    cargo_dt_in, motor_dt_in, sqrt_cdt_in, sqrt_2_motor_diff_in,
    cargo_spring_in, cargo_spring_half_in,
    donor_radial_term_in, acceptor_radial_term_in,
    donor_orient_term_in, acceptor_orient_term_in,
    # --- OUTSIDE time-step dependent constants ---
    cargo_dt_out, motor_dt_out, sqrt_cdt_out, sqrt_2_motor_diff_out,
    cargo_spring_out, cargo_spring_half_out,
    donor_radial_term_out, acceptor_radial_term_out,
    donor_orient_term_out, acceptor_orient_term_out,
    # --- Dye equilibrium radii ---
    rd_eq, ra_eq, i_eq,
    # --- Volume / intensity ---
    a_sq, b_sq, c_sq, c_k_sq, minus_2_w_sq, detection_threshold,
    # --- FRET ---
    forster_r, forster_r6, donor_rate, acceptor_rate,
    # --- Probabilities ---
    prob_qd, prob_qa, prob_as, prob_ds, prob_nd,
    # --- Motor confinement box half-widths ---
    conf_x, conf_y, conf_z):
    """
    Main simulation loop combining:
      - Isotropic dye dynamics: radial OU + Cartesian orientation-vector OU
      - Per-step kappa^2 resampling from isotropically distributed dipoles
      - CTMC photon emission within each cargo timestep window
      - Cargo anchor following motor via OU process
      - Adaptive time stepping (coarse outside, fine inside confocal volume)
      - Motor confinement via reflecting boundary conditions
      - Burst-based termination

    Dye state per dye:
      r       — scalar radius from cargo anchor (OU toward equilibrium)
      (n1,n2,n3) — orientation unit vector (OU toward z-axis, then normalised)
    Absolute dye position:
      donor:    cargo + r_d*(dn1,dn2,dn3)  +  (i_eq, 0, 0)
      acceptor: cargo + r_a*(an1,an2,an3)

    kappa^2 is re-drawn each step from random unit dipoles, making the
    orientation factor isotropic on the timescale of the cargo step.
    """
    max_events = target_bursts * 200  # ~200 photons/burst buffer

    # --- Event storage ---
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

    # --- Scalar state: motor + cargo anchor ---
    motor_x = np.float32(init_motor_x)
    motor_y = np.float32(init_motor_y)
    motor_z = np.float32(init_motor_z)

    cargo_x = np.float32(init_motor_x)
    cargo_y = np.float32(init_motor_y)
    cargo_z = np.float32(init_motor_z)

    # --- Dye radii (OU scalar) ---
    r_d = np.float32(rd_eq)
    r_a = np.float32(ra_eq)

    # --- Dye orientation vectors (OU toward z-axis unit vector) ---
    # Initialise near (0, 0, 1) with small perturbation
    orient_init_std = np.float32(np.sqrt(np.abs(sqrt_2_donor_orient_diff
                                                 * sqrt_2_donor_orient_diff * 0.5)))
    d1 = np.float32(np.random.normal(0.0, orient_init_std))
    d2 = np.float32(np.random.normal(0.0, orient_init_std))
    d3 = np.float32(np.random.normal(1.0, orient_init_std))

    orient_init_std_a = np.float32(np.sqrt(np.abs(sqrt_2_acceptor_orient_diff
                                                    * sqrt_2_acceptor_orient_diff * 0.5)))
    a1 = np.float32(np.random.normal(0.0, orient_init_std_a))
    a2 = np.float32(np.random.normal(0.0, orient_init_std_a))
    a3 = np.float32(np.random.normal(1.0, orient_init_std_a))

    # --- Timers and counters ---
    current_time     = np.float64(0.0)
    last_motor_time  = np.float64(0.0)
    last_photon_time = np.float64(-1.0)   # -1 = no photon yet

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
            cargo_dt     = cargo_dt_in
            motor_dt     = motor_dt_in
            sqrt_cdt     = sqrt_cdt_in
            sqrt2_m      = sqrt_2_motor_diff_in
            c_spr        = cargo_spring_in
            c_spr_half   = cargo_spring_half_in
            d_rad        = donor_radial_term_in
            a_rad        = acceptor_radial_term_in
            d_ori        = donor_orient_term_in
            a_ori        = acceptor_orient_term_in
        else:
            cargo_dt     = cargo_dt_out
            motor_dt     = motor_dt_out
            sqrt_cdt     = sqrt_cdt_out
            sqrt2_m      = sqrt_2_motor_diff_out
            c_spr        = cargo_spring_out
            c_spr_half   = cargo_spring_half_out
            d_rad        = donor_radial_term_out
            a_rad        = acceptor_radial_term_out
            d_ori        = donor_orient_term_out
            a_ori        = acceptor_orient_term_out

        # ── Update motor (time-based trigger + reflecting boundaries) ─────────
        if current_time - last_motor_time >= motor_dt:
            motor_x += sqrt2_m * np.random.randn()
            motor_y += sqrt2_m * np.random.randn()
            motor_z += sqrt2_m * np.random.randn()
            last_motor_time = current_time
            motor_x = reflect(motor_x, conf_x)
            motor_y = reflect(motor_y, conf_y)
            motor_z = reflect(motor_z, conf_z)

        # ── Update cargo anchor (OU toward motor) ─────────────────────────────
        noise_c = sqrt_2_donor_diff * sqrt_cdt
        cargo_x += c_spr      * (motor_x - cargo_x) + noise_c * np.random.randn()
        cargo_y += c_spr_half * (motor_y - cargo_y) + noise_c * np.random.randn()
        cargo_z += c_spr      * (motor_z - cargo_z) + noise_c * np.random.randn()

        # ── Update dye RADII (scalar OU toward equilibrium) ───────────────────
        r_d += d_rad * (rd_eq - r_d) + sqrt_2_donor_diff    * np.random.randn() * sqrt_cdt
        r_a += a_rad * (ra_eq - r_a) + sqrt_2_acceptor_diff * np.random.randn() * sqrt_cdt

        # ── Update dye ORIENTATION VECTORS (OU toward z-axis (0,0,1)) ─────────
        # Donor
        d1 += d_ori * (np.float32(0.0) - d1) + sqrt_2_donor_diff * np.random.randn() * sqrt_cdt
        d2 += d_ori * (np.float32(0.0) - d2) + sqrt_2_donor_diff * np.random.randn() * sqrt_cdt
        d3 += d_ori * (np.float32(1.0) - d3) + sqrt_2_donor_diff * np.random.randn() * sqrt_cdt
        # Acceptor
        a1 += a_ori * (np.float32(0.0) - a1) + sqrt_2_acceptor_diff * np.random.randn() * sqrt_cdt
        a2 += a_ori * (np.float32(0.0) - a2) + sqrt_2_acceptor_diff * np.random.randn() * sqrt_cdt
        a3 += a_ori * (np.float32(1.0) - a3) + sqrt_2_acceptor_diff * np.random.randn() * sqrt_cdt

        # Normalise orientation vectors to unit length
        dn = np.sqrt(d1*d1 + d2*d2 + d3*d3)
        if dn == np.float32(0.0):
            dn = np.float32(1.0)
        an = np.sqrt(a1*a1 + a2*a2 + a3*a3)
        if an == np.float32(0.0):
            an = np.float32(1.0)
        ds1 = d1 / dn;  ds2 = d2 / dn;  ds3 = d3 / dn
        as1 = a1 / an;  as2 = a2 / an;  as3 = a3 / an

        # ── Absolute dye positions ────────────────────────────────────────────
        # i_eq shifts the donor attachment point along x (structural offset)
        donor_x    = cargo_x + r_d * ds1 + i_eq
        donor_y    = cargo_y + r_d * ds2
        donor_z    = cargo_z + r_d * ds3

        acceptor_x = cargo_x + r_a * as1
        acceptor_y = cargo_y + r_a * as2
        acceptor_z = cargo_z + r_a * as3

        # ── Confocal volume check ─────────────────────────────────────────────
        donor_in    = is_inside_volume(donor_x,    donor_y,    donor_z,    a_sq, b_sq, c_sq)
        acceptor_in = is_inside_volume(acceptor_x, acceptor_y, acceptor_z, a_sq, b_sq, c_sq)

        total_steps += 1

        donor_intensity    = np.float32(0.0)
        acceptor_intensity = np.float32(0.0)

        if donor_in:
            donor_intensity = calculate_intensity_fast(
                donor_x, donor_y, donor_z, a_sq, b_sq, c_k_sq, minus_2_w_sq)
            donor_in = donor_intensity >= detection_threshold

        if acceptor_in:
            acceptor_intensity = calculate_intensity_fast(
                acceptor_x, acceptor_y, acceptor_z, a_sq, b_sq, c_k_sq, minus_2_w_sq)
            acceptor_in = acceptor_intensity >= detection_threshold

        currently_in_volume = donor_in or acceptor_in

        # ── Photon simulation (only inside volume) ────────────────────────────
        if currently_in_volume:
            in_volume_count += 1

            # Inter-dye distance
            ddx = donor_x - acceptor_x
            ddy = donor_y - acceptor_y
            ddz = donor_z - acceptor_z
            dist_sq = ddx*ddx + ddy*ddy + ddz*ddz
            if dist_sq < np.float32(1e-12):
                dist_sq = np.float32(1e-12)
            dist   = np.sqrt(dist_sq)
            dist_6 = dist_sq * dist_sq * dist_sq

            fret_eff = np.float32(1.0) / (np.float32(1.0) + dist_6 / forster_r6)

            # ── Isotropic kappa^2: resample dipole orientations each step ──
            # Draw random unit vectors for donor and acceptor transition dipoles
            dvx = np.float32(np.random.randn())
            dvy = np.float32(np.random.randn())
            dvz = np.float32(np.random.randn())
            avx = np.float32(np.random.randn())
            avy = np.float32(np.random.randn())
            avz = np.float32(np.random.randn())

            dnorm = np.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)
            anorm = np.sqrt(avx*avx + avy*avy + avz*avz)
            if dnorm == np.float32(0.0):
                dnorm = np.float32(1.0)
            if anorm == np.float32(0.0):
                anorm = np.float32(1.0)
            dvx /= dnorm;  dvy /= dnorm;  dvz /= dnorm
            avx /= anorm;  avy /= anorm;  avz /= anorm

            # Unit inter-dye vector
            rx = ddx / dist;  ry = ddy / dist;  rz = ddz / dist

            # kappa^2 = (muD . muA  -  3*(muD . r)*(muA . r))^2
            dot_da  = dvx*avx + dvy*avy + dvz*avz
            dot_dr  = dvx*rx  + dvy*ry  + dvz*rz
            dot_ar  = avx*rx  + avy*ry  + avz*rz
            kappa2  = (dot_da - np.float32(3.0) * dot_dr * dot_ar) ** 2

            # FRET rate using isotropic kappa^2
            da_rate = donor_rate * np.float32(1.5) * kappa2 * (forster_r / dist) ** np.float32(6)

            # ── CTMC photon emission ──────────────────────────────────────
            total_int = donor_intensity + acceptor_intensity
            if total_int > np.float32(0.0):
                photon_state, is_detected = simulate_photon_ctmc(
                    cargo_dt, da_rate, donor_rate, acceptor_rate,
                    prob_qd, prob_qa, prob_as, prob_ds, prob_nd
                )

                if is_detected == 1 and event_count < max_events:
                    # New burst if first photon or gap >= threshold
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
print("Note: First run includes JIT compilation time (~1-2 min)\n")

# Motor starts at origin (centre of confinement box / confocal volume)
init_motor_x = 0.0
init_motor_y = 0.0
init_motor_z = 0.0

max_simulation_steps = 10_000_000_000  # hard safety cap

print(f"Maximum steps (safety cap): {max_simulation_steps:,}")
print("Starting simulation...\n")
start_time = time.time()

results = run_simulation(
    max_simulation_steps, TARGET_BURSTS, BURST_MIN_GAP_F32,
    # Initial positions
    init_motor_x, init_motor_y, init_motor_z,
    # Noise amplitudes
    SQRT_2_DONOR_DIFF, SQRT_2_ACCEPTOR_DIFF,
    SQRT_2_DONOR_ORIENT_DIFF, SQRT_2_ACCEPTOR_ORIENT_DIFF,
    # INSIDE constants
    CARGO_DT_IN_F32,  MOTOR_DT_IN_F32,  SQRT_CARGO_DT_IN,  SQRT_2_MOTOR_DIFF_IN,
    CARGO_SPRING_IN,  CARGO_SPRING_HALF_IN,
    DONOR_RADIAL_TERM_IN,  ACCEPTOR_RADIAL_TERM_IN,
    DONOR_ORIENT_TERM_IN,  ACCEPTOR_ORIENT_TERM_IN,
    # OUTSIDE constants
    CARGO_DT_OUT_F32, MOTOR_DT_OUT_F32, SQRT_CARGO_DT_OUT, SQRT_2_MOTOR_DIFF_OUT,
    CARGO_SPRING_OUT, CARGO_SPRING_HALF_OUT,
    DONOR_RADIAL_TERM_OUT, ACCEPTOR_RADIAL_TERM_OUT,
    DONOR_ORIENT_TERM_OUT, ACCEPTOR_ORIENT_TERM_OUT,
    # Dye equilibria and offset
    RD_EQ_F32, RA_EQ_F32, I_EQ_F32,
    # Volume / intensity
    ELLIPSOID_A_SQ, ELLIPSOID_B_SQ, ELLIPSOID_C_SQ,
    ELLIPSOID_C_K_SQ, MINUS_2_OVER_WAIST_SQ, DETECTION_THRESHOLD_F32,
    # FRET
    FORSTER_RADIUS_F32, FORSTER_RADIUS_6,
    DONOR_RATE_F32, ACCEPTOR_RATE_F32,
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

event_times          = event_times[:num_events]
event_colors         = event_colors[:num_events]
event_donor_x        = event_donor_x[:num_events]
event_donor_y        = event_donor_y[:num_events]
event_donor_z        = event_donor_z[:num_events]
event_acceptor_x     = event_acceptor_x[:num_events]
event_acceptor_y     = event_acceptor_y[:num_events]
event_acceptor_z     = event_acceptor_z[:num_events]
event_interdye_dist  = event_interdye_dist[:num_events]
event_fret_eff       = event_fret_eff[:num_events]
event_donor_intensity    = event_donor_intensity[:num_events]
event_acceptor_intensity = event_acceptor_intensity[:num_events]

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
print(f"Data reduction factor:      {num_cargo_steps/max(num_events, 1):.1f}x")
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
    """Identify bursts from inter-photon gaps; return list of burst dicts."""
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


# Use a small positive min_gap for post-hoc analysis when BURST_MIN_GAP is 0
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
    f'confinement_box={MOTOR_CONFINEMENT_X:.0f}x{MOTOR_CONFINEMENT_Y:.0f}x{MOTOR_CONFINEMENT_Z:.0f}nm',
    f'forster_radius={FORSTER_RADIUS}nm',
    f'i_eq={I_EQ}nm',
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
print("ISOTROPIC + NANOSECOND SIMULATION COMPLETE!")
print(f"{'='*70}")
print(f"  Execution time:   {elapsed_time:.1f} s ({elapsed_time/60:.1f} min)")
print(f"  Steps/second:     {num_cargo_steps/elapsed_time:,.0f}")
print(f"  Photons/second:   {total_photons/elapsed_time:,.0f}")
print(f"  Bursts/second:    {total_bursts_collected/elapsed_time:,.1f}")
print(f"  Memory (events):  {num_events*12*4/1e6:.1f} MB  "
      f"(vs {num_cargo_steps*13*4/1e6:.1f} MB if storing all steps)")
print(f"\nFiles saved:")
print(f"  - photon_events_nanosecond.csv  ({len(event_df):,} events)")
print(f"  - photon_bursts_nanosecond.csv  ({len(all_bursts):,} bursts)")
print(f"{'='*70}\n")
