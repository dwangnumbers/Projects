"""
FRET (Förster Resonance Energy Transfer) Simulation - ULTRA-OPTIMIZED NANOSECOND VERSION
=========================================================================================

This simulation models the behavior of donor and acceptor fluorescent dyes
attached to a cargo molecule that is transported by a molecular motor.

ULTRA-OPTIMIZATIONS FOR NANOSECOND TIME SCALE:
- Float32 precision (2x memory savings, faster math)
- Only store data when inside confocal volume (10-100x data reduction)
- Skip photon simulation when outside detection volume
- ADAPTIVE TIME STEPPING: coarse steps outside confocal volume, fine inside
- Motor confined to a box around confocal volume (reflecting boundaries)
- Batch random number generation
- Inlined calculations in hot loops
- Reduced redundant computations
- Pre-computed lookup tables for intensity
"""

import numpy as np
import pandas as pd
from numba import jit
from scipy.special import gammainc
import time


# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

# Thermal energy at room temperature
kT = 4.1e-12  # pN*microns

# Hydrodynamic radii of fluorescent dyes (nanometers)
DONOR_RADIUS = 1.1      # nm
ACCEPTOR_RADIUS = 1.1   # nm

# Local viscosity of the medium
VISCOSITY = 8.9e-12  # kg/(s*nm)

# Friction coefficients (Stokes drag: 6pi x radius x viscosity)
DONOR_FRICTION    = 6 * np.pi * DONOR_RADIUS    * VISCOSITY
ACCEPTOR_FRICTION = 6 * np.pi * ACCEPTOR_RADIUS * VISCOSITY

# Diffusion coefficients (Einstein relation: kT/gamma)
DONOR_DIFFUSION    = kT / DONOR_FRICTION     # nm^2/s
ACCEPTOR_DIFFUSION = kT / ACCEPTOR_FRICTION  # nm^2/s

# Motor diffusion: 100 um^2/s -> 100 x (1000 nm)^2/s = 1e8 nm^2/s
MOTOR_DIFFUSION = 100000000  # nm^2/s
MOTOR_DIFFUSION_INSIDE = 10000

# Spring constants modeling the linker between dye and motor
DONOR_NUM_LINKS    = 12
ACCEPTOR_NUM_LINKS = 11
DONOR_SPRING_CONSTANT    = (1010 / DONOR_NUM_LINKS)    * 1e-9  # N/nm
ACCEPTOR_SPRING_CONSTANT = (1010 / ACCEPTOR_NUM_LINKS) * 1e-9  # N/nm

# Effective spring constants (spring constant / friction)
DONOR_K_HAT    = DONOR_SPRING_CONSTANT    / DONOR_FRICTION
ACCEPTOR_K_HAT = ACCEPTOR_SPRING_CONSTANT / ACCEPTOR_FRICTION

# Fluorescence decay rates
DONOR_FLUORESCENCE_RATE    = 1 / (4e-5)  # s^-1
ACCEPTOR_FLUORESCENCE_RATE = 1 / (4e-5)  # s^-1

# Initial positions of dyes relative to motor (nm)
DONOR_OFFSET_X    = 0.82
DONOR_OFFSET_Y    = 0.0
DONOR_OFFSET_Z    = 0.0
ACCEPTOR_OFFSET_X = 0.0
ACCEPTOR_OFFSET_Y = 0.0
ACCEPTOR_OFFSET_Z = 0.0


# =============================================================================
# SIMULATION TIME STEP PARAMETERS
# =============================================================================
# Two regimes are used:
#   INSIDE  the confocal volume: fine steps needed for accurate photon stats
#   OUTSIDE the confocal volume: coarse steps sufficient (no photon recording)

# INSIDE confocal volume
MOTOR_TIME_STEP_INSIDE = 10e-6    # 10 us  (motor coarse step)
CARGO_TIME_STEP_INSIDE = 100e-9   # 100 ns (cargo Langevin step)

# OUTSIDE confocal volume  (100x coarser cargo, 100x coarser motor)
MOTOR_TIME_STEP_OUTSIDE = 10e-4   # 1 ms
CARGO_TIME_STEP_OUTSIDE = 10e-6   # 10 us


# =============================================================================
# BURST / PHOTON TARGETS
# =============================================================================

TARGET_BURSTS = 10000
BURST_MIN_GAP = 2e-5  # minimum inter-photon gap that opens a new burst (seconds)


# =============================================================================
# SMOOTHED BURST SELECTION PARAMETERS (Fries et al. 1998, Section 3.2)
# =============================================================================
# The raw interphoton time lags are smoothed by averaging n consecutive lags.
# The smoothed lag follows a Gamma distribution for background photons (Eq. 25).
# A burst is declared when the smoothed lag drops below a threshold delta_t_th.
# The dimensionless threshold xi = delta_t_th / delta_t_av controls the
# false-positive rate alpha (Eq. 26).
#
# Paper used:  n = 50,  xi = 0.27,  delta_t_th = 45 us

BURST_SMOOTH_N    = 50     # number of consecutive lags to average
BURST_SMOOTH_XI   = 0.27   # dimensionless threshold (delta_t_th / delta_t_av)


# =============================================================================
# DETECTION PROBABILITIES
# =============================================================================

PROB_DONOR_NONRADIATIVE    = 0.15
PROB_ACCEPTOR_NONRADIATIVE = 0.21
PROB_ACCEPTOR_NOT_DETECTED = 0.25
PROB_DONOR_NOT_DETECTED    = 0.15
PROB_NO_DETECTION          = 0.975

QUANTUM_YIELD_DONOR    = 1 - PROB_DONOR_NONRADIATIVE
QUANTUM_YIELD_ACCEPTOR = 1 - PROB_ACCEPTOR_NONRADIATIVE

# FRET parameters
FORSTER_RADIUS = 1.0  # nm


# =============================================================================
# CONFOCAL MICROSCOPY PARAMETERS
# =============================================================================

ELLIPSOID_SEMI_AXIS_X = 250    # nm
ELLIPSOID_SEMI_AXIS_Y = 250    # nm
ELLIPSOID_SEMI_AXIS_Z = 1000   # nm

INTENSITY_MAX      = 1.0
REFRACTIVE_INDEX   = 1.3
NUMERICAL_APERTURE = 1.2
WAVE_VECTOR        = 2.33 * REFRACTIVE_INDEX / NUMERICAL_APERTURE
BEAM_WAIST         = (1 / (np.pi**(3/2) * WAVE_VECTOR))**(1/3)
DETECTION_THRESHOLD = 0

EPSILON = 1e-1


# =============================================================================
# MOTOR CONFINEMENT PARAMETERS
# =============================================================================
# The motor is confined to a box around the confocal volume using reflecting
# boundary conditions. The box half-widths are defined as a multiple
# (CONFINEMENT_FACTOR) of the ellipsoid semi-axes, centred on the origin.
#
# With MOTOR_DIFFUSION = 1e8 nm^2/s the motor would escape in microseconds
# without confinement, so this keeps it sampling the detection volume.

CONFINEMENT_FACTOR = 4  # box = CONFINEMENT_FACTOR × ellipsoid semi-axes

MOTOR_CONFINEMENT_X = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_X   # ±1000 nm
MOTOR_CONFINEMENT_Y = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Y   # ±1000 nm
MOTOR_CONFINEMENT_Z = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Z   # ±4000 nm


# =============================================================================
# DISPLAY SIMULATION SETUP
# =============================================================================

print("\n" + "="*70)
print("FRET SIMULATION - ULTRA-OPTIMIZED NANOSECOND VERSION")
print("="*70)
print(f"Target bursts:       {TARGET_BURSTS:,}")
print(f"Burst gap threshold: {BURST_MIN_GAP*1e3:.1f} ms")
print(f"Motor diffusion:     {MOTOR_DIFFUSION:.2e} nm^2/s  ({MOTOR_DIFFUSION/1e6:.0f} um^2/s)")
print()
print(f"Motor confinement box (reflecting boundaries):")
print(f"  X: +/-{MOTOR_CONFINEMENT_X:.0f} nm  ({CONFINEMENT_FACTOR}x ellipsoid semi-axis)")
print(f"  Y: +/-{MOTOR_CONFINEMENT_Y:.0f} nm  ({CONFINEMENT_FACTOR}x ellipsoid semi-axis)")
print(f"  Z: +/-{MOTOR_CONFINEMENT_Z:.0f} nm  ({CONFINEMENT_FACTOR}x ellipsoid semi-axis)")
print()
print(f"  Time step          INSIDE volume        OUTSIDE volume")
print(f"  ---------------------------------------------------------")
print(f"  Motor (coarse)     {MOTOR_TIME_STEP_INSIDE*1e6:7.1f} us            {MOTOR_TIME_STEP_OUTSIDE*1e3:7.1f} ms")
print(f"  Cargo (Langevin)   {CARGO_TIME_STEP_INSIDE*1e9:7.1f} ns            {CARGO_TIME_STEP_OUTSIDE*1e6:7.1f} us")
print()
print(f"Forster radius:      {FORSTER_RADIUS} nm")
print(f"Donor lifetime:      {1/DONOR_FLUORESCENCE_RATE*1e6:.1f} us")
print(f"Acceptor lifetime:   {1/ACCEPTOR_FLUORESCENCE_RATE*1e6:.1f} us")
print()
print(f"Smoothed burst selection (Fries et al. 1998, Sec. 3.2):")
print(f"  Smoothing window n:          {BURST_SMOOTH_N}")
print(f"  Dimensionless threshold xi:  {BURST_SMOOTH_XI}")
print("\nULTRA-OPTIMIZATIONS ENABLED:")
print("  + Float32 precision (2x memory savings)")
print("  + Adaptive time stepping (coarse outside, fine inside)")
print("  + Motor confined to box with reflecting boundaries")
print("  + Only store data inside confocal volume")
print("  + Skip photon simulation outside volume")
print("  + Aggressive JIT optimization")
print("="*70 + "\n")


# =============================================================================
# PRE-COMPUTE CONSTANTS FOR PERFORMANCE
# =============================================================================

# Dye diffusion (time-step independent)
SQRT_2_DONOR_DIFF    = np.float32(np.sqrt(2 * DONOR_DIFFUSION))
SQRT_2_ACCEPTOR_DIFF = np.float32(np.sqrt(2 * ACCEPTOR_DIFFUSION))

# INSIDE volume pre-computed constants
SQRT_CARGO_DT_IN        = np.float32(np.sqrt(CARGO_TIME_STEP_INSIDE))
SQRT_2_MOTOR_DIFF_IN    = np.float32(np.sqrt(2 * MOTOR_DIFFUSION_INSIDE * MOTOR_TIME_STEP_INSIDE))
DONOR_SPRING_IN         = np.float32(DONOR_K_HAT         * CARGO_TIME_STEP_INSIDE)
DONOR_SPRING_HALF_IN    = np.float32((DONOR_K_HAT    / 2) * CARGO_TIME_STEP_INSIDE)
ACCEPTOR_SPRING_IN      = np.float32(ACCEPTOR_K_HAT       * CARGO_TIME_STEP_INSIDE)
ACCEPTOR_SPRING_HALF_IN = np.float32((ACCEPTOR_K_HAT / 2) * CARGO_TIME_STEP_INSIDE)

# OUTSIDE volume pre-computed constants
SQRT_CARGO_DT_OUT        = np.float32(np.sqrt(CARGO_TIME_STEP_OUTSIDE))
SQRT_2_MOTOR_DIFF_OUT    = np.float32(np.sqrt(2 * MOTOR_DIFFUSION * MOTOR_TIME_STEP_OUTSIDE))
DONOR_SPRING_OUT         = np.float32(DONOR_K_HAT         * CARGO_TIME_STEP_OUTSIDE)
DONOR_SPRING_HALF_OUT    = np.float32((DONOR_K_HAT    / 2) * CARGO_TIME_STEP_OUTSIDE)
ACCEPTOR_SPRING_OUT      = np.float32(ACCEPTOR_K_HAT       * CARGO_TIME_STEP_OUTSIDE)
ACCEPTOR_SPRING_HALF_OUT = np.float32((ACCEPTOR_K_HAT / 2) * CARGO_TIME_STEP_OUTSIDE)

# Ellipsoid / intensity
ELLIPSOID_A_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_X**2)
ELLIPSOID_B_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_Y**2)
ELLIPSOID_C_SQ          = np.float32(ELLIPSOID_SEMI_AXIS_Z**2)
ELLIPSOID_C_K_SQ        = np.float32((ELLIPSOID_SEMI_AXIS_Z * WAVE_VECTOR)**2)
MINUS_2_OVER_WAIST_SQ   = np.float32(-1 / (BEAM_WAIST**2))
DETECTION_THRESHOLD_F32 = np.float32(DETECTION_THRESHOLD)

# FRET
FORSTER_RADIUS_6        = np.float32(FORSTER_RADIUS**6)
FORSTER_RADIUS_F32      = np.float32(FORSTER_RADIUS)
DONOR_FLUOR_RATE_F32    = np.float32(DONOR_FLUORESCENCE_RATE)
ACCEPTOR_FLUOR_RATE_F32 = np.float32(ACCEPTOR_FLUORESCENCE_RATE)

# Detection probabilities
PROB_QD = np.float32(PROB_DONOR_NONRADIATIVE)
PROB_QA = np.float32(PROB_ACCEPTOR_NONRADIATIVE)
PROB_AS = np.float32(PROB_ACCEPTOR_NOT_DETECTED)
PROB_DS = np.float32(PROB_DONOR_NOT_DETECTED)
PROB_ND = np.float32(PROB_NO_DETECTION)

# Burst gap
BURST_MIN_GAP_F32 = np.float32(BURST_MIN_GAP)

# Time steps as float32 scalars for JIT
CARGO_DT_IN_F32  = np.float32(CARGO_TIME_STEP_INSIDE)
CARGO_DT_OUT_F32 = np.float32(CARGO_TIME_STEP_OUTSIDE)
MOTOR_DT_IN_F32  = np.float32(MOTOR_TIME_STEP_INSIDE)
MOTOR_DT_OUT_F32 = np.float32(MOTOR_TIME_STEP_OUTSIDE)

# Motor confinement box half-widths as float32
MOTOR_CONF_X_F32 = np.float32(MOTOR_CONFINEMENT_X)
MOTOR_CONF_Y_F32 = np.float32(MOTOR_CONFINEMENT_Y)
MOTOR_CONF_Z_F32 = np.float32(MOTOR_CONFINEMENT_Z)


# =============================================================================
# ULTRA-OPTIMIZED FUNCTIONS (USING NUMBA JIT COMPILATION)
# =============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def is_inside_volume(x, y, z, a_sq, b_sq, c_sq):
    """Fast check if position is inside ellipsoid."""
    return (x*x) / a_sq + (y*y) / b_sq + (z*z) / c_sq <= 1.0


@jit(nopython=True, fastmath=True, cache=True)
def calculate_intensity_fast(x, y, z, a_sq, b_sq, c_k_sq, minus_2_over_w_sq):
    """Calculate Gaussian beam intensity (assumes already inside ellipsoid)."""
    exponent = minus_2_over_w_sq * ((x*x)/a_sq + (y*y)/b_sq + (z*z)/c_k_sq)
    return np.exp(exponent)


@jit(nopython=True, fastmath=True, cache=True)
def reflect(value, bound):
    """
    Apply reflecting boundary condition to a 1-D coordinate.

    Folds the value back into [-bound, +bound] by reflecting off each wall.
    Handles large steps that may overshoot multiple times via modular folding.

      - Shift into [0, 2*bound], fold with modulo, shift back.
      - The fold maps [0,b]->[0,b] and [b,2b]->[2b,b] (mirror image).
    """
    span = np.float32(2.0) * bound
    # shift to [0, 2*bound]
    v = value + bound
    # wrap with modulo so large jumps are handled
    v = v - span * np.floor(v / span)
    # reflect upper half back: if v > bound, mirror it
    if v > bound:
        v = span - v
    return v - bound


@jit(nopython=True, fastmath=True, cache=True)
def simulate_photon_fast(starting_state, fret_rate, donor_rate, acceptor_rate,
                         prob_qd, prob_qa, prob_as, prob_ds, prob_nd):
    """
    Ultra-fast photon emission simulation.
    Returns: (state, is_detected)
      state 4 = donor photon detected
      state 5 = acceptor photon detected
      state 3 = no photon
    """
    if np.random.random() < prob_nd:
        return np.int32(3), np.int32(0)

    state = starting_state

    if state == 1:  # Donor excited
        if np.random.random() < prob_qd:
            return np.int32(3), np.int32(0)
        # Exponential race: FRET vs donor emission
        time_to_fret  = -np.log(np.random.random()) / fret_rate
        time_to_donor = -np.log(np.random.random()) / donor_rate
        if time_to_fret < time_to_donor:
            state = 2  # FRET transfers energy to acceptor
        else:
            if np.random.random() < prob_ds:
                return np.int32(3), np.int32(0)
            return np.int32(4), np.int32(1)  # donor photon detected

    if state == 2:  # Acceptor excited
        if np.random.random() < prob_qa:
            return np.int32(3), np.int32(0)
        if np.random.random() < prob_as:
            return np.int32(3), np.int32(0)
        return np.int32(5), np.int32(1)  # acceptor photon detected

    return np.int32(3), np.int32(0)


@jit(nopython=True, fastmath=True, cache=True)
def run_ultra_optimized_simulation(
    max_steps, target_bursts, burst_min_gap,
    # Initial positions
    init_motor_x, init_motor_y, init_motor_z,
    init_donor_x,    init_donor_y,    init_donor_z,
    init_acceptor_x, init_acceptor_y, init_acceptor_z,
    # Dye diffusion (time-step independent)
    sqrt_2_donor_diff, sqrt_2_acceptor_diff,
    # INSIDE time step constants
    cargo_dt_in, motor_dt_in,
    sqrt_cargo_dt_in, sqrt_2_motor_diff_in,
    donor_spring_in,      donor_spring_half_in,
    acceptor_spring_in,   acceptor_spring_half_in,
    # OUTSIDE time step constants
    cargo_dt_out, motor_dt_out,
    sqrt_cargo_dt_out, sqrt_2_motor_diff_out,
    donor_spring_out,     donor_spring_half_out,
    acceptor_spring_out,  acceptor_spring_half_out,
    # Volume / intensity
    a_sq, b_sq, c_sq, c_k_sq, minus_2_w_sq, detection_threshold,
    # FRET
    forster_r, forster_r6, donor_rate, acceptor_rate,
    # Probabilities
    prob_qd, prob_qa, prob_as, prob_ds, prob_nd,
    # Motor confinement box half-widths
    conf_x, conf_y, conf_z):
    """
    Ultra-optimized main loop with ADAPTIVE TIME STEPPING and MOTOR CONFINEMENT.

    Outside the confocal volume: large cargo_dt + large motor_dt
      -> fast propagation through free diffusion, no photon recording
    Inside  the confocal volume: small cargo_dt + small motor_dt
      -> accurate Langevin dynamics, full photon simulation

    Motor updates are triggered by elapsed time (not step count) so the
    coarse/fine switch is seamless when crossing the volume boundary.

    After each motor displacement, reflecting boundary conditions fold the
    motor back into the confinement box [-conf_x, +conf_x] x [-conf_y,
    +conf_y] x [-conf_z, +conf_z], preventing escape due to large diffusion.

    Terminates after target_bursts complete bursts have been collected.
    A new burst is opened whenever a photon arrives after a gap >= burst_min_gap
    from the previous detected photon (or on the very first detected photon).
    """
    max_events = target_bursts * 200  # generous buffer (~200 photons/burst)

    # Photon event storage
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

    # Scalar positions
    motor_x = np.float32(init_motor_x)
    motor_y = np.float32(init_motor_y)
    motor_z = np.float32(init_motor_z)
    donor_x    = np.float32(init_donor_x)
    donor_y    = np.float32(init_donor_y)
    donor_z    = np.float32(init_donor_z)
    acceptor_x = np.float32(init_acceptor_x)
    acceptor_y = np.float32(init_acceptor_y)
    acceptor_z = np.float32(init_acceptor_z)

    # Timers and counters
    current_time     = np.float64(0.0)
    last_motor_time  = np.float64(0.0)
    last_photon_time = np.float64(-1.0)   # -1 signals "no photon yet"

    total_photons = np.int64(0)
    total_bursts  = np.int64(0)
    event_count   = np.int64(0)
    cargo_step    = np.int64(0)

    in_volume_count = np.int64(0)
    total_steps     = np.int64(0)

    # Begin outside the volume
    was_in_volume = False

    # -------------------------------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------------------------------
    while cargo_step < max_steps and total_bursts < target_bursts:

        # Select time step constants based on previous step's volume status
        if was_in_volume:
            cargo_dt      = cargo_dt_in
            motor_dt      = motor_dt_in
            sqrt_cdt      = sqrt_cargo_dt_in
            sqrt2_m       = sqrt_2_motor_diff_in
            d_spr         = donor_spring_in
            d_spr_half    = donor_spring_half_in
            a_spr         = acceptor_spring_in
            a_spr_half    = acceptor_spring_half_in
        else:
            cargo_dt      = cargo_dt_out
            motor_dt      = motor_dt_out
            sqrt_cdt      = sqrt_cargo_dt_out
            sqrt2_m       = sqrt_2_motor_diff_out
            d_spr         = donor_spring_out
            d_spr_half    = donor_spring_half_out
            a_spr         = acceptor_spring_out
            a_spr_half    = acceptor_spring_half_out

        # Update motor when enough time has elapsed (time-based, not step-based)
        if current_time - last_motor_time >= motor_dt:
            motor_x += sqrt2_m * np.random.randn()
            motor_y += sqrt2_m * np.random.randn()
            motor_z += sqrt2_m * np.random.randn()
            last_motor_time = current_time

            # --- Reflecting boundary conditions ---
            # Fold motor back into confinement box after each displacement.
            # reflect() handles arbitrarily large steps via modular folding.
            motor_x = reflect(motor_x, conf_x)
            motor_y = reflect(motor_y, conf_y)
            motor_z = reflect(motor_z, conf_z)

        # Update donor position (Langevin dynamics)
        noise_d = sqrt_2_donor_diff * sqrt_cdt
        donor_x += d_spr      * (motor_x - donor_x) + noise_d * np.random.randn()
        donor_y += d_spr_half * (motor_y - donor_y) + noise_d * np.random.randn()
        donor_z += d_spr      * (motor_z - donor_z) + noise_d * np.random.randn()

        # Update acceptor position (Langevin dynamics)
        noise_a = sqrt_2_acceptor_diff * sqrt_cdt
        acceptor_x += a_spr      * (motor_x - acceptor_x) + noise_a * np.random.randn()
        acceptor_y += a_spr      * (motor_y - acceptor_y) + noise_a * np.random.randn()
        acceptor_z += a_spr_half * (motor_z - acceptor_z) + noise_a * np.random.randn()

        # Confocal volume check
        donor_in    = is_inside_volume(donor_x,    donor_y,    donor_z,    a_sq, b_sq, c_sq)
        acceptor_in = is_inside_volume(acceptor_x, acceptor_y, acceptor_z, a_sq, b_sq, c_sq)

        total_steps += 1

        donor_intensity    = np.float32(0.0)
        acceptor_intensity = np.float32(0.0)

        if donor_in:
            donor_intensity = calculate_intensity_fast(
                donor_x, donor_y, donor_z, a_sq, b_sq, c_k_sq, minus_2_w_sq)
            donor_in = donor_intensity > detection_threshold

        if acceptor_in:
            acceptor_intensity = calculate_intensity_fast(
                acceptor_x, acceptor_y, acceptor_z, a_sq, b_sq, c_k_sq, minus_2_w_sq)
            acceptor_in = acceptor_intensity > detection_threshold

        currently_in_volume = donor_in or acceptor_in

        # Photon simulation -- only when at least one dye is detectable
        if currently_in_volume:
            in_volume_count += 1

            dx = donor_x - acceptor_x
            dy = donor_y - acceptor_y
            dz = donor_z - acceptor_z
            dist_sq   = dx*dx + dy*dy + dz*dz
            dist      = np.sqrt(dist_sq)
            dist_6    = dist_sq * dist_sq * dist_sq

            fret_eff  = np.float32(1.0) / (np.float32(1.0) + dist_6 / forster_r6)
            fret_rate = donor_rate * forster_r6 / dist_6

            total_int = donor_intensity + acceptor_intensity
            if total_int > np.float32(0.0):
                starting_state = np.int32(1) \
                    if np.random.random() < (donor_intensity / total_int) \
                    else np.int32(2)

                photon_state, is_detected = simulate_photon_fast(
                    starting_state, fret_rate, donor_rate, acceptor_rate,
                    prob_qd, prob_qa, prob_as, prob_ds, prob_nd
                )

                if is_detected == 1 and event_count < max_events:
                    # Burst boundary: first photon, or gap exceeds threshold
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
# RUN SIMULATION
# =============================================================================

print("Preparing ultra-optimized simulation...")
print("Note: First run includes JIT compilation time (~1-2 min)\n")

init_motor_x = 250.0   # start at origin (centre of confinement box)
init_motor_y = 0.0
init_motor_z = 0.0

init_donor_x    = init_motor_x + DONOR_OFFSET_X
init_donor_y    = init_motor_y + DONOR_OFFSET_Y
init_donor_z    = init_motor_z + DONOR_OFFSET_Z
init_acceptor_x = init_motor_x + ACCEPTOR_OFFSET_X
init_acceptor_y = init_motor_y + ACCEPTOR_OFFSET_Y
init_acceptor_z = init_motor_z + ACCEPTOR_OFFSET_Z

max_simulation_steps = 10_000_000_000  # hard safety cap

print(f"Maximum simulation steps (safety cap): {max_simulation_steps:,}")
print("Starting simulation...\n")
start_time = time.time()

results = run_ultra_optimized_simulation(
    max_simulation_steps, TARGET_BURSTS, BURST_MIN_GAP_F32,
    init_motor_x, init_motor_y, init_motor_z,
    init_donor_x,    init_donor_y,    init_donor_z,
    init_acceptor_x, init_acceptor_y, init_acceptor_z,
    SQRT_2_DONOR_DIFF, SQRT_2_ACCEPTOR_DIFF,
    # INSIDE
    CARGO_DT_IN_F32,  MOTOR_DT_IN_F32,
    SQRT_CARGO_DT_IN, SQRT_2_MOTOR_DIFF_IN,
    DONOR_SPRING_IN,      DONOR_SPRING_HALF_IN,
    ACCEPTOR_SPRING_IN,   ACCEPTOR_SPRING_HALF_IN,
    # OUTSIDE
    CARGO_DT_OUT_F32,  MOTOR_DT_OUT_F32,
    SQRT_CARGO_DT_OUT, SQRT_2_MOTOR_DIFF_OUT,
    DONOR_SPRING_OUT,     DONOR_SPRING_HALF_OUT,
    ACCEPTOR_SPRING_OUT,  ACCEPTOR_SPRING_HALF_OUT,
    ELLIPSOID_A_SQ, ELLIPSOID_B_SQ, ELLIPSOID_C_SQ,
    ELLIPSOID_C_K_SQ, MINUS_2_OVER_WAIST_SQ, DETECTION_THRESHOLD_F32,
    FORSTER_RADIUS_F32, FORSTER_RADIUS_6,
    DONOR_FLUOR_RATE_F32, ACCEPTOR_FLUOR_RATE_F32,
    PROB_QD, PROB_QA, PROB_AS, PROB_DS, PROB_ND,
    MOTOR_CONF_X_F32, MOTOR_CONF_Y_F32, MOTOR_CONF_Z_F32   # confinement
)

elapsed_time = time.time() - start_time

# Unpack results
(num_cargo_steps, total_photons, total_bursts_collected, num_events, occupancy,
 total_sim_time_f32,
 event_times, event_colors,
 event_donor_x,    event_donor_y,    event_donor_z,
 event_acceptor_x, event_acceptor_y, event_acceptor_z,
 event_interdye_dist, event_fret_eff,
 event_donor_intensity, event_acceptor_intensity) = results

# Trim to actual size
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
print(f"Simulated physical time:    {total_simulation_time:.3f} s")
print(f"Cargo steps taken:          {num_cargo_steps:,}")
print(f"Confocal volume occupancy:  {occupancy*100:.2f}%")
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
print(f"  Mean interdye distance: {np.mean(event_interdye_dist):.2f} nm")


# =============================================================================
# POST-HOC BURST ANALYSIS — SMOOTHED INTERPHOTON TIME METHOD
# (Fries et al. 1998, Section 3.2, Eqs. 24–26)
# =============================================================================
#
# Background:
#   Raw interphoton time lags (delta_t) are exponentially distributed for
#   background (Poisson) photons (Eq. 24).  Averaging n consecutive lags
#   produces a smoothed lag delta_t_sm(n) whose distribution is a Gamma
#   distribution (Eq. 25) — much more sharply peaked than the exponential,
#   enabling robust threshold-based burst discrimination.
#
# Algorithm:
#   1. Compute interphoton time lags:  delta_t_i = t_{i+1} - t_i
#   2. Compute running average over n consecutive lags (smoothing window)
#   3. Estimate background mean lag (delta_t_av) from the full lag distribution
#   4. Compute threshold:  delta_t_th = xi * delta_t_av
#   5. Mark regions where delta_t_sm < delta_t_th as "in-burst"
#   6. Segment contiguous in-burst regions into individual bursts
#   7. Compute the false-positive rate alpha from the regularized lower
#      incomplete Gamma function (Eq. 26)

print("\n" + "-"*70)
print("SMOOTHED BURST SELECTION (Fries et al. 1998, Section 3.2)")
print("-"*70)


def compute_burst_selection_alpha(n, xi):
    """
    Compute the error probability alpha (Eq. 26 of Fries et al. 1998).

    alpha = P(delta_t_sm(n) <= delta_t_th) / delta_t_av
          = integral from 0 to xi of  [n^n / (n-1)!] * u^{n-1} * exp(-n*u) du

    This is the regularized lower incomplete gamma function:
        alpha = gammainc(n, n * xi)

    where gammainc is scipy's regularized lower incomplete gamma function
    P(a, x) = (1/Gamma(a)) * integral_0^x t^{a-1} exp(-t) dt.

    Parameters
    ----------
    n  : int   — smoothing window size
    xi : float — dimensionless threshold (delta_t_th / delta_t_av)

    Returns
    -------
    alpha : float — false-positive probability per background lag
    """
    return float(gammainc(n, n * xi))


def identify_bursts_smoothed(times, colors, n_smooth, xi,
                             interdye_dists=None, fret_effs=None):
    """
    Identify photon bursts using the smoothed interphoton time lag method
    described in Section 3.2 of Fries et al., J. Phys. Chem. A, 102 (1998).

    Parameters
    ----------
    times      : 1-D array of photon arrival times (sorted, seconds)
    colors     : 1-D array of photon color codes (4=donor, 5=acceptor)
    n_smooth   : int — number of consecutive lags to average (Eq. 25)
    xi         : float — dimensionless threshold  delta_t_th / delta_t_av
    interdye_dists : optional 1-D array of interdye distances per photon
    fret_effs      : optional 1-D array of FRET efficiencies per photon

    Returns
    -------
    bursts : list of dicts, each containing:
        entry_time, exit_time, duration, total_photons,
        donor_photons, acceptor_photons, mean_fret_eff, mean_interdye_dist
    dt_smooth   : 1-D array of smoothed interphoton lags (for diagnostics)
    dt_threshold: float — the absolute threshold used (seconds)
    dt_av       : float — estimated mean background lag (seconds)
    """
    n_photons = len(times)

    if n_photons < n_smooth + 1:
        return [], np.array([]), 0.0, 0.0

    # ------------------------------------------------------------------
    # Step 1: Compute raw interphoton time lags  (Eq. 24)
    # ------------------------------------------------------------------
    dt_raw = np.diff(times.astype(np.float64))  # float64 for precision

    # ------------------------------------------------------------------
    # Step 2: Smoothed running average over n consecutive lags  (Eq. 25)
    # ------------------------------------------------------------------
    # Use a uniform convolution kernel of width n_smooth.
    # The output has length  len(dt_raw) - n_smooth + 1.
    kernel = np.ones(n_smooth, dtype=np.float64) / n_smooth
    dt_smooth = np.convolve(dt_raw, kernel, mode='valid')

    # ------------------------------------------------------------------
    # Step 3: Estimate the mean background interphoton lag  (delta_t_av)
    # ------------------------------------------------------------------
    # The background lag is estimated from the median of all raw lags,
    # which is robust against the presence of burst photons (where lags
    # are very short).  For an exponential distribution with mean lambda,
    # the median is lambda * ln(2), so delta_t_av = median / ln(2).
    dt_median = np.median(dt_raw)
    dt_av = dt_median / np.log(2.0)

    # ------------------------------------------------------------------
    # Step 4: Compute absolute threshold  (delta_t_th = xi * delta_t_av)
    # ------------------------------------------------------------------
    dt_threshold = xi * dt_av

    # ------------------------------------------------------------------
    # Step 5: Boolean mask — in-burst where smoothed lag < threshold
    # ------------------------------------------------------------------
    in_burst_mask = dt_smooth < dt_threshold

    # Map smoothed-lag indices back to photon indices.
    # dt_smooth[k] corresponds to photon indices [k, k+n_smooth].
    # When dt_smooth[k] < threshold, photons k through k+n_smooth are
    # considered to be in-burst.
    photon_in_burst = np.zeros(n_photons, dtype=bool)
    for k in range(len(dt_smooth)):
        if in_burst_mask[k]:
            photon_in_burst[k : k + n_smooth + 1] = True

    # ------------------------------------------------------------------
    # Step 6: Segment contiguous in-burst regions into individual bursts
    # ------------------------------------------------------------------
    bursts = []
    i = 0
    while i < n_photons:
        if photon_in_burst[i]:
            # Start of a burst — collect all contiguous in-burst photons
            j = i
            while j < n_photons and photon_in_burst[j]:
                j += 1

            burst_indices = np.arange(i, j)
            bc = colors[burst_indices]

            burst_dict = {
                'entry_time':       float(times[i]),
                'exit_time':        float(times[j - 1]),
                'duration':         float(times[j - 1] - times[i]),
                'total_photons':    len(burst_indices),
                'donor_photons':    int(np.sum(bc == 4)),
                'acceptor_photons': int(np.sum(bc == 5)),
            }

            if interdye_dists is not None:
                burst_dict['mean_interdye_dist'] = float(
                    np.mean(interdye_dists[burst_indices]))
            if fret_effs is not None:
                burst_dict['mean_fret_eff'] = float(
                    np.mean(fret_effs[burst_indices]))

            bursts.append(burst_dict)
            i = j
        else:
            i += 1

    return bursts, dt_smooth, dt_threshold, dt_av


# --- Run the smoothed burst selection ---
all_bursts, dt_smooth, dt_threshold, dt_av = identify_bursts_smoothed(
    event_times, event_colors,
    n_smooth=BURST_SMOOTH_N,
    xi=BURST_SMOOTH_XI,
    interdye_dists=event_interdye_dist,
    fret_effs=event_fret_eff,
)

# --- Compute the false-positive rate alpha (Eq. 26) ---
alpha = compute_burst_selection_alpha(BURST_SMOOTH_N, BURST_SMOOTH_XI)

print(f"\n  Parameters:")
print(f"    Smoothing window n:          {BURST_SMOOTH_N}")
print(f"    Dimensionless threshold xi:  {BURST_SMOOTH_XI}")
print(f"    Estimated background lag:    {dt_av*1e6:.1f} us  (delta_t_av)")
print(f"    Absolute threshold:          {dt_threshold*1e6:.1f} us  (xi * delta_t_av)")
print(f"    False-positive rate alpha:   {alpha:.2e}  (Eq. 26)")
print(f"    Smoothed lag array size:     {len(dt_smooth):,}")
print()
print(f"  Results:")
print(f"    Bursts identified (smoothed):  {len(all_bursts):,}")
print(f"    Bursts tracked in-loop (gap):  {total_bursts_collected:,}")

if len(all_bursts) > 0:
    burst_sizes = np.array([b['total_photons'] for b in all_bursts])
    print(f"    Avg photons/burst:             {np.mean(burst_sizes):.1f}")
    print(f"    Median photons/burst:          {np.median(burst_sizes):.1f}")
    print(f"    Max photons/burst:             {np.max(burst_sizes)}")
    print(f"    Avg donor photons/burst:       "
          f"{np.mean([b['donor_photons'] for b in all_bursts]):.1f}")
    print(f"    Avg acceptor photons/burst:    "
          f"{np.mean([b['acceptor_photons'] for b in all_bursts]):.1f}")
    if 'mean_fret_eff' in all_bursts[0]:
        print(f"    Avg burst FRET efficiency:     "
              f"{np.mean([b['mean_fret_eff'] for b in all_bursts]):.3f}")
    if 'mean_interdye_dist' in all_bursts[0]:
        print(f"    Avg burst interdye distance:   "
              f"{np.mean([b['mean_interdye_dist'] for b in all_bursts]):.2f} nm")

print("-"*70)


# =============================================================================
# SAVE RESULTS TO CSV
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
summary_data = {
    'entry_time': ['SUMMARY'] * 14,
    'exit_time': [
        f'total_photons={total_photons}',
        f'donor_photons={donor_count}',
        f'acceptor_photons={acceptor_count}',
        f'total_bursts_in_loop={total_bursts_collected}',
        f'total_bursts_smoothed={len(all_bursts)}',
        f'target_bursts={TARGET_BURSTS}',
        f'simulation_time={total_simulation_time:.4f}s',
        f'volume_occupancy={occupancy:.4f}',
        f'cargo_dt_in={CARGO_TIME_STEP_INSIDE}s',
        f'cargo_dt_out={CARGO_TIME_STEP_OUTSIDE}s',
        f'confinement_box={MOTOR_CONFINEMENT_X:.0f}x{MOTOR_CONFINEMENT_Y:.0f}x{MOTOR_CONFINEMENT_Z:.0f}nm',
        f'confinement_factor={CONFINEMENT_FACTOR}',
        f'burst_smooth_n={BURST_SMOOTH_N}',
        f'burst_smooth_xi={BURST_SMOOTH_XI}',
    ],
    'duration':          [''] * 14,
    'total_photons':     [''] * 14,
    'donor_photons':     [''] * 14,
    'acceptor_photons':  [''] * 14,
}

# Add optional columns if present in bursts
if len(all_bursts) > 0 and 'mean_fret_eff' in all_bursts[0]:
    summary_data['mean_fret_eff']      = [''] * 14
    summary_data['mean_interdye_dist'] = [''] * 14

summary_df = pd.DataFrame(summary_data)

if len(all_bursts) > 0:
    # Reorder columns to match summary
    cols = ['entry_time', 'exit_time', 'duration',
            'total_photons', 'donor_photons', 'acceptor_photons']
    if 'mean_fret_eff' in burst_df.columns:
        cols += ['mean_fret_eff', 'mean_interdye_dist']
    burst_df = burst_df[cols]

final_burst_df = pd.concat([summary_df, burst_df], ignore_index=True)
final_burst_df.to_csv('photon_bursts_nanosecond.csv', index=False)
print(f"  Saved burst analysis to 'photon_bursts_nanosecond.csv'\n")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"{'='*70}")
print("ULTRA-OPTIMIZED NANOSECOND SIMULATION COMPLETE!")
print(f"{'='*70}")
print(f"  Execution time:  {elapsed_time:.1f} s ({elapsed_time/60:.1f} min)")
print(f"  Steps/second:    {num_cargo_steps/elapsed_time:,.0f}")
print(f"  Photons/second:  {total_photons/elapsed_time:,.0f}")
print(f"  Bursts/second:   {total_bursts_collected/elapsed_time:,.1f}")
print(f"  Memory (events): {num_events*12*4/1e6:.1f} MB  "
      f"(vs {num_cargo_steps*13*4/1e6:.1f} MB if storing all steps)")
print(f"\n  Burst selection method: Smoothed interphoton time lag")
print(f"    n={BURST_SMOOTH_N}, xi={BURST_SMOOTH_XI}, "
      f"alpha={alpha:.2e} (Fries et al. 1998, Eq. 26)")
print(f"    Bursts (smoothed): {len(all_bursts):,}  |  "
      f"Bursts (in-loop gap): {total_bursts_collected:,}")
print(f"\nFiles saved:")
print(f"  - photon_events_nanosecond.csv  ({len(event_df):,} events)")
print(f"  - photon_bursts_nanosecond.csv  ({len(all_bursts):,} bursts)")
print(f"{'='*70}\n")
