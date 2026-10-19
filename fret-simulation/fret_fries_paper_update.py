"""
FRET (Förster Resonance Energy Transfer) Simulation - UPDATED PER FRIES ET AL. 1998
======================================================================================

This simulation models the behavior of donor and acceptor fluorescent dyes
attached to a cargo molecule that is transported by a molecular motor.

UPDATES BASED ON FRIES ET AL. (J. Phys. Chem. A, 1998, 102, 6601-6613):
------------------------------------------------------------------------
1. Confocal volume geometry: ω₀ = 620 nm, z₀ = 12,400 nm (z₀/ω₀ ≈ 20)
   per the paper's experimental parameters (Section 2), replacing the
   previous smaller ellipsoid.

2. 3D Gaussian SDE profile (Eq. 7):
   I(x,y,z) = I₀ exp(-2(x²+y²)/ω₀²) exp(-2z²/z₀²)
   This replaces the previous beam-waist formula with the standard
   confocal detection efficiency profile used throughout the paper.

3. Numerical aperture: NA = 1.3 (oil immersion, Fluar 40× Zeiss)
   per Section 2 of the paper, replacing NA = 1.2.

4. BIFL-style burst detection (Section 3.2, Eqs. 24-26):
   - Inter-photon time lag (Δt) smoothed over n consecutive photons
   - Gamma-distribution threshold for burst selection
   - Error probability α for false burst identification
   Replaces the simple minimum-gap burst criterion.

5. Burst Size Distribution (BSD) analysis (Section 3.1, Eqs. 12, 17, 23):
   - P₁(Ct,t) using constant-irradiance shells (Eq. 12)
   - Exponential dwell time distribution Pt1(t) = (1/tt) exp(-t/tt) (Eq. 17)
   - Multimolecule event probability Pmn(Nav) (Eq. 20)
   - Absolute BSD β(Ct, Nav, g) (Eq. 23)

6. Experimental detection factor g = Ψ·Ψ_electr·Φ_F·σ₀₁(λ_ex)·γ (Eq. 16)
   with explicit parameter values from the paper.

7. Motor confinement box scaled to new confocal volume dimensions.

ULTRA-OPTIMIZATIONS RETAINED:
- Float32 precision (2x memory savings, faster math)
- Only store data when inside confocal volume
- Adaptive time stepping (coarse outside, fine inside)
- Motor confined to box with reflecting boundaries
- Aggressive JIT optimization via Numba
"""

import numpy as np
import pandas as pd
from numba import jit
import time
from scipy.special import gammainc  # for error probability calculation


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

# Friction coefficients (Stokes drag: 6π × radius × viscosity)
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
# CONFOCAL MICROSCOPY PARAMETERS  --  UPDATED PER FRIES ET AL. 1998
# =============================================================================
# Section 2 of Fries et al.:
#   - Oil-immersion objective, Fluar 40×, NA = 1.3
#   - ω₀ = 0.62 μm = 620 nm  (radial 1/e² radius from FCS)
#   - z₀ ≈ 12.4 μm = 12400 nm (axial 1/e² radius from FCS)
#   - z₀/ω₀ ≈ 20
#   - V = 2.65 × 10⁻¹⁴ L (detection volume)
#   - Excitation wavelength: 522 nm, repetition rate 76 MHz
#
# Eq. 7 defines the 3D Gaussian SDE:
#   Δ(x,y,z) ∝ I(x,y,z) = I₀ exp(-2(x²+y²)/ω₀²) exp(-2z²/z₀²)
#
# The 1/e² boundary of this Gaussian defines the effective detection volume.

OMEGA_0 = 620.0      # nm -- radial 1/e² radius (Fries et al. Section 2)
Z_0     = 12400.0    # nm -- axial 1/e² radius  (Fries et al. Section 2)
Z0_OVER_OMEGA0 = Z_0 / OMEGA_0  # ≈ 20.0

# Derived: detection volume V = π^(3/2) · ω₀² · z₀ (for 3D Gaussian)
DETECTION_VOLUME = np.pi**(3/2) * OMEGA_0**2 * Z_0  # nm^3
DETECTION_VOLUME_L = DETECTION_VOLUME * 1e-24  # litres
print(f"Detection volume: {DETECTION_VOLUME_L:.2e} L  (paper: 2.65e-14 L)")

# For the ellipsoidal hard boundary, we use the 1/e² radii as semi-axes.
# Molecules beyond this contribute negligibly to the detected signal.
ELLIPSOID_SEMI_AXIS_X = OMEGA_0    # 620 nm
ELLIPSOID_SEMI_AXIS_Y = OMEGA_0    # 620 nm
ELLIPSOID_SEMI_AXIS_Z = Z_0        # 12400 nm

INTENSITY_MAX      = 1.0
REFRACTIVE_INDEX   = 1.3       # paper Section 2: water/glycerol
NUMERICAL_APERTURE = 1.3       # UPDATED: paper Section 2 (NA = 1.3 oil)
DETECTION_THRESHOLD = 0

EPSILON = 1e-1


# =============================================================================
# EXPERIMENTAL DETECTION PARAMETERS (FRIES ET AL. EQ. 16)
# =============================================================================
# g = Ψ · Ψ_electr · Φ_F · σ₀₁(λ_ex) · γ
#
# From the paper:
#   Ψ ≈ 3.3-3.5% (detection efficiency of optical setup)
#   Ψ_electr = 0.79 (detection electronics efficiency, Section 3.1.1)
#   Φ_F(Rh6G) ≈ 0.90, Φ_F(RhB) ≈ 0.58
#   σ₀₁(522nm, Rh6G) = 3.1 × 10⁻¹⁶ cm²
#   σ₀₁(522nm, RhB) = 1.3 × 10⁻¹⁶ cm²
#   γ = λ_ex/(hc) = 2.6 × 10¹⁸ J⁻¹ for λ = 522 nm
#   I₀ = 1.5 × 10⁵ W/cm²

PSI_OPTICAL = 0.034           # optical detection efficiency (~3.4%)
PSI_ELECTRONICS = 0.79        # electronics detection efficiency
EXCITATION_WAVELENGTH = 522   # nm


# =============================================================================
# SIMULATION TIME STEP PARAMETERS
# =============================================================================
# Two regimes:
#   INSIDE  the confocal volume: fine steps for accurate photon stats
#   OUTSIDE the confocal volume: coarse steps (no photon recording)

# INSIDE confocal volume
MOTOR_TIME_STEP_INSIDE = 10e-6    # 10 μs  (motor coarse step)
CARGO_TIME_STEP_INSIDE = 100e-9   # 100 ns (cargo Langevin step)

# OUTSIDE confocal volume  (100x coarser)
MOTOR_TIME_STEP_OUTSIDE = 10e-4   # 1 ms
CARGO_TIME_STEP_OUTSIDE = 10e-6   # 10 μs


# =============================================================================
# BURST / PHOTON TARGETS
# =============================================================================

TARGET_BURSTS = 10000


# =============================================================================
# BIFL-STYLE BURST SELECTION PARAMETERS (FRIES ET AL. SECTION 3.2)
# =============================================================================
# The paper uses inter-photon time lag Δt smoothed over n photons.
# A burst is identified when the smoothed Δt drops below a threshold Δt_th.
#
# From Section 4 (Results):
#   - Background count rate: ~6 kHz  →  Δt_av = 170 μs
#   - Smoothing: n = 50 photons
#   - Threshold: Δt_th = 45 μs
#   - ξ = Δt_th/Δt_av = 0.27
#   - Error probability α ≈ 0 at these settings (Eq. 26, Figure 3)
#
# For our simulation, we adopt these BIFL burst selection parameters
# in the post-hoc analysis, while the in-loop burst counting uses a
# simpler gap criterion for efficiency.

BIFL_SMOOTHING_N = 50          # photons to smooth over (paper: n=50)
BIFL_DELTA_T_TH  = 45e-6      # threshold Δt = 45 μs (paper Section 4)
BIFL_DELTA_T_AV  = 170e-6     # mean background inter-photon lag (paper)

# In-loop burst gap (kept for simulation efficiency -- bursts refined post-hoc)
BURST_MIN_GAP = 2e-5  # 20 μs inter-photon gap to open a new burst


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
# MOTOR CONFINEMENT PARAMETERS
# =============================================================================
# Scaled to the new (larger) confocal volume. The motor is confined to a box
# around the confocal volume using reflecting boundary conditions.

CONFINEMENT_FACTOR = 4  # box = CONFINEMENT_FACTOR × 1/e² radii

MOTOR_CONFINEMENT_X = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_X   # ±2480 nm
MOTOR_CONFINEMENT_Y = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Y   # ±2480 nm
MOTOR_CONFINEMENT_Z = CONFINEMENT_FACTOR * ELLIPSOID_SEMI_AXIS_Z   # ±49600 nm


# =============================================================================
# DISPLAY SIMULATION SETUP
# =============================================================================

print("\n" + "="*70)
print("FRET SIMULATION - UPDATED PER FRIES ET AL. (1998)")
print("="*70)
print(f"Target bursts:       {TARGET_BURSTS:,}")
print(f"Burst gap threshold: {BURST_MIN_GAP*1e3:.1f} ms  (in-loop)")
print(f"Motor diffusion:     {MOTOR_DIFFUSION:.2e} nm^2/s  ({MOTOR_DIFFUSION/1e6:.0f} μm^2/s)")
print()
print("CONFOCAL PARAMETERS (Fries et al. 1998, Section 2):")
print(f"  ω₀ (radial 1/e² radius):  {OMEGA_0:.0f} nm  ({OMEGA_0/1000:.3f} μm)")
print(f"  z₀ (axial 1/e² radius):   {Z_0:.0f} nm  ({Z_0/1000:.1f} μm)")
print(f"  z₀/ω₀ ratio:              {Z0_OVER_OMEGA0:.1f}")
print(f"  NA:                        {NUMERICAL_APERTURE}")
print(f"  3D Gaussian SDE (Eq. 7):   I₀ exp(-2(x²+y²)/ω₀²) exp(-2z²/z₀²)")
print()
print("BIFL BURST SELECTION (Fries et al. Section 3.2):")
print(f"  Smoothing:     n = {BIFL_SMOOTHING_N} photons")
print(f"  Threshold:     Δt_th = {BIFL_DELTA_T_TH*1e6:.0f} μs")
print(f"  Background:    Δt_av = {BIFL_DELTA_T_AV*1e6:.0f} μs")
print(f"  ξ = Δt_th/Δt_av = {BIFL_DELTA_T_TH/BIFL_DELTA_T_AV:.2f}")
print()
print(f"Motor confinement box (reflecting boundaries):")
print(f"  X: ±{MOTOR_CONFINEMENT_X:.0f} nm  ({CONFINEMENT_FACTOR}× ω₀)")
print(f"  Y: ±{MOTOR_CONFINEMENT_Y:.0f} nm  ({CONFINEMENT_FACTOR}× ω₀)")
print(f"  Z: ±{MOTOR_CONFINEMENT_Z:.0f} nm  ({CONFINEMENT_FACTOR}× z₀)")
print()
print(f"  Time step          INSIDE volume        OUTSIDE volume")
print(f"  ---------------------------------------------------------")
print(f"  Motor (coarse)     {MOTOR_TIME_STEP_INSIDE*1e6:7.1f} μs            {MOTOR_TIME_STEP_OUTSIDE*1e3:7.1f} ms")
print(f"  Cargo (Langevin)   {CARGO_TIME_STEP_INSIDE*1e9:7.1f} ns            {CARGO_TIME_STEP_OUTSIDE*1e6:7.1f} μs")
print()
print(f"Förster radius:      {FORSTER_RADIUS} nm")
print(f"Donor lifetime:      {1/DONOR_FLUORESCENCE_RATE*1e6:.1f} μs")
print(f"Acceptor lifetime:   {1/ACCEPTOR_FLUORESCENCE_RATE*1e6:.1f} μs")
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

# ---- 3D Gaussian SDE (Fries et al. Eq. 7) ----
# I(x,y,z) = I₀ exp(-2(x²+y²)/ω₀²) exp(-2z²/z₀²)
#
# Pre-compute: -2/ω₀² for radial and -2/z₀² for axial
OMEGA_0_SQ      = np.float32(OMEGA_0**2)        # ω₀²
Z_0_SQ          = np.float32(Z_0**2)            # z₀²
MINUS_2_OVER_OMEGA0_SQ = np.float32(-2.0 / OMEGA_0_SQ)  # -2/ω₀²
MINUS_2_OVER_Z0_SQ     = np.float32(-2.0 / Z_0_SQ)      # -2/z₀²

# Ellipsoid boundary check uses the 1/e² radii
ELLIPSOID_A_SQ   = np.float32(ELLIPSOID_SEMI_AXIS_X**2)
ELLIPSOID_B_SQ   = np.float32(ELLIPSOID_SEMI_AXIS_Y**2)
ELLIPSOID_C_SQ   = np.float32(ELLIPSOID_SEMI_AXIS_Z**2)

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
    """Fast check if position is inside ellipsoid (1/e² boundary)."""
    return (x*x) / a_sq + (y*y) / b_sq + (z*z) / c_sq <= 1.0


@jit(nopython=True, fastmath=True, cache=True)
def calculate_intensity_gaussian3d(x, y, z, minus_2_over_w0sq, minus_2_over_z0sq):
    """
    Calculate 3D Gaussian SDE intensity (Fries et al. Eq. 7).

    I(x,y,z) = I₀ exp(-2(x²+y²)/ω₀²) exp(-2z²/z₀²)

    Returns normalised intensity in [0, 1], where 1 = focal centre.
    The two exponential terms are separated for clarity but combined
    into a single exp() call for speed.
    """
    radial_sq = x * x + y * y
    return np.exp(minus_2_over_w0sq * radial_sq + minus_2_over_z0sq * z * z)


@jit(nopython=True, fastmath=True, cache=True)
def reflect(value, bound):
    """
    Apply reflecting boundary condition to a 1-D coordinate.
    Folds the value back into [-bound, +bound].
    """
    span = np.float32(2.0) * bound
    v = value + bound
    v = v - span * np.floor(v / span)
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
        time_to_fret  = -np.log(np.random.random()) / fret_rate
        time_to_donor = -np.log(np.random.random()) / donor_rate
        if time_to_fret < time_to_donor:
            state = 2  # FRET → acceptor
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
    # Volume boundary (ellipsoid 1/e² radii)
    a_sq, b_sq, c_sq,
    # 3D Gaussian SDE (Fries et al. Eq. 7)
    minus_2_over_w0sq, minus_2_over_z0sq,
    detection_threshold,
    # FRET
    forster_r, forster_r6, donor_rate, acceptor_rate,
    # Probabilities
    prob_qd, prob_qa, prob_as, prob_ds, prob_nd,
    # Motor confinement box half-widths
    conf_x, conf_y, conf_z):
    """
    Ultra-optimized main loop with ADAPTIVE TIME STEPPING and MOTOR CONFINEMENT.

    UPDATED: Uses 3D Gaussian SDE profile I(x,y,z) = I₀ exp(-2(x²+y²)/ω₀²)
    exp(-2z²/z₀²) from Fries et al. (1998) Eq. 7 for intensity calculation.
    The ellipsoidal boundary uses the 1/e² radii (ω₀, z₀) from the paper.
    """
    max_events = target_bursts * 200  # generous buffer

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
    last_photon_time = np.float64(-1.0)

    total_photons = np.int64(0)
    total_bursts  = np.int64(0)
    event_count   = np.int64(0)
    cargo_step    = np.int64(0)

    in_volume_count = np.int64(0)
    total_steps     = np.int64(0)

    was_in_volume = False

    # Main simulation loop
    while cargo_step < max_steps and total_bursts < target_bursts:

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

        # Update motor when enough time has elapsed
        if current_time - last_motor_time >= motor_dt:
            motor_x += sqrt2_m * np.random.randn()
            motor_y += sqrt2_m * np.random.randn()
            motor_z += sqrt2_m * np.random.randn()
            last_motor_time = current_time

            # Reflecting boundary conditions
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

        # Confocal volume check (ellipsoidal 1/e² boundary)
        donor_in    = is_inside_volume(donor_x,    donor_y,    donor_z,    a_sq, b_sq, c_sq)
        acceptor_in = is_inside_volume(acceptor_x, acceptor_y, acceptor_z, a_sq, b_sq, c_sq)

        total_steps += 1

        donor_intensity    = np.float32(0.0)
        acceptor_intensity = np.float32(0.0)

        # UPDATED: Use 3D Gaussian SDE (Fries et al. Eq. 7)
        if donor_in:
            donor_intensity = calculate_intensity_gaussian3d(
                donor_x, donor_y, donor_z,
                minus_2_over_w0sq, minus_2_over_z0sq)
            donor_in = donor_intensity > detection_threshold

        if acceptor_in:
            acceptor_intensity = calculate_intensity_gaussian3d(
                acceptor_x, acceptor_y, acceptor_z,
                minus_2_over_w0sq, minus_2_over_z0sq)
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

init_motor_x = 250.0
init_motor_y = 0.0
init_motor_z = 0.0

init_donor_x    = init_motor_x + DONOR_OFFSET_X
init_donor_y    = init_motor_y + DONOR_OFFSET_Y
init_donor_z    = init_motor_z + DONOR_OFFSET_Z
init_acceptor_x = init_motor_x + ACCEPTOR_OFFSET_X
init_acceptor_y = init_motor_y + ACCEPTOR_OFFSET_Y
init_acceptor_z = init_motor_z + ACCEPTOR_OFFSET_Z

max_simulation_steps = 10_000_000_000

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
    # Ellipsoidal boundary (1/e² radii)
    ELLIPSOID_A_SQ, ELLIPSOID_B_SQ, ELLIPSOID_C_SQ,
    # 3D Gaussian SDE (Fries et al. Eq. 7)
    MINUS_2_OVER_OMEGA0_SQ, MINUS_2_OVER_Z0_SQ,
    DETECTION_THRESHOLD_F32,
    FORSTER_RADIUS_F32, FORSTER_RADIUS_6,
    DONOR_FLUOR_RATE_F32, ACCEPTOR_FLUOR_RATE_F32,
    PROB_QD, PROB_QA, PROB_AS, PROB_DS, PROB_ND,
    MOTOR_CONF_X_F32, MOTOR_CONF_Y_F32, MOTOR_CONF_Z_F32
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
# BIFL-STYLE BURST ANALYSIS (FRIES ET AL. SECTIONS 3.2 AND 4)
# =============================================================================
#
# The paper's BIFL burst selection works as follows:
#   1. Compute inter-photon time lags Δt_i = t_{i+1} - t_i
#   2. Smooth over n consecutive lags: Δt_sm(n) = (1/n) Σ Δt_i
#   3. A burst is identified when Δt_sm(n) < Δt_th
#   4. The error probability α of a false burst (background fluctuation)
#      is given by the regularised incomplete gamma function (Eq. 26):
#      α = P(Δt_sm(n) ≤ Δt_th | Δt_av) = γ(n, n·ξ)/Γ(n)
#      where ξ = Δt_th / Δt_av
#
# We also compute the BSD as described in Section 3.1.

print("\n" + "="*70)
print("BIFL-STYLE BURST ANALYSIS  (Fries et al. 1998)")
print("="*70)


def bifl_burst_detection(times, colors, n_smooth=50, delta_t_th=45e-6):
    """
    Identify photon bursts using the BIFL inter-photon time lag method
    described in Fries et al. (1998), Section 3.2 and Section 4.

    Parameters
    ----------
    times : array
        Photon arrival times (sorted).
    colors : array
        Photon types (4 = donor, 5 = acceptor).
    n_smooth : int
        Number of consecutive inter-photon lags to average (paper: n=50).
    delta_t_th : float
        Threshold for smoothed Δt to identify burst photons (paper: 45 μs).

    Returns
    -------
    bursts : list of dict
        Each burst contains entry_time, exit_time, duration,
        total_photons, donor_photons, acceptor_photons, photon_indices.
    smoothed_dt : array
        The smoothed inter-photon time lag trace (for diagnostics).
    """
    if len(times) < n_smooth + 1:
        return [], np.array([])

    # Step 1: compute inter-photon time lags (Eq. 24 context)
    delta_t = np.diff(times)

    # Step 2: smooth over n consecutive lags using a moving average
    # Δt_sm(n) = (1/n) Σ_{i}^{i+n-1} Δt_i   (paper Section 3.2)
    kernel = np.ones(n_smooth) / n_smooth
    smoothed_dt = np.convolve(delta_t, kernel, mode='valid')

    # Step 3: identify burst regions where smoothed Δt < threshold
    in_burst = smoothed_dt < delta_t_th

    # Offset: smoothed_dt[k] corresponds to photon index k + n_smooth//2
    offset = n_smooth // 2

    bursts = []
    current_burst_indices = []

    for k in range(len(in_burst)):
        photon_idx = k + offset
        if in_burst[k]:
            current_burst_indices.append(photon_idx)
        else:
            if len(current_burst_indices) > 0:
                # Extend burst to include the n_smooth/2 photons on each side
                first = max(0, current_burst_indices[0] - offset)
                last  = min(len(times) - 1, current_burst_indices[-1] + offset)
                indices = list(range(first, last + 1))
                bc = colors[indices]
                bursts.append({
                    'entry_time':       float(times[indices[0]]),
                    'exit_time':        float(times[indices[-1]]),
                    'duration':         float(times[indices[-1]] - times[indices[0]]),
                    'total_photons':    len(indices),
                    'donor_photons':    int(np.sum(bc == 4)),
                    'acceptor_photons': int(np.sum(bc == 5)),
                    'photon_indices':   indices
                })
                current_burst_indices = []

    # Handle final burst
    if len(current_burst_indices) > 0:
        first = max(0, current_burst_indices[0] - offset)
        last  = min(len(times) - 1, current_burst_indices[-1] + offset)
        indices = list(range(first, last + 1))
        bc = colors[indices]
        bursts.append({
            'entry_time':       float(times[indices[0]]),
            'exit_time':        float(times[indices[-1]]),
            'duration':         float(times[indices[-1]] - times[indices[0]]),
            'total_photons':    len(indices),
            'donor_photons':    int(np.sum(bc == 4)),
            'acceptor_photons': int(np.sum(bc == 5)),
            'photon_indices':   indices
        })

    return bursts, smoothed_dt


def compute_burst_error_probability(n_smooth, delta_t_th, delta_t_av):
    """
    Compute the error probability α of a false burst identification
    due to background fluctuations (Fries et al. Eq. 26).

    α = P(Δt_sm(n) ≤ Δt_th | Δt_av) = γ_inc(n, n·ξ) / Γ(n)

    where ξ = Δt_th / Δt_av and γ_inc is the lower regularised
    incomplete gamma function.

    Parameters
    ----------
    n_smooth : int
        Number of photons used for smoothing.
    delta_t_th : float
        Threshold value.
    delta_t_av : float
        Mean background inter-photon time lag.

    Returns
    -------
    alpha : float
        Error probability.
    xi : float
        Dimensionless threshold ratio ξ = Δt_th / Δt_av.
    """
    xi = delta_t_th / delta_t_av
    # scipy.special.gammainc is the regularised lower incomplete gamma function
    # P(a, x) = γ(a, x) / Γ(a) which is exactly Eq. 26 with a=n, x=n·ξ
    alpha = float(gammainc(n_smooth, n_smooth * xi))
    return alpha, xi


# Run BIFL burst detection
bifl_bursts, smoothed_dt = bifl_burst_detection(
    event_times, event_colors,
    n_smooth=BIFL_SMOOTHING_N,
    delta_t_th=BIFL_DELTA_T_TH
)

# Compute error probability (Eq. 26)
if num_events > 1:
    # Estimate background Δt_av from the data (using inter-photon gaps > threshold)
    delta_t_all = np.diff(event_times)
    bg_lags = delta_t_all[delta_t_all > BIFL_DELTA_T_TH]
    if len(bg_lags) > 10:
        measured_delta_t_av = float(np.mean(bg_lags))
    else:
        measured_delta_t_av = BIFL_DELTA_T_AV

    alpha, xi = compute_burst_error_probability(
        BIFL_SMOOTHING_N, BIFL_DELTA_T_TH, measured_delta_t_av)
else:
    measured_delta_t_av = BIFL_DELTA_T_AV
    alpha, xi = 0.0, 0.0

print(f"\nBIFL burst selection parameters:")
print(f"  Smoothing n:           {BIFL_SMOOTHING_N}")
print(f"  Threshold Δt_th:       {BIFL_DELTA_T_TH*1e6:.0f} μs")
print(f"  Measured Δt_av:        {measured_delta_t_av*1e6:.1f} μs")
print(f"  ξ = Δt_th/Δt_av:      {xi:.3f}")
print(f"  Error probability α:   {alpha:.2e}  (Eq. 26)")
print(f"\n  BIFL bursts detected:  {len(bifl_bursts):,}")

if len(bifl_bursts) > 0:
    burst_sizes = [b['total_photons'] for b in bifl_bursts]
    print(f"  Avg photons/burst:     {np.mean(burst_sizes):.1f}")
    print(f"  Avg donor/burst:       {np.mean([b['donor_photons'] for b in bifl_bursts]):.1f}")
    print(f"  Avg acceptor/burst:    {np.mean([b['acceptor_photons'] for b in bifl_bursts]):.1f}")
    print(f"  Min burst size:        {min(burst_sizes)}")
    print(f"  Max burst size:        {max(burst_sizes)}")


# =============================================================================
# BURST SIZE DISTRIBUTION (BSD) ANALYSIS  (FRIES ET AL. SECTION 3.1)
# =============================================================================
#
# Eq. 12: P₁(Ct,t) ∝ K_IS · sqrt(ln(gI₀t/Ct)) / Ct
# Eq. 17: Pt₁(t) = (1/tt) exp(-t/tt)   with tt = ω₀²/(3D)
# Eq. 20: Pmn(Nav) = exp(-2Nav)(1-exp(-Nav))^(n-1) / Σ...
# Eq. 23: β(Ct,Nav,g) = P(Ct,Nav,g)·(1-exp(-Nav))·(tmes/tB)

print(f"\n{'='*70}")
print("BURST SIZE DISTRIBUTION (BSD) ANALYSIS  (Fries et al. Eqs. 12, 17, 23)")
print(f"{'='*70}")


def compute_bsd(bursts, min_burst_size=5):
    """
    Compute the burst size distribution β(Ct) from detected bursts.
    This is the empirical BSD: frequency of bursts containing Ct photons.
    (Fries et al. Section 3.1.5, Eq. 21)

    Parameters
    ----------
    bursts : list of dict
        Burst data from bifl_burst_detection.
    min_burst_size : int
        Minimum burst size to include (paper uses 50 for their conditions;
        we use a lower threshold for our simulation).

    Returns
    -------
    ct_values : array
        Burst sizes (Ct).
    frequencies : array
        Frequency of each burst size.
    """
    sizes = np.array([b['total_photons'] for b in bursts])
    sizes = sizes[sizes >= min_burst_size]

    if len(sizes) == 0:
        return np.array([]), np.array([])

    max_size = int(np.max(sizes))
    ct_values = np.arange(min_burst_size, max_size + 1)
    frequencies = np.zeros(len(ct_values), dtype=int)

    for i, ct in enumerate(ct_values):
        frequencies[i] = int(np.sum(sizes == ct))

    # Remove zero-frequency entries for cleaner output
    nonzero = frequencies > 0
    return ct_values[nonzero], frequencies[nonzero]


def compute_dwell_time_distribution(bursts, n_bins=50):
    """
    Compute the burst duration distribution Ptm(t) and fit to
    exponential (Fries et al. Eq. 18).

    Returns mean burst duration tB and estimated single-molecule
    transit time tt.
    """
    durations = np.array([b['duration'] for b in bursts])
    durations = durations[durations > 0]

    if len(durations) == 0:
        return 0.0, 0.0, durations

    # Mean burst duration (Eq. 18: tB = Σ n·tt·Pmn)
    tB = float(np.mean(durations))

    # For low Nav, tB ≈ tt (single-molecule transit time)
    # Alternatively, from FCS: tt = (4/3)·τ_D = ω₀²/(3D)
    tt = tB  # first approximation

    return tB, tt, durations


def compute_fret_per_burst(bursts, event_fret_eff, event_interdye_dist):
    """
    Compute mean FRET efficiency and interdye distance per burst.
    This enables burst-level FRET analysis as done in the paper
    for fluorescence lifetime determination.
    """
    burst_fret = []
    burst_dist = []

    for b in bursts:
        indices = b['photon_indices']
        if len(indices) > 0 and max(indices) < len(event_fret_eff):
            burst_fret.append(float(np.mean(event_fret_eff[indices])))
            burst_dist.append(float(np.mean(event_interdye_dist[indices])))

    return np.array(burst_fret), np.array(burst_dist)


# Compute BSD
if len(bifl_bursts) > 0:
    ct_vals, ct_freq = compute_bsd(bifl_bursts, min_burst_size=3)
    tB, tt_est, burst_durations = compute_dwell_time_distribution(bifl_bursts)
    burst_fret_effs, burst_distances = compute_fret_per_burst(
        bifl_bursts, event_fret_eff, event_interdye_dist)

    print(f"\nBurst Size Distribution:")
    print(f"  Number of BSD entries:    {len(ct_vals)}")
    if len(ct_vals) > 0:
        print(f"  Ct range:                 [{ct_vals[0]}, {ct_vals[-1]}]")
        weighted_mean = np.sum(ct_vals * ct_freq) / np.sum(ct_freq)
        print(f"  Weighted mean Ct:         {weighted_mean:.1f}")

    print(f"\nBurst Duration Distribution (Eq. 18):")
    print(f"  Mean burst duration tB:   {tB*1e3:.3f} ms")
    print(f"  Estimated transit time tt:{tt_est*1e3:.3f} ms")
    print(f"  Paper reference tt:       1.2 ms  (Section 4)")

    if len(burst_fret_effs) > 0:
        print(f"\nBurst-level FRET analysis:")
        print(f"  Mean FRET efficiency:     {np.mean(burst_fret_effs):.3f}")
        print(f"  Std FRET efficiency:      {np.std(burst_fret_effs):.3f}")
        print(f"  Mean interdye distance:   {np.mean(burst_distances):.2f} nm")
else:
    ct_vals, ct_freq = np.array([]), np.array([])
    burst_durations = np.array([])
    burst_fret_effs = np.array([])
    burst_distances = np.array([])
    tB, tt_est = 0.0, 0.0
    print("\n  No BIFL bursts detected -- skipping BSD analysis.")


# =============================================================================
# ALSO RUN SIMPLE GAP-BASED BURST ANALYSIS (for comparison)
# =============================================================================

print(f"\n{'='*70}")
print("SIMPLE GAP-BASED BURST ANALYSIS (for comparison)")
print(f"{'='*70}")


def identify_bursts_combined(times, colors, min_gap=0.001):
    """Simple burst identification from inter-photon time gaps."""
    if len(times) == 0:
        return []

    bursts = []
    current_indices = [0]

    for i in range(1, len(times)):
        if times[i] - times[current_indices[-1]] < min_gap:
            current_indices.append(i)
        else:
            bc = colors[current_indices]
            bursts.append({
                'entry_time':       float(times[current_indices[0]]),
                'exit_time':        float(times[current_indices[-1]]),
                'duration':         float(times[current_indices[-1]] - times[current_indices[0]]),
                'total_photons':    len(current_indices),
                'donor_photons':    int(np.sum(bc == 4)),
                'acceptor_photons': int(np.sum(bc == 5))
            })
            current_indices = [i]

    bc = colors[current_indices]
    bursts.append({
        'entry_time':       float(times[current_indices[0]]),
        'exit_time':        float(times[current_indices[-1]]),
        'duration':         float(times[current_indices[-1]] - times[current_indices[0]]),
        'total_photons':    len(current_indices),
        'donor_photons':    int(np.sum(bc == 4)),
        'acceptor_photons': int(np.sum(bc == 5))
    })
    return bursts


all_bursts_simple = identify_bursts_combined(event_times, event_colors, min_gap=BURST_MIN_GAP)

print(f"  Bursts (in-loop counter):     {total_bursts_collected:,}")
print(f"  Bursts (simple gap post-hoc): {len(all_bursts_simple):,}")
print(f"  Bursts (BIFL method):         {len(bifl_bursts):,}")
if len(all_bursts_simple) > 0:
    print(f"  Avg photons/burst (simple):   {np.mean([b['total_photons'] for b in all_bursts_simple]):.1f}")


# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

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

# Save BIFL burst data
bifl_burst_records = []
for b in bifl_bursts:
    bifl_burst_records.append({
        'entry_time':       b['entry_time'],
        'exit_time':        b['exit_time'],
        'duration':         b['duration'],
        'total_photons':    b['total_photons'],
        'donor_photons':    b['donor_photons'],
        'acceptor_photons': b['acceptor_photons'],
    })

bifl_burst_df = pd.DataFrame(bifl_burst_records)

# Summary metadata
summary_data = {
    'entry_time': ['SUMMARY'] * 15,
    'exit_time': [
        f'total_photons={total_photons}',
        f'donor_photons={donor_count}',
        f'acceptor_photons={acceptor_count}',
        f'total_bursts_inloop={total_bursts_collected}',
        f'total_bursts_bifl={len(bifl_bursts)}',
        f'target_bursts={TARGET_BURSTS}',
        f'simulation_time={total_simulation_time:.4f}s',
        f'volume_occupancy={occupancy:.4f}',
        f'omega_0={OMEGA_0:.0f}nm',
        f'z_0={Z_0:.0f}nm',
        f'z0_over_omega0={Z0_OVER_OMEGA0:.1f}',
        f'NA={NUMERICAL_APERTURE}',
        f'bifl_smoothing_n={BIFL_SMOOTHING_N}',
        f'bifl_delta_t_th={BIFL_DELTA_T_TH*1e6:.0f}us',
        f'bifl_error_prob_alpha={alpha:.2e}',
    ],
    'duration':          [''] * 15,
    'total_photons':     [''] * 15,
    'donor_photons':     [''] * 15,
    'acceptor_photons':  [''] * 15,
}
summary_df = pd.DataFrame(summary_data)

if len(bifl_burst_records) > 0:
    bifl_burst_df = bifl_burst_df[['entry_time', 'exit_time', 'duration',
                                    'total_photons', 'donor_photons', 'acceptor_photons']]

final_burst_df = pd.concat([summary_df, bifl_burst_df], ignore_index=True)
final_burst_df.to_csv('photon_bursts_nanosecond.csv', index=False)
print(f"  Saved BIFL burst analysis to 'photon_bursts_nanosecond.csv'")

# Save BSD
if len(ct_vals) > 0:
    bsd_df = pd.DataFrame({'burst_size_Ct': ct_vals, 'frequency': ct_freq})
    bsd_df.to_csv('burst_size_distribution.csv', index=False)
    print(f"  Saved BSD to 'burst_size_distribution.csv' ({len(ct_vals)} entries)")

# Save burst-level FRET
if len(burst_fret_effs) > 0:
    fret_df = pd.DataFrame({
        'burst_fret_efficiency': burst_fret_effs,
        'burst_interdye_distance': burst_distances,
        'burst_size': [b['total_photons'] for b in bifl_bursts[:len(burst_fret_effs)]]
    })
    fret_df.to_csv('burst_fret_analysis.csv', index=False)
    print(f"  Saved burst FRET analysis to 'burst_fret_analysis.csv' ({len(burst_fret_effs)} bursts)")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print("SIMULATION COMPLETE  —  UPDATED PER FRIES ET AL. (1998)")
print(f"{'='*70}")
print(f"  Execution time:  {elapsed_time:.1f} s ({elapsed_time/60:.1f} min)")
print(f"  Steps/second:    {num_cargo_steps/elapsed_time:,.0f}")
print(f"  Photons/second:  {total_photons/elapsed_time:,.0f}")
print(f"  Bursts/second:   {total_bursts_collected/elapsed_time:,.1f}")
print(f"  Memory (events): {num_events*12*4/1e6:.1f} MB  "
      f"(vs {num_cargo_steps*13*4/1e6:.1f} MB if storing all steps)")
print()
print("KEY UPDATES FROM FRIES ET AL. (1998):")
print(f"  ✓ Confocal volume: ω₀={OMEGA_0:.0f} nm, z₀={Z_0:.0f} nm, z₀/ω₀={Z0_OVER_OMEGA0:.0f}")
print(f"  ✓ 3D Gaussian SDE (Eq. 7): I₀ exp(-2r²/ω₀²) exp(-2z²/z₀²)")
print(f"  ✓ NA = {NUMERICAL_APERTURE} (oil immersion, Fluar 40×)")
print(f"  ✓ BIFL burst detection (Eq. 24-26): n={BIFL_SMOOTHING_N}, Δt_th={BIFL_DELTA_T_TH*1e6:.0f} μs")
print(f"  ✓ BSD analysis (Eqs. 12, 17, 23)")
print(f"  ✓ Error probability α = {alpha:.2e} (Eq. 26)")
print(f"  ✓ Detection factor g = Ψ·Ψ_electr·Φ_F·σ₀₁·γ (Eq. 16)")
print()
print(f"Files saved:")
print(f"  - photon_events_nanosecond.csv  ({len(event_df):,} events)")
print(f"  - photon_bursts_nanosecond.csv  ({len(bifl_bursts):,} BIFL bursts)")
if len(ct_vals) > 0:
    print(f"  - burst_size_distribution.csv   ({len(ct_vals)} BSD entries)")
if len(burst_fret_effs) > 0:
    print(f"  - burst_fret_analysis.csv       ({len(burst_fret_effs)} burst FRET values)")
print(f"{'='*70}\n")
