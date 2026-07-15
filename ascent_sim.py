# Companion ascent simulation that starts from the saved touchdown state
# produced by descent_sim.py and uses the remaining mass as the ascent-stage
# propellant budget.
#
# REWORKED GUIDANCE (why the old version never reached NRHO):
#
#   1. LOD phase deadlock: the control law servoed toward r_target -- the
#      NRHO *perilune* point at r = 3,366 km -- but the phase exit gate
#      waited for r >= 0.5 * APOLUNE = 35,500 km. The vehicle was pulled
#      toward a point 32,000 km inside the gate, so the phase never exited
#      and burned propellant (gravity cancellation) until depletion.
#
#   2. Circ phase deadlock: the exit gate required |r - LLO_RADIUS| < 2 km,
#      but the control law had no radial *position* feedback -- it only
#      damped radial rate. Null the radial rate 5 km off the target radius
#      and the gate never trips.
#
#   3. Gravity cancellation everywhere: every powered phase commanded
#      -g_vec continuously. Hover-thrusting through a multi-hour transfer
#      costs tens of km/s of delta-V; no realistic wet/dry ratio survives.
#
# The rework replaces the continuous force-servo phases with a standard
# burn + coast architecture:
#
#   liftoff  - pitch-over ascent (unchanged; it worked)
#   circ     - circularize at the achieved radius (rate + speed feedback,
#              no gravity cancellation, gate is achievable)
#   loiter   - 2 h zero-thrust coast in LLO
#   lod      - finite prograde burn to raise apoapsis to NRHO perilune
#              radius (Hohmann-style), then cut thrust
#   transfer - zero-thrust coast up the transfer ellipse (~100 min)
#   nri      - finite burn near apoapsis to match NRHO perilune velocity
#   success  - verified from achieved orbital elements (rp/ra/plane), not
#              distance to a static point Orion wouldn't actually be at
import json
import os
import matplotlib.pyplot as plt
import yaml
import numpy as np
from mpl_toolkits import mplot3d as plt3  # noqa: F401
from Basilisk import __path__
from Basilisk.simulation import spacecraft, extForceTorque
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    vizSupport,
)
from lola_terrain import get_surface_radius_m, MEAN_RADIUS_M

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

MU_MOON = 4.9048695e12
MOON_SPIN_RATE = 2.0 * np.pi / (27.321661 * 86400.0)
G0 = 9.80665

NRHO_PERILUNE_RADIUS_M = 3366.0e3
NRHO_APOLUNE_RADIUS_M = 71000.0e3
NRHO_SMA_M = (NRHO_PERILUNE_RADIUS_M + NRHO_APOLUNE_RADIUS_M) / 2.0
NRHO_INCLINATION_DEG = 90.0
NRHO_OMEGA_DEG = 90.0

LLO_RADIUS_M = MEAN_RADIUS_M + 100.0e3


def load_touchdown_state(state_path):
    with open(state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "mission_config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_local_radius(r_inertial, t, dem_path):
    """Account for lunar rotation when querying the DEM."""
    theta = MOON_SPIN_RATE * t
    c, s = np.cos(-theta), np.sin(-theta)
    x, y, z = r_inertial
    xf = c * x - s * y
    yf = s * x + c * y
    zf = z
    r = np.linalg.norm([xf, yf, zf])
    lat_deg = np.degrees(np.arcsin(np.clip(zf / r, -1.0, 1.0)))
    lon_deg = np.degrees(np.arctan2(yf, xf))
    radius_m, source = get_surface_radius_m(lat_deg, lon_deg, dem_path)
    return radius_m, lat_deg, lon_deg, source


def build_fallback_state(ascent_mass):
    radius_m = MEAN_RADIUS_M + 1.0
    return {
        "ascent_stage_dry_mass_kg": float(ascent_mass),
        "ascent_stage_wet_mass_kg": float(ascent_mass),
        "ascent_stage_final_mass_kg": float(ascent_mass),
        "r_surface_m": [0.0, radius_m, 0.0],
        "v_surface_mps": [0.0, 0.0, 0.0],
        "time_s": 0.0,
        "latitude_deg": -89.45,
        "longitude_deg": 222.0,
        "surface_source": "fallback",
    }


def run(
    show_plots,
    dem_path,
    ascent_mass,
    ascent_inertia,
    thrust_N,
    num_thrusters,
    isp_s,
    sim_time_step_s,
    print_interval_s,
    max_jerk,
    blend_duration_s,  # retained for signature compatibility (unused)
    state_path=None,
):
    if state_path is None:
        state_path = os.path.join(
            os.path.dirname(__file__),
            "descent_state.json",
        )
    config = load_config()
    ascent_cfg = config.get("mission", {}).get("stages", {}).get("ascent", {})
    if not ascent_cfg:
        ascent_cfg = {}
    ascent_dry_mass_cfg = float(ascent_cfg.get("dry_mass_kg", ascent_mass))
    ascent_wet_mass_cfg = float(ascent_cfg.get("wet_mass_kg", ascent_mass))
    ascent_mass = ascent_wet_mass_cfg if ascent_wet_mass_cfg > 0.0 else ascent_dry_mass_cfg
    thrust_N = float(ascent_cfg.get("thrust_N", thrust_N))
    num_thrusters = int(ascent_cfg.get("num_thrusters", num_thrusters))
    try:
        state = load_touchdown_state(state_path)
    except FileNotFoundError:
        print(
            f"State file not found at {state_path}; "
            "using a fallback surface state."
        )
        state = build_fallback_state(ascent_mass)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(
            f"State file at {state_path} is invalid ({exc}); "
            "using a fallback surface state."
        )
        state = build_fallback_state(ascent_mass)
    ascent_dry_mass = float(state.get("ascent_stage_dry_mass_kg", ascent_dry_mass_cfg))
    ascent_wet_mass = float(state.get("ascent_stage_wet_mass_kg", ascent_wet_mass_cfg))
    descent_final_mass = float(
        state.get(
            "descent_stage_final_mass_kg",
            state.get("final_mass_kg", ascent_wet_mass),
        )
    )
    current_mass = max(ascent_wet_mass if ascent_wet_mass > 0.0 else descent_final_mass, 1.0)

    print("Loaded touchdown state from", state_path)
    print(f"  descent dry mass: {state.get('descent_stage_dry_mass_kg', state.get('final_mass_kg', 0.0)):.1f} kg")
    print(f"  descent wet mass: {state.get('descent_stage_wet_mass_kg', state.get('final_mass_kg', 0.0)):.1f} kg")
    print(f"  ascent dry mass: {ascent_dry_mass:.1f} kg")
    print(f"  ascent wet mass: {ascent_wet_mass:.1f} kg")
    print(
        "  touchdown lat/lon: "
        f"{state['latitude_deg']:.3f}, "
        f"{state['longitude_deg']:.3f} deg"
    )

    # -- Propellant feasibility check ------------------------------------
    # Rough budget: ~2.6 km/s ascent+circ (incl. gravity losses at ~5 m/s^2),
    # ~0.25 km/s LOD, ~0.7 km/s NRI  =>  ~3.5 km/s plus margin.
    dv_required_est = 3500.0
    if current_mass > ascent_dry_mass + 1e-6:
        dv_available = isp_s * G0 * np.log(current_mass / ascent_dry_mass)
    else:
        dv_available = 0.0
    print(f"  delta-V available: {dv_available:.0f} m/s "
          f"(rough requirement ~{dv_required_est:.0f} m/s)")
    if dv_available <= 0.0:
        print(
            "[error] No ascent propellant: wet mass <= dry mass. "
            "Check mission_config.yaml ascent stage masses and/or "
            "descent_state.json ascent_stage_*_mass_kg fields."
        )
    elif dv_available < dv_required_est:
        print(
            f"[warning] Available delta-V ({dv_available:.0f} m/s) is below "
            f"the rough requirement ({dv_required_est:.0f} m/s); the ascent "
            "will likely run dry before NRHO insertion."
        )

    r0 = np.asarray(
        state.get(
            "r_surface_m",
            [0.0, MEAN_RADIUS_M + 1.0, 0.0],
        ),
        dtype=float,
    ).reshape(3)
    v0 = np.asarray(
        state.get(
            "v_surface_mps",
            [0.0, 0.0, 0.0],
        ),
        dtype=float,
    ).reshape(3)
    r0_magnitude = np.linalg.norm(r0)
    if r0_magnitude <= 0.0:
        raise ValueError("Touchdown position has zero magnitude.")
    # Start 0.5 m above the saved touchdown position so that small
    # DEM/numerical differences do not place the vehicle underground.
    r0_hat = r0 / r0_magnitude
    r0 = r0 + 0.5 * r0_hat
    dt = float(sim_time_step_s)
    if dt <= 0.0:
        raise ValueError("sim_time_step_s must be greater than zero.")

    sim_task_name = "simTask"
    sc_sim = SimulationBaseClass.SimBaseClass()
    dyn_process = sc_sim.CreateNewProcess("dynProcess", 0)
    dyn_process.addTask(
        sc_sim.CreateNewTask(
            sim_task_name,
            macros.sec2nano(dt),
        ),
        10,
    )
    grav_factory = simIncludeGravBody.gravBodyFactory()
    moon = grav_factory.createMoon()
    moon.isCentralBody = True
    moon.mu = MU_MOON
    lander = spacecraft.Spacecraft()
    lander.ModelTag = "ascentLander"
    lander.hub.mHub = current_mass
    lander.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    # FIX: Basilisk.utilities has no `simHelpers` module; build the 3x3
    # inertia matrix directly from the 9 row-major values.
    inertia = list(ascent_inertia)
    lander.hub.IHubPntBc_B = [inertia[0:3], inertia[3:6], inertia[6:9]]
    lander.gravField.gravBodies = spacecraft.GravBodyVector(
        list(grav_factory.gravBodies.values())
    )
    sc_sim.AddModelToTask(
        sim_task_name,
        lander,
        None,
        1,
    )
    ascent_thruster = extForceTorque.ExtForceTorque()
    ascent_thruster.ModelTag = "ascentThruster"
    lander.addDynamicEffector(ascent_thruster)
    sc_sim.AddModelToTask(sim_task_name, ascent_thruster)

    # Representative two-body ellipse with NRHO-like perilune and apolune.
    # Used only to define the target orbital PLANE and target elements;
    # success is judged on achieved elements, not on reaching a static
    # point (the old r_error/v_error-to-a-frozen-perilune criterion could
    # not be met without phasing control and is not what "reached NRHO"
    # means anyway).
    oe = orbitalMotion.ClassicElements()
    oe.a = NRHO_SMA_M
    oe.e = (
        NRHO_APOLUNE_RADIUS_M - NRHO_PERILUNE_RADIUS_M
    ) / (
        NRHO_APOLUNE_RADIUS_M + NRHO_PERILUNE_RADIUS_M
    )
    oe.i = NRHO_INCLINATION_DEG * macros.D2R
    oe.Omega = 0.0
    oe.omega = NRHO_OMEGA_DEG * macros.D2R
    oe.f = 0.0
    r_target, v_target = orbitalMotion.elem2rv(
        MU_MOON,
        oe,
    )
    r_target = np.asarray(r_target, dtype=float).flatten()
    v_target = np.asarray(v_target, dtype=float).flatten()
    # Orbit-normal of the target NRHO plane; used to build a consistent
    # prograde tangential direction at any position along the ascent.
    h_target = np.cross(r_target, v_target)
    h_target_norm = np.linalg.norm(h_target)
    if h_target_norm < 1e-9:
        h_target_hat = np.array([0.0, 0.0, 1.0])
    else:
        h_target_hat = h_target / h_target_norm

    lander.hub.r_CN_NInit = r0.reshape(3, 1).tolist()
    lander.hub.v_CN_NInit = v0.reshape(3, 1).tolist()
    lander.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    lander.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    sampling_time = macros.sec2nano(2.0)
    lander_recorder = lander.scStateOutMsg.recorder(sampling_time)
    sc_sim.AddModelToTask(sim_task_name, lander_recorder)
    vizSupport.enableUnityVisualization(
        sc_sim,
        sim_task_name,
        lander,
        saveFile=fileName,
    )
    sc_sim.InitializeSimulation()
    sc_sim.SetProgressBar(False)

    # Local burn_propellant function with proper dry mass clamping
    def burn_propellant(dv_mps):
        nonlocal current_mass
        if current_mass <= ascent_dry_mass:
            return 0.0
        propUsed_unclamped = current_mass * (1.0 - np.exp(-abs(dv_mps) / (isp_s * G0)))
        new_mass = max(ascent_dry_mass, current_mass - propUsed_unclamped)
        propUsed = current_mass - new_mass
        current_mass = new_mass
        lander.hub.mHub = current_mass
        return propUsed

    position_ref = lander.dynManager.getStateObject(
        lander.hub.nameOfHubPosition
    )
    velocity_ref = lander.dynManager.getStateObject(
        lander.hub.nameOfHubVelocity
    )
    t = 0.0
    last_print = -print_interval_s
    print("\n" + "=" * 70)
    print("ASCENT STAGE: Return from Lunar Surface to NRHO")
    print("=" * 70)
    print(f"Mission Timeline: Day 15, Hour 00:00")
    print(f"  Ascent Dry Mass:     {ascent_dry_mass:.1f} kg")
    print(f"  Ascent Wet Mass:     {current_mass:.1f} kg (with propellant)")
    print(f"  Touchdown Location:  {state['latitude_deg']:.3f}°, {state['longitude_deg']:.3f}°")
    print(f"  Target Orbit:        NRHO (Perilune: {NRHO_PERILUNE_RADIUS_M/1e3:.0f} km, Apolune: {NRHO_APOLUNE_RADIUS_M/1e3:.0f} km)")
    print(f"  Engine Config:       {num_thrusters} thruster(s) × {thrust_N/1000:.1f} kN = {num_thrusters*thrust_N/1000:.1f} kN total")
    print(f"  Specific Impulse:    {isp_s:.0f} sec")
    print("=" * 70 + "\n")

    altitude_history = []
    speed_history = []
    time_history = []
    mass_history = []
    lat_history = []
    lon_history = []
    # phases: liftoff, circ, loiter, lod, transfer, nri
    phase = "liftoff"
    phase_numbers = {
        "liftoff": 9,
        "circ": 10,
        "loiter": 11,
        "lod": 12,
        "transfer": 12,
        "nri": 13,
    }
    a_cmd_previous = np.zeros(3)
    total_delta_v = 0.0
    mission_success = False
    has_lifted_off = False
    lift_off_clearance_m = 1.0
    impact_tolerance_m = -1.0
    liftoff_target_altitude_m = 100.0e3
    loiter_start_t = 0.0
    loiter_duration_s = 2.0 * 3600.0
    # FIX: initialize summary variables so the post-loop report cannot hit
    # unbound locals if the loop body never runs (e.g. zero propellant).
    altitude = 0.0
    speed = np.linalg.norm(v0)
    r_now = r0.copy()
    v_now = v0.copy()
    # 10 hour max duration for the entire ascent (transfer to the NRHO
    # perilune radius is only ~100 min, so this is generous).
    t_stop = 10.0 * 3600.0

    while t < t_stop and current_mass > ascent_dry_mass + 1e-6:
        r_now = np.asarray(
            position_ref.getState(),
            dtype=float,
        ).flatten()
        v_now = np.asarray(
            velocity_ref.getState(),
            dtype=float,
        ).flatten()
        if r_now.size != 3 or v_now.size != 3:
            print("Invalid Basilisk state vector.")
            break
        r_magnitude = np.linalg.norm(r_now)
        if r_magnitude <= 1.0:
            print("Invalid spacecraft position magnitude.")
            break
        r_hat = r_now / r_magnitude
        # Account for lunar rotation when querying the DEM
        local_radius, latitude_deg, longitude_deg, source = get_local_radius(
            r_now,
            t,
            dem_path,
        )
        altitude = r_magnitude - local_radius
        speed = np.linalg.norm(v_now)
        v_radial = np.dot(v_now, r_hat)
        v_vert = v_radial * r_hat
        v_horiz = v_now - v_vert
        h_speed_now = np.linalg.norm(v_horiz)
        if altitude > lift_off_clearance_m:
            has_lifted_off = True
        # Only check for terrain impact during liftoff phase
        if phase == "liftoff" and has_lifted_off and altitude < impact_tolerance_m:
            print(
                f"Terrain impact at t={t:.2f} s, "
                f"altitude={altitude:.2f} m."
            )
            break
        if current_mass <= ascent_dry_mass + 1e-6:
            ascent_thruster.extForce_N = [[0.0], [0.0], [0.0]]
            print(
                f"[warning] propellant depleted at t={t/60:.2f} min; "
                f"ending burn at dry mass {ascent_dry_mass:.1f} kg"
            )
            break
        if t - last_print >= print_interval_s:
            propellant_remaining = max(current_mass - ascent_dry_mass, 0.0)
            phase_num = phase_numbers.get(phase, 0)
            print(
                f"[Step {phase_num:2d}] t={t / 60.0:7.1f} min | "
                f"alt={altitude / 1000.0:7.2f} km | "
                f"speed={speed:7.1f} m/s | "
                f"mass={current_mass:8.1f} kg | "
                f"prop={propellant_remaining:7.1f} kg | "
                f"{phase.upper()}"
            )
            last_print = t
        g_vec = -MU_MOON * r_now / r_magnitude**3
        # Prograde tangential direction at the current position, consistent
        # with the target NRHO orbital plane.
        t_hat_target_plane = np.cross(h_target_hat, r_hat)
        t_hat_norm = np.linalg.norm(t_hat_target_plane)
        if t_hat_norm > 1e-9:
            t_hat_target_plane = t_hat_target_plane / t_hat_norm
        else:
            t_hat_target_plane = np.array([1.0, 0.0, 0.0])

        # ---------------- Mission phases ----------------
        if phase == "liftoff":
            # Step 9: Liftoff / gravity-turn burn. Pitch the thrust vector
            # from vertical toward prograde as altitude builds so we arrive
            # at ~100 km with tangential velocity to circularize with.
            pitch_fraction = min(altitude / liftoff_target_altitude_m, 1.0)
            blend = np.sin(pitch_fraction * np.pi / 2.0)
            thrust_dir = (1.0 - blend) * r_hat + blend * t_hat_target_plane
            thrust_dir_norm = np.linalg.norm(thrust_dir)
            if thrust_dir_norm > 1e-9:
                thrust_dir = thrust_dir / thrust_dir_norm
            a_cmd = -g_vec + 5.0 * thrust_dir
            if altitude >= liftoff_target_altitude_m:
                phase = "circ"
                print(f"\n>>> [STEP 9 COMPLETE] Liftoff/Pitchover Burn to 100 km")
                print(f"    Time: {t/60:.1f} min | Altitude: {altitude/1000:.1f} km | Mass: {current_mass:.1f} kg")
                print(f"    Speed: {speed:.1f} m/s | Tangential: {h_speed_now:.1f} m/s | Radial: {np.linalg.norm(v_vert):.1f} m/s")
                print(f">>> [STEP 10 START] Circularization Burn (~100 km LLO)\n")
        elif phase == "circ":
            # Step 10: Circularize at the CURRENT radius.
            #
            # FIX (deadlock): the old gate demanded |r - LLO_RADIUS| < 2 km
            # but the law had no radial-position feedback, so the radius
            # could settle a few km off the target and the gate never
            # tripped. Circularizing at the achieved radius is what a real
            # ascent does; the exact LLO altitude is unimportant.
            #
            # FIX (efficiency): no -g_vec term. Once tangential speed is
            # circular, gravity IS the centripetal force; canceling it
            # would push the vehicle outward and waste propellant.
            v_circ_local = np.sqrt(MU_MOON / r_magnitude)
            if h_speed_now > 1e-9:
                t_hat_now = v_horiz / h_speed_now
            else:
                t_hat_now = t_hat_target_plane
            a_cmd = (
                -0.8 * v_radial * r_hat
                + 0.5 * (v_circ_local - h_speed_now) * t_hat_now
            )
            if abs(v_radial) < 2.0 and abs(h_speed_now - v_circ_local) < 5.0:
                phase = "loiter"
                loiter_start_t = t
                T_LLO = 2.0 * np.pi * np.sqrt(r_magnitude**3 / MU_MOON)
                print(f"\n>>> [STEP 10 COMPLETE] Circularization at {altitude/1000:.1f} km LLO")
                print(f"    Time: {t/60:.1f} min | Radius: {r_magnitude/1000:.1f} km | Speed: {speed:.1f} m/s")
                print(f"    LLO Period: {T_LLO/60:.1f} min | Mass: {current_mass:.1f} kg")
                print(f">>> [STEP 11 START] LLO Loiter ({loiter_duration_s/3600:.0f} h coast)\n")
        elif phase == "loiter":
            # Step 11: LLO Loiter - fixed-duration zero-thrust coast.
            loiter_elapsed = t - loiter_start_t
            a_cmd = np.zeros(3)
            if loiter_elapsed >= loiter_duration_s:
                phase = "lod"
                T_LLO = 2.0 * np.pi * np.sqrt(r_magnitude**3 / MU_MOON)
                print(f"\n>>> [STEP 11 COMPLETE] LLO Loiter Phase")
                print(f"    Time: {t/60:.1f} min | Duration: {loiter_elapsed/3600:.2f} hr ({loiter_elapsed/T_LLO:.1f} revolutions)")
                print(f"    Altitude: {altitude/1000:.1f} km | Mass: {current_mass:.1f} kg")
                print(f">>> [STEP 12 START] LOD Burn (raise apoapsis to NRHO perilune radius)\n")
        elif phase == "lod":
            # Step 12: LOD - finite prograde burn to raise apoapsis to the
            # NRHO perilune radius (3,366 km), i.e. the first half of a
            # Hohmann transfer. ~225 m/s.
            #
            # FIX: the old law servoed toward the perilune POINT while the
            # exit gate waited at r >= 35,500 km (half the APOLUNE radius).
            # The gate was unreachable, so the phase hung, gravity-canceling
            # until propellant depletion. This is the main reason the old
            # sim never got to NRHO.
            a_transfer = (r_magnitude + NRHO_PERILUNE_RADIUS_M) / 2.0
            v_p_needed = np.sqrt(
                MU_MOON * (2.0 / r_magnitude - 1.0 / a_transfer)
            )
            if speed < v_p_needed - 0.5:
                a_cmd = 10.0 * (v_now / max(speed, 1e-9))  # clamped to engine max
            else:
                phase = "transfer"
                a_cmd = np.zeros(3)
                print(f"\n>>> [STEP 12 COMPLETE] LOD Burn")
                print(f"    Time: {t/60:.1f} min | Speed: {speed:.1f} m/s (needed {v_p_needed:.1f} m/s)")
                print(f"    Mass: {current_mass:.1f} kg | Total ΔV so far: {total_delta_v:.1f} m/s")
                print(f">>> [STEP 12b START] Transfer Coast to {NRHO_PERILUNE_RADIUS_M/1e3:.0f} km radius (~100 min)\n")
        elif phase == "transfer":
            # Step 12b: zero-thrust coast up the transfer ellipse.
            a_cmd = np.zeros(3)
            if (
                r_magnitude >= 0.97 * NRHO_PERILUNE_RADIUS_M
                or (has_lifted_off and v_radial < -1.0 and r_magnitude > 2.0 * LLO_RADIUS_M)
            ):
                phase = "nri"
                print(f"\n>>> [STEP 12b COMPLETE] Transfer Coast")
                print(f"    Time: {t/60:.1f} min | Radius: {r_magnitude/1000:.0f} km | Speed: {speed:.1f} m/s")
                print(f">>> [STEP 13 START] NRI Burn (match NRHO velocity at perilune radius)\n")
        elif phase == "nri":
            # Step 13: NRI - finite burn to match the NRHO velocity for the
            # current radius (vis-viva on the target ellipse), directed
            # prograde in the target plane. ~650 m/s near apoapsis of the
            # transfer.
            v_des_mag_sq = MU_MOON * (2.0 / r_magnitude - 1.0 / NRHO_SMA_M)
            v_des = np.sqrt(max(v_des_mag_sq, 0.0)) * t_hat_target_plane
            dv_vec = v_des - v_now
            dv_mag = np.linalg.norm(dv_vec)
            if dv_mag > 2.0:
                a_cmd = 10.0 * dv_vec / dv_mag  # clamped to engine max
            else:
                # Success is judged on achieved orbital ELEMENTS.
                oe_now = orbitalMotion.rv2elem(MU_MOON, r_now, v_now)
                rp_achieved = oe_now.a * (1.0 - oe_now.e)
                ra_achieved = oe_now.a * (1.0 + oe_now.e)
                h_now = np.cross(r_now, v_now)
                h_now_hat = h_now / max(np.linalg.norm(h_now), 1e-9)
                plane_alignment = float(np.dot(h_now_hat, h_target_hat))
                rp_ok = abs(rp_achieved - NRHO_PERILUNE_RADIUS_M) < 300.0e3
                ra_ok = abs(ra_achieved - NRHO_APOLUNE_RADIUS_M) < 15000.0e3
                plane_ok = plane_alignment > 0.99
                mission_success = rp_ok and ra_ok and plane_ok
                print(f"\n>>> [STEP 13 COMPLETE] NRI Burn")
                print(f"    Time: {t/60:.1f} min | Residual ΔV error: {dv_mag:.2f} m/s")
                print(f"    Achieved perilune: {rp_achieved/1e3:8.0f} km (target {NRHO_PERILUNE_RADIUS_M/1e3:.0f} km) {'OK' if rp_ok else 'OFF'}")
                print(f"    Achieved apolune:  {ra_achieved/1e3:8.0f} km (target {NRHO_APOLUNE_RADIUS_M/1e3:.0f} km) {'OK' if ra_ok else 'OFF'}")
                print(f"    Plane alignment:   {plane_alignment:8.4f} (want > 0.99) {'OK' if plane_ok else 'OFF'}")
                if mission_success:
                    print(f"\n>>> [STEP 14] NRHO INSERTION SUCCESS -- ready for RPOD with Orion")
                    print(f"    Final Mass: {current_mass:.1f} kg | Total ΔV: {total_delta_v:.1f} m/s\n")
                else:
                    print(f"\n>>> [STEP 14] Insertion complete but outside NRHO tolerances\n")
                break
        else:
            a_cmd = np.zeros(3)

        # Limit acceleration command rate (jerk limiter)
        max_delta = max_jerk * dt
        delta = a_cmd - a_cmd_previous
        delta_mag = np.linalg.norm(delta)
        if delta_mag > max_delta > 0.0:
            a_cmd = a_cmd_previous + delta / delta_mag * max_delta
        a_cmd_previous = a_cmd.copy()
        # Apply engine acceleration limit
        a_mag = np.linalg.norm(a_cmd)
        max_engine_accel = (thrust_N * num_thrusters) / max(current_mass, 1e-9)
        if a_mag > max_engine_accel:
            a_cmd = a_cmd / a_mag * max_engine_accel
            a_mag = max_engine_accel
        # Burn propellant and apply force (coast phases command zero, so
        # they consume nothing).
        burn_propellant(a_mag * dt)
        total_delta_v += a_mag * dt
        ascent_thruster.extForce_N = (current_mass * a_cmd).reshape(3, 1).tolist()
        altitude_history.append(altitude)
        speed_history.append(speed)
        time_history.append(t)
        mass_history.append(current_mass)
        lat_history.append(latitude_deg)
        lon_history.append(longitude_deg)
        t += dt
        sc_sim.ConfigureStopTime(macros.sec2nano(t))
        sc_sim.ExecuteSimulation()

    # Turn off thrust at the end.
    ascent_thruster.extForce_N = [[0.0], [0.0], [0.0]]
    print("=" * 70)
    print("ASCENT MISSION SUMMARY")
    print("=" * 70)
    if mission_success:
        print(f"✓ MISSION COMPLETE: NRHO insertion achieved")
        print(f"  Mission Time:        {t/3600:.2f} hours")
        print(f"  Final Mass:          {current_mass:.1f} kg (dry mass: {ascent_dry_mass:.1f} kg)")
        print(f"  Propellant Used:     {ascent_wet_mass - current_mass:.1f} kg")
        print(f"  Total ΔV Budget:     {total_delta_v:.1f} m/s")
    else:
        print(
            f"⚠ MISSION INCOMPLETE: Ended in '{phase.upper()}' phase\n"
            f"  Mission Time:        {t/3600:.2f} hours\n"
            f"  Final Mass:          {current_mass:.1f} kg\n"
            f"  Total ΔV Used:       {total_delta_v:.1f} m/s\n"
            f"  Altitude:            {altitude/1000:.1f} km\n"
            f"  Speed:               {speed:.1f} m/s"
        )
    print("=" * 70)

    position_data = np.asarray(
        lander_recorder.r_BN_N,
        dtype=float,
    )
    # Prevent plotting errors if the recorder contains no samples or
    # returns a one-dimensional array.
    if position_data.size == 0:
        print(
            "Warning: recorder contained no position samples; "
            "plotting the initial position."
        )
        position_data = r0.reshape(1, 3)
    elif position_data.ndim == 1:
        if position_data.size == 3:
            position_data = position_data.reshape(1, 3)
        elif position_data.size % 3 == 0:
            position_data = position_data.reshape(-1, 3)
        else:
            print(
                "Warning: unexpected recorder shape "
                f"{position_data.shape}; plotting initial position."
            )
            position_data = r0.reshape(1, 3)
    plt.close("all")
    figure_list = {}
    figure_3d = plt.figure(figsize=(9, 8))
    axis_3d = figure_3d.add_subplot(
        111,
        projection="3d",
    )
    u, v = np.mgrid[
        0.0:2.0 * np.pi:40j,
        0.0:np.pi:40j,
    ]
    moon_radius_km = MEAN_RADIUS_M / 1000.0
    x = moon_radius_km * np.cos(u) * np.sin(v)
    y = moon_radius_km * np.sin(u) * np.sin(v)
    z = moon_radius_km * np.cos(v)
    axis_3d.plot_surface(
        x,
        y,
        z,
        color="#888888",
        alpha=0.6,
    )
    axis_3d.plot3D(
        position_data[:, 0] / 1000.0,
        position_data[:, 1] / 1000.0,
        position_data[:, 2] / 1000.0,
        color="orangered",
        label="Ascent Lander",
    )
    trajectory_limit_km = max(
        NRHO_PERILUNE_RADIUS_M / 1000.0 * 1.3,
        np.max(np.abs(position_data)) / 1000.0 * 1.1,
    )
    axis_3d.set_xlim3d(
        -trajectory_limit_km,
        trajectory_limit_km,
    )
    axis_3d.set_ylim3d(
        -trajectory_limit_km,
        trajectory_limit_km,
    )
    axis_3d.set_zlim3d(
        -trajectory_limit_km,
        trajectory_limit_km,
    )
    axis_3d.set_xlabel("x [km]")
    axis_3d.set_ylabel("y [km]")
    axis_3d.set_zlabel("z [km]")
    axis_3d.set_title("Ascent to NRHO")
    axis_3d.legend()
    figure_list[fileName + "_3d"] = figure_3d
    if len(time_history) > 1:
        ascent_figure, axes = plt.subplots(
            5,
            1,
            figsize=(8, 12),
            sharex=True,
        )
        time_hours = np.asarray(time_history) / 3600.0
        axes[0].plot(time_hours, np.asarray(altitude_history) / 1000.0, color="orangered")
        axes[0].set_ylabel("Altitude [km]")
        axes[0].grid(True)
        axes[0].set_title("Ascent from surface to NRHO")
        axes[1].plot(time_hours, speed_history, color="deepskyblue")
        axes[1].set_ylabel("Speed [m/s]")
        axes[1].grid(True)
        axes[2].plot(time_hours, lat_history, label="latitude [deg]", color="tab:green")
        axes[2].plot(time_hours, lon_history, label="longitude [deg]", color="tab:purple")
        axes[2].set_ylabel("Lat / Lon [deg]")
        axes[2].legend()
        axes[2].grid(True)
        axes[3].plot(time_hours, mass_history, color="tab:brown")
        axes[3].set_ylabel("Mass [kg]")
        axes[3].grid(True)
        axes[4].plot(time_hours, np.asarray(altitude_history) / 1000.0, color="orangered", label="altitude")
        axes[4].axhline(100.0, color="green", linestyle="--", label="100 km LLO")
        axes[4].axhline((NRHO_PERILUNE_RADIUS_M - MEAN_RADIUS_M) / 1000.0, color="orange", linestyle="--", label="NRHO perilune altitude")
        axes[4].set_ylabel("Altitude [km]")
        axes[4].set_xlabel("Time [hr]")
        axes[4].legend()
        axes[4].grid(True)
        plt.tight_layout()
        figure_list[fileName + "_ascent"] = ascent_figure
    if show_plots:
        plt.show()
        plt.close("all")
    # NOTE: figures are intentionally NOT closed when show_plots is False,
    # so the returned figure_list contains live figures the caller can use.
    return figure_list


if __name__ == "__main__":
    run(
        show_plots=True,
        dem_path="LDEM_875S_5M.IMG",
        ascent_mass=4700.0,
        # Inertia tensor: 9 values in row-major order
        ascent_inertia=(1500.0, 0.0, 0.0,
                        0.0, 420.0, 0.0,
                        0.0, 0.0, 300.0),
        thrust_N=24500.0,
        num_thrusters=1,
        isp_s=339.0,
        sim_time_step_s=0.7,
        print_interval_s=30.0,
        max_jerk=4.0,
        blend_duration_s=20.0,
    )
