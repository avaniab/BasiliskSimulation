# Companion ascent simulation that starts from the saved touchdown state
# produced by descent_sim.py and uses the remaining mass as the ascent-stage
# propellant budget.
#
# ============================ GUIDANCE REWORK ============================
# Why the old version never reached NRHO:
#
#   1. LOD phase deadlock: the control law servoed toward r_target -- the
#      NRHO *perilune* point at r = 3,366 km -- but the phase exit gate
#      waited for r >= 0.5 * APOLUNE = 35,500 km. Unreachable gate; the
#      phase hung, gravity-canceling until propellant depletion.
#   2. Circ phase deadlock: exit gate required |r - LLO_RADIUS| < 2 km but
#      the law had no radial-position feedback.
#   3. Gravity cancellation everywhere: hover-thrusting through multi-hour
#      phases costs tens of km/s of delta-V.
#
# The rework uses a standard burn + coast architecture:
#   liftoff -> circ -> loiter -> lod (prograde burn) -> transfer (coast)
#   -> nri (velocity-match burn) -> element-based success check.
#
# ========================= VIZARD VISUALIZATION =========================
# The sim streams/records a full Vizard scene:
#   * The ascent lander AND an "OrionNRHO" target vehicle propagating the
#     NRHO ellipse, so the rendezvous geometry is visible end to end.
#   * A location pin on the lunar surface at the touchdown site.
#   * True (flown) trajectory lines, labels, and preset cameras.
#   * viz_mode selects the output:
#       "file"  - record a playback file:  _VizFiles/<name>_UnityViz.bin
#                 (open it in Vizard: File > Load; Vizard download:
#                  https://hanspeterschaub.info/basilisk/Vizard/Vizard.html)
#       "live"  - 2-way live stream; start Vizard FIRST and connect in
#                 "Live Display" mode, else the sim waits for a connection
#       "both"  - record and stream simultaneously
#       "off"   - no visualization
#
# ============================ CAD MODEL HOOK ============================
# Custom CAD geometry is plumbed through but OFF by default. When you have
# a model ready (e.g. exported from Fusion 360 / SolidWorks as .obj with
# units in meters, +X forward, +Z up), just pass:
#     lander_cad_path="models/ascent_stage.obj"
#     orion_cad_path="models/orion.obj"
# and optionally cad_scale / cad_offset_m / cad_rotation_deg. The hook uses
# vizSupport.createCustomModel(), so .obj/.stl/.fbx and Unity primitive
# names ("CUBE", "CYLINDER", "SPHERE", ...) all work. Until then, Vizard's
# built-in spacecraft model is used. Every viz feature is wrapped
# defensively: if your Basilisk build lacks a feature (or vizInterface
# entirely), the sim still runs and just prints a [viz] note.
# ========================================================================
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

ORION_MASS_KG = 26520.0

# Colors used for the phase shading in the summary plots.
PHASE_COLORS = {
    "liftoff": "#ffd9b3",
    "circ": "#ffe9a8",
    "loiter": "#d6f5d6",
    "lod": "#ffc4c4",
    "transfer": "#cfe8ff",
    "nri": "#e6ccff",
}


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


def setup_visualization(
    sc_sim,
    sim_task_name,
    sc_list,
    moon_body,
    viz_mode,
    viz_file_name,
    landing_site_r_P,
    lander_tag,
    cad_models,
):
    """Create the Vizard scene. Returns the viz instance or None.

    Every optional feature is wrapped in try/except so version drift in
    Basilisk's vizSupport API (or a build without vizInterface) can never
    take down the simulation itself.
    """
    if viz_mode == "off":
        print("[viz] Vizard output disabled (viz_mode='off').")
        return None
    if not getattr(vizSupport, "vizFound", True):
        print("[viz] This Basilisk build has no vizInterface; skipping Vizard output.")
        return None
    viz_kwargs = {}
    if viz_mode in ("file", "both"):
        viz_kwargs["saveFile"] = viz_file_name
    if viz_mode in ("live", "both"):
        viz_kwargs["liveStream"] = True
        print("[viz] Live streaming enabled -- start Vizard and connect in "
              "'Live Display' mode before/when the sim starts.")
    try:
        viz = vizSupport.enableUnityVisualization(
            sc_sim,
            sim_task_name,
            sc_list,
            **viz_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - never let viz kill the sim
        print(f"[viz] enableUnityVisualization failed ({exc}); continuing without Vizard.")
        return None
    if viz is None:
        print("[viz] vizInterface unavailable; continuing without Vizard.")
        return None

    # --- Scene settings -------------------------------------------------
    try:
        viz.settings.trueTrajectoryLinesOn = 1   # draw the actual flown path
        viz.settings.orbitLinesOn = 1            # osculating orbit lines
        viz.settings.showSpacecraftLabels = 1
        viz.settings.showCelestialBodyLabels = 1
        viz.settings.mainCameraTarget = lander_tag
        viz.settings.spacecraftSizeMultiplier = 2.0
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] could not apply scene settings ({exc}).")

    # --- Landing-site location pin ---------------------------------------
    # r_GP_P is the site position in the Moon-fixed frame; at t=0 the
    # inertial and Moon-fixed frames coincide in this sim.
    for parent_name in ("moon",
                        getattr(moon_body, "displayName", "") or "",
                        getattr(moon_body, "planetName", "") or ""):
        if not parent_name:
            continue
        try:
            vizSupport.addLocation(
                viz,
                stationName="Touchdown Site",
                parentBodyName=parent_name,
                r_GP_P=list(np.asarray(landing_site_r_P, dtype=float)),
                fieldOfView=np.radians(170.0),
                color="red",
                range=2000.0e3,
            )
            print(f"[viz] landing-site marker added (parent body '{parent_name}').")
            break
        except Exception:  # noqa: BLE001 - try the next candidate name
            continue
    else:
        print("[viz] could not add the landing-site marker; continuing.")

    # --- Preset cameras ---------------------------------------------------
    try:
        # Wide shot keeping the Moon in frame.
        vizSupport.createStandardCamera(
            viz,
            setMode=0,
            bodyTarget="moon",
            setView=0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] moon camera not created ({exc}).")
    try:
        # Nadir-looking camera riding on the lander.
        vizSupport.createStandardCamera(
            viz,
            setMode=1,
            spacecraftName=lander_tag,
            displayName="LanderNadirCam",
            fieldOfView=np.radians(70.0),
            pointingVector_B=[0.0, 0.0, -1.0],
            position_B=[0.0, 0.0, 2.0],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] lander camera not created ({exc}).")

    # --- CAD models (optional) --------------------------------------------
    # Each entry: {"path", "bodies", "scale", "offset_m", "rotation_deg"}.
    # Accepts .obj/.stl/.fbx paths or Unity primitives ("CUBE", ...).
    for model in cad_models:
        path = model.get("path")
        if not path:
            continue
        is_primitive = str(path).isupper() and "." not in str(path)
        if not is_primitive and not os.path.exists(path):
            print(f"[viz] CAD model not found at '{path}'; using default geometry.")
            continue
        try:
            scale = float(model.get("scale", 1.0))
            vizSupport.createCustomModel(
                viz,
                modelPath=str(path),
                simBodiesToModify=list(model.get("bodies", [])),
                scale=[scale, scale, scale],
                offset=list(model.get("offset_m", (0.0, 0.0, 0.0))),
                rotation=list(np.radians(model.get("rotation_deg", (0.0, 0.0, 0.0)))),
            )
            print(f"[viz] CAD model '{path}' attached to {model.get('bodies')}.")
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] could not attach CAD model '{path}' ({exc}).")

    if viz_mode in ("file", "both"):
        print(f"[viz] recording playback file: look for "
              f"'_VizFiles/{viz_file_name}_UnityViz.bin' next to this script.")
    return viz


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
    # ------------- visualization options -------------
    viz_mode="file",          # "file" | "live" | "both" | "off"
    viz_file_name=None,       # defaults to this script's name
    show_orion_target=True,   # add Orion propagating the NRHO for context
    # ------------- CAD hooks (leave None until models exist) -------------
    lander_cad_path=None,     # e.g. "models/ascent_stage.obj" or "CYLINDER"
    orion_cad_path=None,      # e.g. "models/orion.obj"
    cad_scale=1.0,
    cad_offset_m=(0.0, 0.0, 0.0),
    cad_rotation_deg=(0.0, 0.0, 0.0),
):
    if state_path is None:
        state_path = os.path.join(
            os.path.dirname(__file__),
            "descent_state.json",
        )
    if viz_file_name is None:
        viz_file_name = fileName
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
    grav_bodies = spacecraft.GravBodyVector(
        list(grav_factory.gravBodies.values())
    )
    lander = spacecraft.Spacecraft()
    lander.ModelTag = "ascentLander"
    lander.hub.mHub = current_mass
    lander.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    # NOTE: Basilisk.utilities has no `simHelpers` module; build the 3x3
    # inertia matrix directly from the 9 row-major values.
    inertia = list(ascent_inertia)
    lander.hub.IHubPntBc_B = [inertia[0:3], inertia[3:6], inertia[6:9]]
    lander.gravField.gravBodies = grav_bodies
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
    # Defines the target orbital PLANE and elements; success is judged on
    # achieved elements, not on reaching a static point.
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

    # --- Orion target vehicle on the NRHO (visual + range reference) -----
    # A second spacecraft propagating the target ellipse ballistically, so
    # Vizard shows both vehicles and the rendezvous geometry, and the sim
    # can report range-to-Orion.
    orion = None
    if show_orion_target:
        orion = spacecraft.Spacecraft()
        orion.ModelTag = "OrionNRHO"
        orion.hub.mHub = ORION_MASS_KG
        orion.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
        orion.hub.IHubPntBc_B = [
            [9.0e5, 0.0, 0.0],
            [0.0, 9.0e5, 0.0],
            [0.0, 0.0, 9.0e5],
        ]
        orion.gravField.gravBodies = grav_bodies
        orion.hub.r_CN_NInit = r_target.reshape(3, 1).tolist()
        orion.hub.v_CN_NInit = v_target.reshape(3, 1).tolist()
        orion.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
        orion.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
        sc_sim.AddModelToTask(sim_task_name, orion, None, 1)

    lander.hub.r_CN_NInit = r0.reshape(3, 1).tolist()
    lander.hub.v_CN_NInit = v0.reshape(3, 1).tolist()
    lander.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    lander.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    sampling_time = macros.sec2nano(2.0)
    lander_recorder = lander.scStateOutMsg.recorder(sampling_time)
    sc_sim.AddModelToTask(sim_task_name, lander_recorder)
    orion_recorder = None
    if orion is not None:
        orion_recorder = orion.scStateOutMsg.recorder(sampling_time)
        sc_sim.AddModelToTask(sim_task_name, orion_recorder)

    # --- Vizard scene ------------------------------------------------------
    cad_models = []
    if lander_cad_path:
        cad_models.append({
            "path": lander_cad_path,
            "bodies": [lander.ModelTag],
            "scale": cad_scale,
            "offset_m": cad_offset_m,
            "rotation_deg": cad_rotation_deg,
        })
    if orion_cad_path and orion is not None:
        cad_models.append({
            "path": orion_cad_path,
            "bodies": [orion.ModelTag],
            "scale": cad_scale,
            "offset_m": cad_offset_m,
            "rotation_deg": cad_rotation_deg,
        })
    sc_list = [lander] if orion is None else [lander, orion]
    setup_visualization(
        sc_sim,
        sim_task_name,
        sc_list,
        moon,
        viz_mode,
        viz_file_name,
        landing_site_r_P=r0,
        lander_tag=lander.ModelTag,
        cad_models=cad_models,
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
    orion_position_ref = None
    if orion is not None:
        orion_position_ref = orion.dynManager.getStateObject(
            orion.hub.nameOfHubPosition
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
    if orion is not None:
        print(f"  Target Vehicle:      OrionNRHO ({ORION_MASS_KG:.0f} kg) propagating the NRHO")
    print("=" * 70 + "\n")

    altitude_history = []
    speed_history = []
    time_history = []
    mass_history = []
    lat_history = []
    lon_history = []
    orion_range_history = []
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
    phase_events = [(0.0, "liftoff")]  # (start time, phase) for plot shading
    a_cmd_previous = np.zeros(3)
    total_delta_v = 0.0
    mission_success = False
    has_lifted_off = False
    lift_off_clearance_m = 1.0
    impact_tolerance_m = -1.0
    liftoff_target_altitude_m = 100.0e3
    loiter_start_t = 0.0
    loiter_duration_s = 2.0 * 3600.0
    min_orion_range = np.inf
    min_orion_range_t = 0.0
    # Initialize summary variables so the post-loop report cannot hit
    # unbound locals if the loop body never runs (e.g. zero propellant).
    altitude = 0.0
    speed = np.linalg.norm(v0)
    r_now = r0.copy()
    v_now = v0.copy()
    # 10 hour max duration for the entire ascent (transfer to the NRHO
    # perilune radius is only ~100 min, so this is generous).
    t_stop = 10.0 * 3600.0

    def switch_phase(new_phase):
        nonlocal phase
        phase = new_phase
        phase_events.append((t, new_phase))

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
        orion_range = np.nan
        if orion_position_ref is not None:
            orion_r = np.asarray(
                orion_position_ref.getState(),
                dtype=float,
            ).flatten()
            if orion_r.size == 3:
                orion_range = np.linalg.norm(orion_r - r_now)
                if orion_range < min_orion_range:
                    min_orion_range = orion_range
                    min_orion_range_t = t
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
            range_txt = ""
            if np.isfinite(orion_range):
                range_txt = f" | Orion rng={orion_range/1000.0:8.0f} km"
            print(
                f"[Step {phase_num:2d}] t={t / 60.0:7.1f} min | "
                f"alt={altitude / 1000.0:7.2f} km | "
                f"speed={speed:7.1f} m/s | "
                f"mass={current_mass:8.1f} kg | "
                f"prop={propellant_remaining:7.1f} kg | "
                f"{phase.upper()}{range_txt}"
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
                switch_phase("circ")
                print(f"\n>>> [STEP 9 COMPLETE] Liftoff/Pitchover Burn to 100 km")
                print(f"    Time: {t/60:.1f} min | Altitude: {altitude/1000:.1f} km | Mass: {current_mass:.1f} kg")
                print(f"    Speed: {speed:.1f} m/s | Tangential: {h_speed_now:.1f} m/s | Radial: {np.linalg.norm(v_vert):.1f} m/s")
                print(f">>> [STEP 10 START] Circularization Burn (~100 km LLO)\n")
        elif phase == "circ":
            # Step 10: Circularize at the CURRENT radius. Rate + speed
            # feedback only -- no gravity cancellation, achievable gate.
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
                switch_phase("loiter")
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
                switch_phase("lod")
                T_LLO = 2.0 * np.pi * np.sqrt(r_magnitude**3 / MU_MOON)
                print(f"\n>>> [STEP 11 COMPLETE] LLO Loiter Phase")
                print(f"    Time: {t/60:.1f} min | Duration: {loiter_elapsed/3600:.2f} hr ({loiter_elapsed/T_LLO:.1f} revolutions)")
                print(f"    Altitude: {altitude/1000:.1f} km | Mass: {current_mass:.1f} kg")
                print(f">>> [STEP 12 START] LOD Burn (raise apoapsis to NRHO perilune radius)\n")
        elif phase == "lod":
            # Step 12: LOD - finite prograde burn to raise apoapsis to the
            # NRHO perilune radius (Hohmann first half, ~225 m/s).
            a_transfer = (r_magnitude + NRHO_PERILUNE_RADIUS_M) / 2.0
            v_p_needed = np.sqrt(
                MU_MOON * (2.0 / r_magnitude - 1.0 / a_transfer)
            )
            if speed < v_p_needed - 0.5:
                a_cmd = 10.0 * (v_now / max(speed, 1e-9))  # clamped to engine max
            else:
                switch_phase("transfer")
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
                switch_phase("nri")
                print(f"\n>>> [STEP 12b COMPLETE] Transfer Coast")
                print(f"    Time: {t/60:.1f} min | Radius: {r_magnitude/1000:.0f} km | Speed: {speed:.1f} m/s")
                print(f">>> [STEP 13 START] NRI Burn (match NRHO velocity at perilune radius)\n")
        elif phase == "nri":
            # Step 13: NRI - finite burn to match the NRHO velocity for the
            # current radius (vis-viva on the target ellipse), directed
            # prograde in the target plane (~650 m/s near transfer apoapsis).
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
                if np.isfinite(orion_range):
                    print(f"    Range to Orion:    {orion_range/1000.0:8.0f} km (phasing handled during RPOD)")
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
        orion_range_history.append(orion_range)
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
    if np.isfinite(min_orion_range):
        print(f"  Closest Orion Range: {min_orion_range/1000.0:.0f} km "
              f"(at t={min_orion_range_t/3600:.2f} hr)")
    if viz_mode in ("file", "both"):
        print(f"  Vizard playback:     _VizFiles/{viz_file_name}_UnityViz.bin")
    print("=" * 70)

    def to_position_array(recorder, fallback_r):
        data = np.asarray(recorder.r_BN_N, dtype=float) if recorder is not None else np.zeros(0)
        if data.size == 0:
            return fallback_r.reshape(1, 3)
        if data.ndim == 1:
            if data.size == 3:
                return data.reshape(1, 3)
            if data.size % 3 == 0:
                return data.reshape(-1, 3)
            return fallback_r.reshape(1, 3)
        return data

    position_data = to_position_array(lander_recorder, r0)
    orion_position_data = None
    if orion_recorder is not None:
        orion_position_data = to_position_array(orion_recorder, r_target)

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
    if orion_position_data is not None and orion_position_data.shape[0] > 1:
        axis_3d.plot3D(
            orion_position_data[:, 0] / 1000.0,
            orion_position_data[:, 1] / 1000.0,
            orion_position_data[:, 2] / 1000.0,
            color="deepskyblue",
            linestyle="--",
            label="Orion (NRHO)",
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
        n_rows = 6 if orion_position_data is not None else 5
        ascent_figure, axes = plt.subplots(
            n_rows,
            1,
            figsize=(8, 2.4 * n_rows),
            sharex=True,
        )
        time_hours = np.asarray(time_history) / 3600.0
        # Phase shading across every panel.
        span_edges = [event_t / 3600.0 for event_t, _ in phase_events]
        span_edges.append(time_hours[-1])
        for axis in axes:
            for k, (event_t, event_phase) in enumerate(phase_events):
                axis.axvspan(
                    span_edges[k],
                    span_edges[k + 1],
                    color=PHASE_COLORS.get(event_phase, "#eeeeee"),
                    alpha=0.35,
                    zorder=0,
                )
        for k, (event_t, event_phase) in enumerate(phase_events):
            mid = 0.5 * (span_edges[k] + span_edges[k + 1])
            axes[0].text(
                mid,
                0.98,
                event_phase.upper(),
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color="#555555",
            )
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
        axes[4].legend()
        axes[4].grid(True)
        if orion_position_data is not None:
            axes[5].plot(
                time_hours,
                np.asarray(orion_range_history) / 1000.0,
                color="tab:blue",
            )
            axes[5].set_ylabel("Orion range [km]")
            axes[5].grid(True)
        axes[-1].set_xlabel("Time [hr]")
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
        # ---- visualization ----
        viz_mode="file",              # "file" | "live" | "both" | "off"
        show_orion_target=True,
        # ---- CAD hooks: uncomment when your models are exported ----
        # lander_cad_path="models/ascent_stage.obj",
        # orion_cad_path="models/orion.obj",
        # cad_scale=1.0,
        # cad_offset_m=(0.0, 0.0, 0.0),
        # cad_rotation_deg=(0.0, 0.0, 0.0),
    )
