#  Adapted from Basilisk example scenarioHohmann.py (AVS Lab, CU Boulder).

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
    simHelpers,
    vizSupport,
)

from lola_terrain import get_surface_radius_m, get_local_slope_deg, assess_landing_stability, MEAN_RADIUS_M

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

MU_MOON = 4.9048695e12
MOON_SPIN_RATE = 2 * np.pi / (27.321661 * 86400.0)
G0 = 9.80665  # standard gravity, m/s^2 -- used for Isp -> propellant conversion regardless of body

NRHO_PERILUNE_RADIUS_M = 3366.0e3
NRHO_APOLUNE_RADIUS_M = 71000.0e3
NRHO_INCLINATION_DEG = 90.0
NRHO_OMEGA_DEG = 90.0

LLO_RADIUS_M = MEAN_RADIUS_M + 100.0e3   # 100 km circular low lunar orbit
PDI_RADIUS_M = MEAN_RADIUS_M + 15.0e3    # ~15 km perilune, matches real DOI targets

LEG_RADIUS = 8.0 
COG_HEIGHT = 4.334


def get_local_radius(r_inertial, t, dem_path):
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


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "mission_config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_impulsive_burn(velRef, currentV, deltaV, direction_hat, label):
    newV = currentV + deltaV * direction_hat
    velRef.setState(newV.reshape(3, 1).tolist())
    print(f"[{label}] deltaV = {deltaV:.2f} m/s")
    return newV


def save_touchdown_state(state_path, dry_mass_kg, wet_mass_kg, r_m, v_mps, t_s, lat_deg, lon_deg, source):
    payload = {
        "descent_stage_dry_mass_kg": float(dry_mass_kg),
        "descent_stage_wet_mass_kg": float(wet_mass_kg),
        "descent_stage_final_mass_kg": float(wet_mass_kg),
        "r_surface_m": [float(x) for x in np.asarray(r_m).flatten()],
        "v_surface_mps": [float(x) for x in np.asarray(v_mps).flatten()],
        "time_s": float(t_s),
        "latitude_deg": float(lat_deg),
        "longitude_deg": float(lon_deg),
        "surface_source": str(source),
    }
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved touchdown state to {state_path}")


def print_mission_timeline():
    print("=" * 60)
    print("Mission timeline")
    print("- Day 8, Hour 00:00 (Step 2 - Separation): At perilune (3,200 km over the South Pole), the lander uncouples from the autonomous Orion.")
    print("- Day 8, Hour 01:00 (Step 3 - NRD Acceleration Burn): The lander fires its descent engine for the NRHO Departure (NRD) burn, dropping out of the halo track.")
    print("- Day 8, Hour 12:00 (Step 4 - LLO Braking Burn): The lander fires its engine for a Lunar Orbit Insertion (LOI) braking burn to capture into a stable circular Low Lunar Orbit (LLO) at a 100 km altitude.")
    print("- Day 8, Hours 12:00–20:00 (Step 5 - LLO Loiter): The lander coasts in LLO for 3 to 4 revolutions (8 hours total) to perform landing site alignment.")
    print("- Day 8, Hour 20:00 (Step 6 - DOI Braking Burn): The lander executes the Descent Orbit Insertion (DOI) braking burn, altering its circular path into a 100 km x 15.3 km elliptical orbit.")
    print("- Day 8, Hour 21:40 (Step 7 - PDI Burn): At the 15.3 km periapsis, the lander starts its 12-minute forward-facing Powered Descent Insertion (PDI) burn to cancel out its forward velocity.")
    print("- Day 8, Hour 22:00 (Touchdown): The lander touches down safely near the lunar South Pole.")
    print("=" * 60)


def run(show_plots, dem_path, lander_dry_mass, lander_wet_mass, lander_inertia,
        thrust_N, num_thrusters, brake_gain, vsink_target, approach_altitude_m,
        terminal_gain, k_speed, isp_s, sim_time_step_s, print_interval_s,
        max_jerk, blend_duration_s):
    config = load_config()
    descent_cfg = config.get("mission", {}).get("stages", {}).get("descent", {})
    target_brake_gain = max(0.08, 2.0 * brake_gain)  # legacy value; kept for config compatibility but unused by the new brake law
    if descent_cfg:
        lander_dry_mass = float(descent_cfg.get("dry_mass_kg", lander_dry_mass))
        lander_wet_mass = float(descent_cfg.get("wet_mass_kg", lander_wet_mass))
        isp_s = float(descent_cfg.get("isp_s", isp_s))
        thrust_N = float(descent_cfg.get("thrust_N", thrust_N))
        num_thrusters = int(descent_cfg.get("num_thrusters", num_thrusters))
        brake_gain = float(descent_cfg.get("brake_gain", brake_gain))
        vsink_target = float(descent_cfg.get("vsink_target_mps", vsink_target))
        approach_altitude_m = float(descent_cfg.get("approach_altitude_m", approach_altitude_m))
        terminal_gain = float(descent_cfg.get("terminal_gain", terminal_gain))
        k_speed = float(descent_cfg.get("k_speed", k_speed))
        max_jerk = float(descent_cfg.get("max_jerk_mps3", max_jerk))
        blend_duration_s = float(descent_cfg.get("blend_duration_s", blend_duration_s))
        sim_time_step_s = float(descent_cfg.get("sim_time_step_s", sim_time_step_s))
        print_interval_s = float(descent_cfg.get("print_interval_s", print_interval_s))
        target_brake_gain = float(descent_cfg.get("target_brake_gain", target_brake_gain))

    if lander_wet_mass < lander_dry_mass:
        raise ValueError("descent wet_mass_kg must be greater than or equal to dry_mass_kg")

    fuel_budget_kg = lander_wet_mass - lander_dry_mass
    if fuel_budget_kg <= 0.0:
        print("[warning] descent stage has no propellant margin (wet mass equals dry mass)")

    dt_guidane = sim_time_step_s
    simTaskName = "simTask"
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess("dynProcess", 0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(sim_time_step_s)), 10)

    gravFactory = simIncludeGravBody.gravBodyFactory()
    moon = gravFactory.createMoon()
    moon.isCentralBody = True
    moon.mu = MU_MOON

    lander = spacecraft.Spacecraft()
    lander.ModelTag = "lander"
    lander.hub.mHub = lander_wet_mass
    lander.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    lander.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
        [lander_inertia[0], 0., 0., 0., lander_inertia[1], 0., 0., 0., lander_inertia[2]])
    lander.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))
    scSim.AddModelToTask(simTaskName, lander, None, 1)

    descentThruster = extForceTorque.ExtForceTorque()
    descentThruster.ModelTag = "descentThruster"
    lander.addDynamicEffector(descentThruster)
    scSim.AddModelToTask(simTaskName, descentThruster)

    # Step 1: NRHO staging orbit -- start at perilune (north side)

    oe = orbitalMotion.ClassicElements()
    oe.a = (NRHO_PERILUNE_RADIUS_M + NRHO_APOLUNE_RADIUS_M) / 2.0
    oe.e = (NRHO_APOLUNE_RADIUS_M - NRHO_PERILUNE_RADIUS_M) / (NRHO_APOLUNE_RADIUS_M + NRHO_PERILUNE_RADIUS_M)
    oe.i = NRHO_INCLINATION_DEG * macros.D2R 
    oe.Omega = 0.0
    oe.omega = NRHO_OMEGA_DEG * macros.D2R 
    oe.f = 0.0
    r0, v0 = orbitalMotion.elem2rv(MU_MOON, oe)
    r0, v0 = np.array(r0).flatten(), np.array(v0).flatten()

    lander.hub.r_CN_NInit = r0.reshape(3, 1).tolist()
    lander.hub.v_CN_NInit = v0.reshape(3, 1).tolist()
    lander.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    lander.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

    samplingTime = macros.sec2nano(2.0)
    landerRec = lander.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, landerRec)

    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, lander, saveFile=fileName)

    scSim.InitializeSimulation()

    # prevent interval s helps with ignoring the basilisk progress bars 
    scSim.SetProgressBar(False)

    posRef = lander.dynManager.getStateObject(lander.hub.nameOfHubPosition)
    velRef = lander.dynManager.getStateObject(lander.hub.nameOfHubVelocity)

    t = 0.0
    print_mission_timeline()
    print("=" * 60)
    print("STEP 1: NRHO staging orbit")
    print(f"  perilune radius {NRHO_PERILUNE_RADIUS_M/1000:.1f} km, "
          f"apolune {NRHO_APOLUNE_RADIUS_M/1000:.1f} km, |v0|={np.linalg.norm(v0):.2f} m/s")
    print("sim time to NRHO perilune:  min")
    

    # Step 2: transfer to low lunar orbit -- burn A + coast + burn B

    print("=" * 60)
    print("STEP 2: transfer to low lunar orbit (LLO)")
    a1 = (np.linalg.norm(r0) + LLO_RADIUS_M) / 2.0
    v1AtR0 = np.sqrt(MU_MOON * (2.0 / np.linalg.norm(r0) - 1.0 / a1))
    vHat = v0 / np.linalg.norm(v0)
    dvA = v1AtR0 - np.linalg.norm(v0)
    v_after_A = apply_impulsive_burn(velRef, v0, dvA, vHat, "burn A (LOI-style, at NRHO perilune)")

    T1 = 2 * np.pi * np.sqrt(a1 ** 3 / MU_MOON)
    coastA = T1 / 2.0
    print(f"  coasting {coastA/60:.1f} min to reach transfer-orbit periapsis (LLO altitude)")
    t += coastA
    scSim.ConfigureStopTime(macros.sec2nano(t))
    scSim.ExecuteSimulation()

    r_now = np.array(posRef.getState()).flatten()
    v_now = np.array(velRef.getState()).flatten()
    v1Peri = np.sqrt(MU_MOON * (2.0 / LLO_RADIUS_M - 1.0 / a1))
    vCirc = np.sqrt(MU_MOON / LLO_RADIUS_M)
    vHatNow = v_now / np.linalg.norm(v_now)
    dvB = vCirc - np.linalg.norm(v_now)
    v_after_B = apply_impulsive_burn(velRef, v_now, dvB, vHatNow, "burn B (circularize into 100 km LLO)")

    T_LLO = 2 * np.pi * np.sqrt(LLO_RADIUS_M ** 3 / MU_MOON)
    coastB = T_LLO / 2.0
    print(f"  coasting {coastB/60:.1f} min (half the circular LLO period) to get back around "
          f"to the north side for DOI phasing")
    t += coastB
    scSim.ConfigureStopTime(macros.sec2nano(t))
    scSim.ExecuteSimulation()

    # ------------------------------------------------------------------
    # Step 3: descent orbit insertion (DOI) -- burn C
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: descent orbit insertion (DOI)")
    r_now = np.array(posRef.getState()).flatten()
    v_now = np.array(velRef.getState()).flatten()
    a2 = (LLO_RADIUS_M + PDI_RADIUS_M) / 2.0
    v2AtLLO = np.sqrt(MU_MOON * (2.0 / LLO_RADIUS_M - 1.0 / a2))
    vHatNow = v_now / np.linalg.norm(v_now)
    dvC = v2AtLLO - np.linalg.norm(v_now)
    v_after_C = apply_impulsive_burn(velRef, v_now, dvC, vHatNow, "burn C (DOI, lowers perilune to ~15 km)")

    T2 = 2 * np.pi * np.sqrt(a2 ** 3 / MU_MOON)

    total_orbit_lowering_dv = abs(dvA) + abs(dvB) + abs(dvC)
    print(f"  total delta-v for steps 2-3 (transfer + circularize + DOI): "
          f"{total_orbit_lowering_dv:.1f} m/s")

    # ------------------------------------------------------------------
    # Propellant tracking (Tsiolkovsky rocket equation). Guidance here is
    # acceleration-commanded (F = mass * aCmd), which means the trajectory
    # itself doesn't depend on mass at all -- but the PROPELLANT cost does,
    # so we track current mass explicitly and feed it back into both the
    # force command and Basilisk's own hub mass every step, for a real
    # depleting-mass simulation rather than a post-hoc estimate.
    # ------------------------------------------------------------------
    currentMass = lander_wet_mass
    totalDeltaV = total_orbit_lowering_dv

    def burn_propellant(dv_mps):
        nonlocal currentMass
        if currentMass <= lander_dry_mass:
            return 0.0
        propUsed_unclamped = currentMass * (1.0 - np.exp(-abs(dv_mps) / (isp_s * G0)))
        newMass = max(lander_dry_mass, currentMass - propUsed_unclamped)
        propUsed = currentMass - newMass
        currentMass = newMass
        lander.hub.mHub = currentMass
        return propUsed

    burn_propellant(abs(dvA))
    burn_propellant(abs(dvB))
    burn_propellant(abs(dvC))
    print(f"  propellant used for steps 2-3: {lander_wet_mass - currentMass:.1f} kg "
          f"(mass now {currentMass:.1f} kg, Isp={isp_s:.0f} s)")
    print(f"  propellant remaining at PDI start: {max(currentMass - lander_dry_mass, 0.0):.1f} kg")

    target_lat_deg = -89.45
    target_lon_deg = 222.0
    targetRadius, targetRadiusSource = get_surface_radius_m(target_lat_deg, target_lon_deg, dem_path)
    print(f"  target ({target_lat_deg:+.2f}°{'S' if target_lat_deg < 0 else 'N'}, {target_lon_deg:.2f}°{'W' if target_lon_deg < 0 else 'E'}) local radius: {targetRadius/1000:.2f} km ({targetRadiusSource})")

    def get_target_position(t_eval):
        theta = MOON_SPIN_RATE * t_eval
        c, s = np.cos(theta), np.sin(theta)
        target_lat_rad = np.radians(target_lat_deg)
        target_lon_rad = np.radians(target_lon_deg)
        target_r_fixed = targetRadius * np.array([
            np.cos(target_lat_rad) * np.cos(target_lon_rad),
            np.cos(target_lat_rad) * np.sin(target_lon_rad),
            np.sin(target_lat_rad),
        ])
        return np.array([
            c * target_r_fixed[0] - s * target_r_fixed[1],
            s * target_r_fixed[0] + c * target_r_fixed[1],
            target_r_fixed[2],
        ])

    def get_target_bearing(r_now, rHat, target_r):
        # Straight-line-chord tangent-plane projection (target_r - r_now,
        # minus its radial component) only approximates ground range/bearing
        # when the two points are close together. For large separations --
        # e.g. right after DOI burn C, where the lander sits near the north
        # pole and the target is near the south pole -- that chord is nearly
        # entirely radial, so subtracting the radial part leaves a tiny,
        # meaningless residual instead of the true ~5,470 km separation.
        # Use actual great-circle range (angle between position vectors x
        # radius) and project the TARGET's own position (not the chord) onto
        # the local tangent plane for bearing; both stay correct at any range.
        targetHat = target_r / np.linalg.norm(target_r)
        cosAngle = np.clip(np.dot(rHat, targetHat), -1.0, 1.0)
        angularSep = np.arccos(cosAngle)
        groundDist = angularSep * targetRadius

        dirRaw = target_r - np.dot(target_r, rHat) * rHat
        dirNorm = np.linalg.norm(dirRaw)
        if dirNorm < 1e-6:
            # Directly overhead or antipodal -- bearing is undefined at this
            # exact instant; caller should treat groundDist as authoritative
            # and fall back to whatever velocity-direction logic applies.
            dirHat = np.zeros(3)
        else:
            dirHat = dirRaw / dirNorm

        return groundDist, dirHat

    # ------------------------------------------------------------------
    # STEP 3.5: range-based PDI ignition.
    #
    # The old code always coasted ballistically all the way to the ~15 km
    # perilune of the DOI orbit before switching on guidance. But with this
    # orbit geometry perilune sits almost directly over the target -- so by
    # the time the engine lit, the lander had ~1600 m/s of horizontal speed
    # and only a few km of horizontal offset from the target to burn it off
    # in. At the current thrust/mass, available horizontal deceleration is
    # roughly a few m/s^2, so stopping ~1600 m/s physically requires on the
    # order of hundreds of km of ground track -- there is no guidance law
    # that closes that gap in a few km. Real PDI burns ignite far uprange of
    # the landing site for exactly this reason.
    #
    # So instead of coasting for a fixed time, coast step-by-step and ignite
    # as soon as the *current* required stopping distance (from current
    # speed and current available deceleration) is about to exceed the
    # actual remaining distance to the target. That way ignition timing
    # follows from the real thrust/mass numbers instead of an assumed
    # periapsis radius.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3.5: coasting to a range-based PDI ignition point")

    coast_dt = 2.0
    ignition_margin = 1.25  # ignite a bit early to leave margin for gravity losses / vertical control
    horiz_accel_fraction = 0.5  # fraction of total thrust authority assumed available for horizontal braking

    ignited = False
    coast_t_start = t
    max_coast_time = max(T2 * 1.5, 3600.0 * 4)  # generous ceiling so we don't loop forever

    while t < coast_t_start + max_coast_time:
        r_now = np.array(posRef.getState()).flatten()
        v_now = np.array(velRef.getState()).flatten()
        rMag = np.linalg.norm(r_now)
        rHat = r_now / rMag
        vHoriz = v_now - np.dot(v_now, rHat) * rHat
        hSpeedNow = np.linalg.norm(vHoriz)

        localRadius, lat, lon, source = get_local_radius(r_now, t, dem_path)
        altitude = rMag - localRadius

        target_r = get_target_position(t)
        targetHorizDist, _ = get_target_bearing(r_now, rHat, target_r)

        max_engine_accel_now = (thrust_N * num_thrusters) / max(currentMass, 1e-9)
        avail_decel_now = horiz_accel_fraction * max_engine_accel_now
        stopping_distance = (hSpeedNow ** 2) / (2.0 * max(avail_decel_now, 1e-6))

        # Safety floor: never let the unpowered coast run the lander into
        # the ground or below a sane minimum ignition altitude, even if the
        # range condition hasn't triggered yet.
        min_ignition_altitude_m = 20000.0

        if stopping_distance * ignition_margin >= targetHorizDist or altitude <= min_ignition_altitude_m:
            ignited = True
            print(f"  [ignition] t={t/60:.1f} min  alt={altitude/1000:.1f} km  "
                  f"speed={hSpeedNow:.1f} m/s  target dist={targetHorizDist/1000:.1f} km  "
                  f"required stopping dist={stopping_distance/1000:.1f} km")
            break

        if altitude <= 0.0:
            print(f"[warning] reached the surface during the unpowered DOI coast at t={t/60:.2f} min "
                  f"before PDI ignition -- ignition criteria never satisfied in time")
            break

        t += coast_dt
        scSim.ConfigureStopTime(macros.sec2nano(t))
        scSim.ExecuteSimulation()

    if not ignited:
        print("[warning] PDI ignition condition never triggered within the coast safety window; "
              "starting active guidance now regardless")

    print("=" * 60)
    print("STEPS 4-6: powered descent, braking, approach, terminal descent")
    lastPrint = t
    altHistory, speedHistory, timeHistory, latHistory, lonHistory = [], [], [], [], []
    attitudeHistory, massHistory = [], []
    touchdown = False
    speedFinal = latFinal = lonFinal = sourceFinal = None
    horizSpeedFinal = vertSpeedFinal = None
    phase = "brake"
    transitionStartT = None
    aCmd_prev = np.zeros(3)

    tStop = t + 3600.0 * 2  # 2 hours for the whole phase

    while t < tStop:
        r_now = np.array(posRef.getState()).flatten()
        v_now = np.array(velRef.getState()).flatten()
        rMag = np.linalg.norm(r_now)
        rHat = r_now / rMag

        localRadius, lat, lon, source = get_local_radius(r_now, t, dem_path)
        altitude = rMag - localRadius
        speed = np.linalg.norm(v_now)
        vVert = np.dot(v_now, rHat) * rHat
        vHoriz = v_now - vVert
        hSpeedNow = np.linalg.norm(vHoriz)

        altHistory.append(altitude)
        speedHistory.append(speed)
        timeHistory.append(t)
        latHistory.append(lat)
        lonHistory.append(lon)
        massHistory.append(currentMass)

        if t - lastPrint >= print_interval_s:
            propellantRemaining = max(currentMass - lander_dry_mass, 0.0)
            print(f"  t={t/60:6.1f} min  alt={altitude/1000:8.2f} km  speed={speed:7.2f} m/s  "
                f"lat={lat:7.2f} deg  lon={lon:7.2f} deg  phase={phase}  mass={currentMass:.1f} kg  "
                f"prop={propellantRemaining:.1f} kg")
            lastPrint = t

        if altitude <= 0.0:
            touchdown = True
            speedFinal = speed
            latFinal, lonFinal, sourceFinal = lat, lon, source
            vertSpeedFinal = abs(np.dot(v_now, rHat))
            horizSpeedFinal = hSpeedNow
            save_touchdown_state(
                os.path.join(os.path.dirname(__file__), "descent_state.json"),
                lander_dry_mass,
                currentMass,
                r_now,
                v_now,
                t,
                latFinal,
                lonFinal,
                sourceFinal,
            )
            break

        if currentMass <= lander_dry_mass + 1e-6:
            descentThruster.extForce_N = [[0.0], [0.0], [0.0]]
            print(f"[warning] propellant depleted at t={t/60:.2f} min; ending PDI burn at dry mass {lander_dry_mass:.1f} kg")
            break

        if phase == "brake" and (hSpeedNow < 5.0 or altitude < approach_altitude_m):
            phase = "terminal"
            transitionStartT = t
            print(f"  [phase change] brake -> terminal at t={t/60:.1f} min, "
                  f"alt={altitude/1000:.2f} km, horiz speed={hSpeedNow:.1f} m/s")

        target_r = get_target_position(t)

        toTarget = target_r - r_now
        targetHorizDist, targetHorizDir = get_target_bearing(r_now, rHat, target_r)

        if phase == "brake" and targetHorizDist < 1500.0:
            phase = "terminal"
            transitionStartT = t
            print(f"  [phase change] brake -> terminal (target lock) at t={t/60:.1f} min, "
                  f"alt={altitude/1000:.2f} km, target horiz dist={targetHorizDist:.0f} m")

        gVec = -MU_MOON * r_now / rMag ** 3

        # Required-deceleration horizontal law: command exactly the
        # acceleration that, at a constant rate, would bring the along-track
        # speed to zero exactly as range reaches zero (v^2 = 2*a*d, solved
        # for a and re-evaluated closed-loop every step so it self-corrects
        # for gravity, mass depletion, and bearing drift). This replaces an
        # earlier capped-speed P-controller that had a stable fixed point
        # around 88 m/s regardless of remaining range -- fine at the ~16 km
        # terminal ranges it was tuned for, but at the hundreds-of-km ranges
        # PDI now legitimately ignites at, that fixed point meant the burn
        # cruised at a near-constant speed far too slow to close the
        # distance before running out of propellant. This law instead
        # spreads deceleration across however much range is actually left.
        v_along = np.dot(vHoriz, targetHorizDir)
        v_cross = vHoriz - v_along * targetHorizDir
        a_req_along = (max(v_along, 0.0) ** 2) / (2.0 * max(targetHorizDist, 50.0))
        cross_track_gain = 0.3  # damps any lateral drift off the direct bearing to target
        aCmd_horiz = -targetHorizDir * a_req_along - cross_track_gain * v_cross

        aCmd_brake = (-gVec
                      + aCmd_horiz
                      + 0.8 * (vsink_target - np.dot(v_now, rHat)) * rHat)

        dirHat = toTarget / np.linalg.norm(toTarget)
        vDesired_mag = k_speed * np.sqrt(max(altitude, 1.0))
        vDesired_terminal = dirHat * vDesired_mag

        aCmd_terminal = -gVec + terminal_gain * (vDesired_terminal - v_now)

        if phase == "brake":
            aCmd = aCmd_brake
        else:
            # smoothstep blend from brake law to terminal law over
            # blend_duration_s, zero slope at both ends so it doesn't
            # itself introduce a kink
            w = np.clip((t - transitionStartT) / blend_duration_s, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            aCmd = (1.0 - w) * aCmd_brake + w * aCmd_terminal

        # NOTE this is to prevent the commanded acceleration from changing too fast for the people 
        maxDelta = max_jerk * dt_guidane
        delta = aCmd - aCmd_prev
        deltaMag = np.linalg.norm(delta)
        if deltaMag > maxDelta:
            aCmd = aCmd_prev + delta / deltaMag * maxDelta
        aCmd_prev = aCmd.copy()

        aMag = np.linalg.norm(aCmd)
        max_engine_accel = (thrust_N * num_thrusters) / max(currentMass, 1e-9)
        if aMag > max_engine_accel:
            aCmd = aCmd / aMag * max_engine_accel
            aMag = np.linalg.norm(aCmd)

        # 90 deg is wtright 0 deg is horizontal 
        cosFromVertical = np.clip(np.dot(aCmd / max(aMag, 1e-9), rHat), -1.0, 1.0)
        pitchFromHorizontal = 90.0 - np.degrees(np.arccos(cosFromVertical))
        attitudeHistory.append(pitchFromHorizontal)

        burn_propellant(aMag * dt_guidane)
        totalDeltaV += aMag * dt_guidane
        descentThruster.extForce_N = (currentMass * aCmd).reshape(3, 1).tolist()

        t += dt_guidane
        scSim.ConfigureStopTime(macros.sec2nano(t))
        scSim.ExecuteSimulation()

    print("=" * 60)
    if touchdown:
        print(f"TOUCHDOWN at t={t/60:.2f} min")
        print("  mission marker: Day 8, Hour 22:00 (Touchdown): The lander touches down safely near the lunar South Pole.")
        print(f"  latitude  = {latFinal:.3f} deg")
        print(f"  longitude = {lonFinal:.3f} deg")
        print(f"  speed = {speedFinal:.2f} m/s  (horizontal {horizSpeedFinal:.2f}, vertical {vertSpeedFinal:.2f})")
        print(f"  surface source: {sourceFinal}")
        print("  -> soft landing" if speedFinal < 3.0 else "  -> HARD landing")

        propellantUsed = lander_wet_mass - currentMass
        print(f"  total delta-v: {totalDeltaV:.1f} m/s")
        print(f"  propellant used: {propellantUsed:.1f} kg "
              f"({100*propellantUsed/lander_wet_mass:.1f}% of the {lander_wet_mass:.0f} kg starting mass, "
              f"Isp={isp_s:.0f} s)")
        print(f"  final mass: {currentMass:.1f} kg")

        slopeDeg, slopeSource = get_local_slope_deg(latFinal, lonFinal, dem_path)

        # Convert the stored pitch-from-horizontal value into a tilt-from-vertical
        # angle and use the local gravitational acceleration magnitude at touchdown.
        last_attitude_deg = attitudeHistory[-1] if attitudeHistory else 0.0
        attitude_tilt_deg = abs(90.0 - last_attitude_deg)
        g_local = float(MU_MOON / max(rMag**2, 1e-12))

        stability = assess_landing_stability(
            slope_deg=slopeDeg,
            horizontal_speed_mps=horizSpeedFinal,
            vertical_speed_mps=vertSpeedFinal,
            leg_radius_m=LEG_RADIUS,
            cg_height_m=COG_HEIGHT,
            attitude_tilt_deg=attitude_tilt_deg,
            g_local=g_local,
        )
        print(f"  local slope: {slopeDeg:.2f} deg ({slopeSource})")
        print(f"  tip-over screen: critical angle {stability['theta_crit_deg']:.1f} deg, "
              f"effective tilt {stability['effective_tilt_deg']:.1f} deg, "
              f"margin {stability['margin_deg']:+.1f} deg -> "
              f"{'STABLE (rough estimate)' if stability['stable'] else 'AT RISK OF TIPPING (rough estimate)'}")
    else:
        print(f"[warning] did not reach the surface within the {tStop/3600:.1f} hr ceiling")

    posData = landerRec.r_BN_N

    plt.close("all")
    figureList = {}

    ax = plt.axes(projection='3d')
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:40j]
    R = MEAN_RADIUS_M / 1000.0
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_surface(x, y, z, color='#888888', alpha=0.6)
    ax.plot3D(posData[:, 0] / 1000, posData[:, 1] / 1000, posData[:, 2] / 1000, color='orangered', label='Lander')
    lim = NRHO_PERILUNE_RADIUS_M / 1000.0 * 1.3
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title('NRHO to south pole: full trajectory')
    ax.legend()
    figureList[fileName + "_3d"] = plt.figure(1)

    if len(timeHistory) > 1:
        fig, axs = plt.subplots(5, 1, figsize=(8, 13), sharex=True)
        tHrs = np.array(timeHistory) / 3600.0
        tHrsAtt = tHrs[:len(attitudeHistory)]

        axs[0].plot(tHrs, np.array(altHistory) / 1000.0, color='orangered')
        axs[0].set_ylabel('Altitude [km]')
        axs[0].axhline(0, color='k', linewidth=0.5)
        axs[0].set_title('Powered descent, braking, and terminal approach')

        axs[1].plot(tHrs, speedHistory, color='deepskyblue')
        axs[1].set_ylabel('Speed [m/s]')

        axs[2].plot(tHrsAtt, attitudeHistory, color='tab:orange')
        axs[2].set_ylabel('Attitude [deg]')
        axs[2].axhline(90, color='k', linewidth=0.5, linestyle='--')
        axs[2].text(tHrsAtt[0] if len(tHrsAtt) else 0, 92, 'vertical', fontsize=8, color='gray')

        axs[3].plot(tHrs, latHistory, label='latitude [deg]', color='tab:green')
        axs[3].plot(tHrs, lonHistory, label='longitude [deg]', color='tab:purple')
        axs[3].set_ylabel('Lat / Lon [deg]')
        axs[3].legend()

        axs[4].plot(tHrs, massHistory, color='tab:brown')
        axs[4].set_ylabel('Mass [kg]')
        axs[4].set_xlabel('Time since PDI start [hr]')

        plt.tight_layout()
        figureList[fileName + "_descent"] = plt.figure(2)

    if show_plots:
        plt.show()
    plt.close("all")

    return figureList

if __name__ == "__main__":
    thruster_mass = 82.0
    run(
        show_plots=True,
        dem_path="LDEM_875S_5M.IMG",
        lander_dry_mass=5300,
        lander_wet_mass=19300,
        lander_inertia=(1500., 420., 300.),
        thrust_N=24500.0,
        num_thrusters=2,
        brake_gain=0.05,
        vsink_target=-5.0,

        approach_altitude_m=2500.0,

        terminal_gain=1.0,
        k_speed=0.5,

        max_jerk=4.0, #6 mps^3is passenger comfort 
        blend_duration_s=20.0,

        sim_time_step_s=0.1,
        print_interval_s=30.0,

        isp_s=339.0,
    )