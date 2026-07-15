# Companion ascent simulation that starts from the saved touchdown state
# produced by descent_sim.py and uses the remaining mass as the ascent-stage
# propellant budget.

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

from lola_terrain import get_surface_radius_m, MEAN_RADIUS_M


bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

MU_MOON = 4.9048695e12
MOON_SPIN_RATE = 2.0 * np.pi / (27.321661 * 86400.0)
G0 = 9.80665

NRHO_PERILUNE_RADIUS_M = 3366.0e3
NRHO_APOLUNE_RADIUS_M = 71000.0e3
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


def apply_impulsive_burn(
    vel_ref,
    current_velocity,
    delta_v,
    direction_hat,
    label,
):
    new_velocity = current_velocity + delta_v * direction_hat
    vel_ref.setState(new_velocity.reshape(3, 1).tolist())

    print(f"[{label}] deltaV = {delta_v:.2f} m/s")

    return new_velocity


def burn_propellant(current_mass, dv_mps, isp_s):
    if current_mass <= 0.0:
        return 0.0, 0.0

    mass_ratio = np.exp(-abs(dv_mps) / (isp_s * G0))
    new_mass = current_mass * mass_ratio
    propellant_used = current_mass - new_mass

    return new_mass, propellant_used


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
    blend_duration_s,
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

    lander.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
        [
            ascent_inertia[0], 0.0, 0.0,
            0.0, ascent_inertia[1], 0.0,
            0.0, 0.0, ascent_inertia[2],
        ]
    )

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
    oe = orbitalMotion.ClassicElements()
    oe.a = (
        NRHO_PERILUNE_RADIUS_M + NRHO_APOLUNE_RADIUS_M
    ) / 2.0

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

    position_ref = lander.dynManager.getStateObject(
        lander.hub.nameOfHubPosition
    )

    velocity_ref = lander.dynManager.getStateObject(
        lander.hub.nameOfHubVelocity
    )

    t = 0.0
    last_print = -print_interval_s

    print("=" * 60)
    print("ASCENT STAGE: return from surface to NRHO")

    altitude_history = []
    speed_history = []
    time_history = []
    mass_history = []

    phase = "lift"

    # Start the jerk limiter with enough upward acceleration to avoid
    # limiting the initial thrust below lunar gravity.
    a_cmd_previous = 2.5 * r0_hat

    total_delta_v = 0.0
    has_lifted_off = False

    lift_off_clearance_m = 1.0
    impact_tolerance_m = -1.0

    t_stop = 6.0 * 3600.0

    while t < t_stop and current_mass > 1.0:
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

        latitude_deg = np.degrees(
            np.arcsin(
                np.clip(
                    r_now[2] / r_magnitude,
                    -1.0,
                    1.0,
                )
            )
        )

        longitude_deg = np.degrees(
            np.arctan2(
                r_now[1],
                r_now[0],
            )
        )

        # get_surface_radius_m returns two values in lola_terrain.py.
        local_radius, _ = get_surface_radius_m(
            latitude_deg,
            longitude_deg,
            dem_path,
        )

        altitude = r_magnitude - local_radius
        speed = np.linalg.norm(v_now)

        if altitude > lift_off_clearance_m:
            has_lifted_off = True

        # Do not stop just because the vehicle begins at altitude zero.
        # Only stop for terrain impact after it has already lifted off.
        if has_lifted_off and altitude < impact_tolerance_m:
            print(
                f"Terrain impact at t={t:.2f} s, "
                f"altitude={altitude:.2f} m."
            )
            break

        if t - last_print >= print_interval_s:
            print(
                f"  t={t / 60.0:6.1f} min"
                f"  alt={altitude / 1000.0:8.2f} km"
                f"  speed={speed:7.2f} m/s"
                f"  mass={current_mass:.1f} kg"
                f"  phase={phase}"
            )

            last_print = t

        gravity_vector = (
            -MU_MOON
            * r_now
            / r_magnitude**3
        )

        if phase == "lift":
            raw_command = (
                -gravity_vector
                + 7.0 * r_hat
                + 0.25 * (v_target - v_now)
            )

            # Switch after reaching approximately 100 km altitude.
            if altitude >= 100.0e3:
                phase = "transfer"

        elif phase == "transfer":
            position_error = r_target - r_now
            position_error_hat = position_error / max(
                np.linalg.norm(position_error),
                1.0,
            )

            raw_command = (
                -gravity_vector
                + 0.35 * (v_target - v_now)
                + 0.25 * position_error_hat
            )

            if r_magnitude >= NRHO_APOLUNE_RADIUS_M * 0.6:
                phase = "capture"

        else:
            raw_command = (
                -gravity_vector
                + 0.5 * (v_target - v_now)
            )

        # Gradually blend from a safe vertical lift command into the
        # orbital guidance command.
        blend_factor = np.clip(
            t / max(blend_duration_s, dt),
            0.0,
            1.0,
        )

        minimum_lift_command = (
            -gravity_vector
            + 2.0 * r_hat
        )

        a_cmd = (
            minimum_lift_command
            + blend_factor
            * (raw_command - minimum_lift_command)
        )

        # Limit acceleration command rate.
        maximum_acceleration_change = max_jerk * dt
        command_change = a_cmd - a_cmd_previous
        command_change_magnitude = np.linalg.norm(command_change)

        if (
            command_change_magnitude
            > maximum_acceleration_change
            > 0.0
        ):
            a_cmd = (
                a_cmd_previous
                + command_change
                / command_change_magnitude
                * maximum_acceleration_change
            )

        # Apply the engine acceleration limit from thrust and thruster count.
        acceleration_magnitude = np.linalg.norm(a_cmd)
        max_engine_accel = (thrust_N * num_thrusters) / max(current_mass, 1e-9)

        if acceleration_magnitude > max_engine_accel:
            a_cmd = (
                a_cmd
                / acceleration_magnitude
                * max_engine_accel
            )
            acceleration_magnitude = max_engine_accel

        a_cmd_previous = a_cmd.copy()

        mass_before_burn = current_mass

        current_mass, propellant_used = burn_propellant(
            current_mass,
            acceleration_magnitude * dt,
            isp_s,
        )

        # Stop at 1 kg instead of allowing negative or zero mass.
        if current_mass < 1.0:
            current_mass = 1.0

        average_mass = 0.5 * (
            mass_before_burn + current_mass
        )

        ascent_thruster.extForce_N = (
            average_mass * a_cmd
        ).reshape(3, 1).tolist()

        # Update the spacecraft mass as propellant is consumed.
        lander.hub.mHub = current_mass

        total_delta_v += acceleration_magnitude * dt

        altitude_history.append(altitude)
        speed_history.append(speed)
        time_history.append(t)
        mass_history.append(current_mass)

        t += dt

        sc_sim.ConfigureStopTime(
            macros.sec2nano(t)
        )

        sc_sim.ExecuteSimulation()

    # Turn off thrust at the end.
    ascent_thruster.extForce_N = [
        [0.0],
        [0.0],
        [0.0],
    ]

    print("=" * 60)
    print(
        f"Ascent complete. final mass = {current_mass:.1f} kg, "
        f"total delta-v = {total_delta_v:.1f} m/s"
    )

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
            3,
            1,
            figsize=(8, 8),
            sharex=True,
        )

        time_hours = (
            np.asarray(time_history) / 3600.0
        )

        axes[0].plot(
            time_hours,
            np.asarray(altitude_history) / 1000.0,
            color="orangered",
        )
        axes[0].set_ylabel("Altitude [km]")
        axes[0].grid(True)

        axes[1].plot(
            time_hours,
            speed_history,
            color="deepskyblue",
        )
        axes[1].set_ylabel("Speed [m/s]")
        axes[1].grid(True)

        axes[2].plot(
            time_hours,
            mass_history,
            color="tab:brown",
        )
        axes[2].set_ylabel("Mass [kg]")
        axes[2].set_xlabel("Time [hr]")
        axes[2].grid(True)

        plt.tight_layout()

        figure_list[fileName + "_ascent"] = ascent_figure

    if show_plots:
        plt.show()

    plt.close("all")

    return figure_list


if __name__ == "__main__":
    run(
        show_plots=True,
        dem_path="LDEM_875S_5M.IMG",
        ascent_mass=4700.0,
        ascent_inertia=(1500.0, 420.0, 300.0),
        thrust_N=24500.0,
        num_thrusters=1,
        isp_s=339.0,
        sim_time_step_s=0.1,
        print_interval_s=30.0,
        max_jerk=4.0,
        blend_duration_s=20.0,
    )