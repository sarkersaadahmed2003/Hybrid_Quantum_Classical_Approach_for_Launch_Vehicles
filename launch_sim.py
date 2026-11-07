"""
launch_sim.py
=============
Sample program for the four-phase launch-vehicle autonomy framework.

What it contains (mapped to the phases in the abstract):

  Phase 1  Simulation   - pitch-plane rigid-body dynamics (3-DOF: x, z, pitch)
                          with time-varying mass and inertia, moving CG vs CP,
                          stratified atmosphere with wind, thrust curve, Isp,
                          throttle lag, gimbal rate limit and lag.
  Phase 2  GNC          - Navigation : linear Kalman filter fusing IMU, GPS, baro
                          Guidance   : open-loop pitch program (gravity-turn style)
                          Control    : gain-scheduled PID on pitch -> gimbal angle
  Phase 3  FDIR         - GPS fault detection with a chi-square innovation test,
                          isolation (stop using GPS), recovery (IMU + baro only);
                          engine under-thrust detection and a safe-mode pitch hold.
  Analysis Monte Carlo  - dispersed runs, impact-point footprint, fault statistics.

Simplifications (be honest about these in the book):
  * Pitch plane only (3-DOF). A full 6-DOF adds yaw/roll and quaternions
    (Zipfel Ch. 10.1). The code is organised so that upgrade is local.
  * Flat, non-rotating Earth with gravity that falls off with altitude.
  * Attitude is measured directly with noise (no attitude filter).
  * Quantum parts are NOT implemented. See the section at the bottom for the
    exact functions where a quantum/quantum-inspired method would plug in.

Run:   python launch_sim.py            (one nominal flight + 200 Monte Carlo runs)
Needs: numpy, matplotlib
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

G0 = 9.80665          # standard gravity, m/s^2
R_EARTH = 6_371_000.0  # m
DEG = math.pi / 180.0


# =============================================================================
# PHASE 1 - ENVIRONMENT AND VEHICLE MODELS
# =============================================================================
def atmosphere(h):
    """Stratified atmosphere: ISA troposphere (0-11 km), isothermal layer above.
    Returns density (kg/m^3), temperature (K), pressure (Pa)."""
    h = max(h, 0.0)
    if h < 11_000.0:
        T = 288.15 - 0.0065 * h
        p = 101_325.0 * (T / 288.15) ** 5.2559
    else:
        T = 216.65
        p = 22_632.0 * math.exp(-(h - 11_000.0) / 6341.6)
    rho = p / (287.05 * T)
    return rho, T, p


def wind(h, w_ref):
    """Horizontal wind (m/s) with a jet-stream-like peak near 10 km."""
    if 0.0 < h < 20_000.0:
        return w_ref * math.sin(math.pi * h / 20_000.0)
    return 0.0


def gravity(h):
    return G0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2


@dataclass
class Vehicle:
    m_dry: float = 150.0        # kg
    m_prop: float = 350.0       # kg
    isp: float = 230.0          # s
    burn_time: float = 60.0     # s (nominal)
    length: float = 6.0         # m
    diameter: float = 0.40      # m
    cd: float = 0.40            # axial drag coefficient
    cn_alpha: float = 2.0       # normal-force slope, 1/rad
    cmq: float = 10.0           # pitch-damping coefficient
    x_cp: float = 4.2           # centre of pressure from nose, m
    x_cg_full: float = 3.6      # CG from nose with full tanks, m
    x_cg_empty: float = 3.0     # CG from nose when empty, m
    gimbal_max: float = 5.0 * DEG
    gimbal_rate: float = 20.0 * DEG   # rad/s
    gimbal_tau: float = 0.05          # s, actuator lag
    throttle_tau: float = 0.20        # s, engine throttle response
    thrust_scale: float = 1.0         # 1.0 = healthy; <1 models a fault

    @property
    def area(self):
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def thrust_level(self):
        """Nominal full-throttle thrust, N."""
        return self.isp * G0 * self.m_prop / self.burn_time

    def thrust_shape(self, t):
        """Thrust curve: 0.5 s start-up ramp, then mildly regressive."""
        ramp = min(t / 0.5, 1.0)
        return ramp * (1.05 - 0.10 * t / self.burn_time)

    def mass_props(self, m):
        """CG position and pitch inertia for the current mass."""
        frac = min(max((m - self.m_dry) / self.m_prop, 0.0), 1.0)
        x_cg = self.x_cg_empty + (self.x_cg_full - self.x_cg_empty) * frac
        inertia = m * self.length ** 2 / 12.0
        return x_cg, inertia


# State vector s = [x, z, vx, vz, theta, q, m]
#   x  downrange (m), z altitude (m), theta pitch from horizontal (rad),
#   q  pitch rate (rad/s), m mass (kg)
def dynamics(t, s, delta, throttle, veh, w_ref):
    """Equations of motion. Returns (ds/dt, specific force, dynamic pressure)."""
    x, z, vx, vz, th, q, m = s
    rho, _, _ = atmosphere(z)
    vrx, vrz = vx - wind(z, w_ref), vz           # velocity relative to air
    V = math.hypot(vrx, vrz)
    qbar = 0.5 * rho * V * V

    bx, bz = math.cos(th), math.sin(th)          # body axis (nose direction)
    nx, nz = -math.sin(th), math.cos(th)         # body normal

    x_cg, inertia = veh.mass_props(m)
    Fx = Fz = M = 0.0

    # --- aerodynamics: axial drag, normal force at CP, pitch damping ---
    if V > 1.0:
        gamma = math.atan2(vrz, vrx)
        alpha = (th - gamma + math.pi) % (2 * math.pi) - math.pi
        alpha = max(-0.5, min(0.5, alpha))
        D = qbar * veh.area * veh.cd
        Fx -= D * vrx / V
        Fz -= D * vrz / V
        N = qbar * veh.area * veh.cn_alpha * alpha
        Fx += N * nx
        Fz += N * nz
        M -= (veh.x_cp - x_cg) * N                         # CP behind CG -> restoring
        M -= veh.cmq * qbar * veh.area * veh.length ** 2 * q / (2.0 * V)

    # --- propulsion with gimballed nozzle at the tail ---
    mdot = 0.0
    if m > veh.m_dry:
        T = throttle * veh.thrust_level * veh.thrust_shape(t) * veh.thrust_scale
        mdot = -T / (veh.isp * G0)
        cd_, sd_ = math.cos(delta), math.sin(delta)
        Fx += T * (cd_ * bx - sd_ * nx)
        Fz += T * (cd_ * bz - sd_ * nz)
        M += (veh.length - x_cg) * T * sd_

    g = gravity(z)
    ax, az = Fx / m, Fz / m - g
    deriv = np.array([vx, vz, ax, az, q, M / inertia, mdot])
    specific_force = (Fx / m, Fz / m)            # what an accelerometer measures
    return deriv, specific_force, qbar


def rk4_step(t, s, dt, delta, throttle, veh, w_ref):
    k1, f, qbar = dynamics(t, s, delta, throttle, veh, w_ref)
    k2, _, _ = dynamics(t + dt / 2, s + dt / 2 * k1, delta, throttle, veh, w_ref)
    k3, _, _ = dynamics(t + dt / 2, s + dt / 2 * k2, delta, throttle, veh, w_ref)
    k4, _, _ = dynamics(t + dt, s + dt * k3, delta, throttle, veh, w_ref)
    s_new = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    s_new[6] = max(s_new[6], veh.m_dry)
    return s_new, f, qbar


# =============================================================================
# PHASE 2 - NAVIGATION (Kalman filter), GUIDANCE, CONTROL
# =============================================================================
CHI2_2DOF_999 = 13.82   # 99.9% threshold for a 2-D innovation test


class NavFilter:
    """Linear Kalman filter, state [x, z, vx, vz].
    Predict with accelerometer (specific force + gravity).
    Update with GPS position and barometric altitude.
    Process noise uses the discrete white-noise-acceleration model
    (Labbe Ch. 7.5)."""

    def __init__(self, x0, sigma_acc=0.15):
        self.x = np.array(x0, dtype=float)
        self.P = np.diag([10.0, 10.0, 1.0, 1.0]) ** 2
        self.sigma_acc = sigma_acc
        # FDIR state for GPS
        self.gps_reject_count = 0
        self.gps_isolated = False
        self.gps_isolated_time = None

    def predict(self, f_meas, dt):
        g = gravity(self.x[1])
        a = np.array([f_meas[0], f_meas[1] - g])
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        B = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.x = F @ self.x + B @ a
        Q = B @ B.T * self.sigma_acc ** 2
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return y, S

    def update_gps(self, z_meas, t, sigma=(5.0, 8.0)):
        """GPS update with FDIR: detect -> isolate -> recover."""
        if self.gps_isolated:
            return                                   # recovered mode: GPS ignored
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.diag(sigma) ** 2
        y = z_meas - H @ self.x
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))       # normalised innovation squared
        if nis > CHI2_2DOF_999:
            self.gps_reject_count += 1               # detection: reject this sample
            if self.gps_reject_count >= 5:           # isolation: persistent fault
                self.gps_isolated = True
                self.gps_isolated_time = t
            return
        self.gps_reject_count = 0
        self._update(z_meas, H, R)

    def update_baro(self, h_meas):
        sigma = 2.0 + self.x[1] / 2000.0             # baro worsens with altitude
        H = np.array([[0, 1, 0, 0]], float)
        self._update(np.array([h_meas]), H, np.array([[sigma ** 2]]))


def guidance_pitch(t, safe_mode):
    """Open-loop pitch program. Vertical rise, pitch-over, hold.
    In safe mode (engine fault) hold a steeper attitude to keep altitude."""
    if t < 5.0:
        cmd = 90.0
    elif t < 45.0:
        cmd = 90.0 - (90.0 - 55.0) * (t - 5.0) / 40.0
    else:
        cmd = 55.0
    if safe_mode:
        cmd = max(cmd, 75.0)
    return cmd * DEG


class PitchController:
    """PID on pitch error, gains scheduled on control effectiveness.
    Control effectiveness b = (moment arm * thrust) / inertia, which grows
    about 3x during the burn as mass and inertia drop."""

    def __init__(self, wn=3.0, zeta=0.7, ki_ratio=0.3):
        self.wn, self.zeta, self.ki_ratio = wn, zeta, ki_ratio
        self.integral = 0.0

    def command(self, theta_cmd, theta_est, q_est, b_est, dt):
        if b_est < 1.0:                     # no thrust -> no control authority
            return 0.0
        kp = self.wn ** 2 / b_est
        kd = 2 * self.zeta * self.wn / b_est
        ki = self.ki_ratio * kp
        err = theta_cmd - theta_est
        self.integral = max(-0.2, min(0.2, self.integral + err * dt))
        return kp * err - kd * q_est + ki * self.integral


def gimbal_actuator(delta, delta_cmd, veh, dt):
    """Saturation, first-order lag and rate limit."""
    target = max(-veh.gimbal_max, min(veh.gimbal_max, delta_cmd))
    rate = (target - delta) / veh.gimbal_tau
    rate = max(-veh.gimbal_rate, min(veh.gimbal_rate, rate))
    return delta + rate * dt


# =============================================================================
# ONE FLIGHT: plant + sensors + GNC + FDIR
# =============================================================================
@dataclass
class FlightConfig:
    wind_ref: float = 0.0
    gps_fault_time: float = None      # time a 300 m GPS bias appears
    engine_fault_time: float = None   # time thrust drops to 60 %
    seed: int = 0
    dt_burn: float = 0.01
    dt_coast: float = 0.02


def fly(veh, cfg, log=False):
    rng = np.random.default_rng(cfg.seed)
    s = np.array([0.0, 0.0, 0.0, 0.0, 90.0 * DEG, 0.0, veh.m_dry + veh.m_prop])
    nav = NavFilter([0.0, 0.0, 0.0, 0.0])
    ctrl = PitchController()
    accel_bias = rng.normal(0.0, 0.02, 2)

    t, delta, throttle = 0.0, 0.0, 0.0
    f_meas = (0.0, G0)
    m_est = s[6]                                  # flight computer's mass model
    safe_mode, engine_fault_time_detected, low_thrust_timer = False, None, 0.0
    apogee, max_qbar, max_nav_err = 0.0, 0.0, 0.0
    hist = []
    k = 0

    while True:
        burning = s[6] > veh.m_dry + 1e-6
        dt = cfg.dt_burn if burning else cfg.dt_coast

        # ---- fault injection (truth side) ----
        if cfg.engine_fault_time is not None and t >= cfg.engine_fault_time:
            veh.thrust_scale = 0.6

        # ---- sensors ----
        theta_meas = s[4] + rng.normal(0, 0.2 * DEG)
        q_meas = s[5] + rng.normal(0, 0.5 * DEG)

        # ---- navigation ----
        nav.predict(f_meas, dt)
        if k % 10 == 0:                           # GPS at 10 Hz (dt_burn)
            gps = s[0:2] + rng.normal(0, [5.0, 8.0])
            if cfg.gps_fault_time is not None and t >= cfg.gps_fault_time:
                gps = gps + np.array([300.0, -300.0])
            nav.update_gps(gps, t)
        if k % 5 == 0 and s[1] < 40_000:          # baro at 20 Hz, below 40 km
            nav.update_baro(s[1] + rng.normal(0, 2.0 + s[1] / 2000.0))

        # ---- flight-computer models of thrust and mass ----
        throttle_cmd = 1.0
        rho_e, _, _ = atmosphere(nav.x[1])
        qbar_est = 0.5 * rho_e * (nav.x[2] ** 2 + nav.x[3] ** 2)
        if qbar_est > 25_000.0:
            throttle_cmd = 0.8                    # throttle bucket near max-q
        T_expected = 0.0
        if m_est > veh.m_dry:
            T_expected = throttle * veh.thrust_level * veh.thrust_shape(t)
            m_est = max(veh.m_dry, m_est - T_expected / (veh.isp * G0) * dt)

        # ---- FDIR: engine under-thrust ----
        axial_f = f_meas[0] * math.cos(s[4]) + f_meas[1] * math.sin(s[4])
        if T_expected > 0.3 * veh.thrust_level and t > 2.0 and not safe_mode:
            ratio = axial_f * m_est / T_expected
            low_thrust_timer = low_thrust_timer + dt if ratio < 0.7 else 0.0
            if low_thrust_timer > 0.5:
                safe_mode = True
                engine_fault_time_detected = t

        # ---- guidance and control ----
        theta_cmd = guidance_pitch(t, safe_mode)
        x_cg, inertia = veh.mass_props(m_est)
        b_est = (veh.length - x_cg) * T_expected / inertia
        delta_cmd = ctrl.command(theta_cmd, theta_meas, q_meas, b_est, dt)
        delta = gimbal_actuator(delta, delta_cmd, veh, dt)
        throttle += (throttle_cmd - throttle) * dt / veh.throttle_tau

        # ---- propagate truth ----
        s, f_true, qbar = rk4_step(t, s, dt, delta, throttle, veh, cfg.wind_ref)
        f_meas = (f_true[0] + accel_bias[0] + rng.normal(0, 0.05),
                  f_true[1] + accel_bias[1] + rng.normal(0, 0.05))
        t += dt
        k += 1

        apogee = max(apogee, s[1])
        max_qbar = max(max_qbar, qbar)
        nav_err = math.hypot(nav.x[0] - s[0], nav.x[1] - s[1])
        if t < 120.0:
            max_nav_err = max(max_nav_err, nav_err)
        if log and k % 10 == 0:
            hist.append((t, *s, nav.x[0], nav.x[1], theta_cmd, delta, qbar, nav_err))

        if (t > 5.0 and s[1] <= 0.0) or t > 900.0:
            break

    veh.thrust_scale = 1.0
    result = dict(
        impact_x=s[0], apogee=apogee, max_qbar=max_qbar, max_nav_err=max_nav_err,
        flight_time=t,
        gps_isolated_time=nav.gps_isolated_time,
        engine_fault_detected_time=engine_fault_time_detected,
    )
    if log:
        result["hist"] = np.array(hist)
    return result


# =============================================================================
# MONTE CARLO DISPERSION ANALYSIS
# =============================================================================
def monte_carlo(n_runs=200, seed=42):
    rng = np.random.default_rng(seed)
    base = Vehicle()
    rows = []
    for i in range(n_runs):
        veh = replace(
            base,
            isp=base.isp * rng.normal(1.0, 0.01),
            m_dry=base.m_dry * rng.normal(1.0, 0.01),
            cd=base.cd * rng.normal(1.0, 0.05),
            x_cp=base.x_cp + rng.normal(0.0, 0.05),
        )
        u = rng.random()
        cfg = FlightConfig(wind_ref=rng.normal(0.0, 8.0), seed=int(rng.integers(1e9)))
        if u < 0.10:
            cfg.gps_fault_time = rng.uniform(20.0, 80.0)
        elif u < 0.20:
            cfg.engine_fault_time = rng.uniform(10.0, 45.0)
        r = fly(veh, cfg)
        r["gps_fault_time"] = cfg.gps_fault_time
        r["engine_fault_time"] = cfg.engine_fault_time
        rows.append(r)
    return rows


def summarise(rows):
    def stats(vals):
        a = np.array(vals)
        return a.mean(), a.std(ddof=1)

    nominal = [r for r in rows if r["gps_fault_time"] is None and r["engine_fault_time"] is None]
    gps_f = [r for r in rows if r["gps_fault_time"] is not None]
    eng_f = [r for r in rows if r["engine_fault_time"] is not None]

    print(f"\nMonte Carlo: {len(rows)} runs "
          f"({len(nominal)} healthy, {len(gps_f)} GPS fault, {len(eng_f)} engine fault)")
    m, sd = stats([r["impact_x"] / 1000 for r in nominal])
    print(f"  Healthy impact range : {m:8.2f} km  +/- {sd:.2f} km (1 sigma)")
    m, sd = stats([r["apogee"] / 1000 for r in nominal])
    print(f"  Healthy apogee       : {m:8.2f} km  +/- {sd:.2f} km")
    m, sd = stats([r["max_qbar"] / 1000 for r in nominal])
    print(f"  Healthy max-q        : {m:8.2f} kPa +/- {sd:.2f} kPa")
    m, sd = stats([r["max_nav_err"] for r in nominal])
    print(f"  Healthy max nav error: {m:8.2f} m   +/- {sd:.2f} m (first 120 s)")

    false_gps = sum(r["gps_isolated_time"] is not None for r in nominal)
    false_eng = sum(r["engine_fault_detected_time"] is not None for r in nominal)
    print(f"  False alarms (healthy runs): GPS {false_gps}, engine {false_eng}")

    if gps_f:
        det = [r for r in gps_f if r["gps_isolated_time"] is not None]
        lat = [r["gps_isolated_time"] - r["gps_fault_time"] for r in det]
        print(f"  GPS faults isolated  : {len(det)}/{len(gps_f)}"
              + (f", mean latency {np.mean(lat):.2f} s" if lat else ""))
        m, sd = stats([r["max_nav_err"] for r in gps_f]) if len(gps_f) > 1 else (gps_f[0]["max_nav_err"], 0)
        print(f"  Max nav error with GPS fault: {m:.1f} m +/- {sd:.1f} m")
    if eng_f:
        det = [r for r in eng_f if r["engine_fault_detected_time"] is not None]
        lat = [r["engine_fault_detected_time"] - r["engine_fault_time"] for r in det]
        print(f"  Engine faults detected: {len(det)}/{len(eng_f)}"
              + (f", mean latency {np.mean(lat):.2f} s" if lat else ""))
    return nominal, gps_f, eng_f


def make_plots(nominal_hist, nominal, gps_f, eng_f, path="launch_sim_results.png"):
    h = nominal_hist
    t, x, z = h[:, 0], h[:, 1], h[:, 2]
    theta, theta_cmd, delta = h[:, 5], h[:, 10], h[:, 11]
    qbar, nav_err = h[:, 12], h[:, 13]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ax[0, 0].plot(x / 1000, z / 1000)
    ax[0, 0].set(title="Nominal trajectory", xlabel="Downrange (km)", ylabel="Altitude (km)")
    burn = t < 75
    ax[0, 1].plot(t[burn], np.degrees(theta[burn]), label="pitch")
    ax[0, 1].plot(t[burn], np.degrees(theta_cmd[burn]), "--", label="command")
    ax[0, 1].set(title="Guidance tracking (burn)", xlabel="Time (s)", ylabel="deg")
    ax[0, 1].legend()
    ax[0, 2].plot(t[burn], np.degrees(delta[burn]))
    ax[0, 2].set(title="Gimbal angle", xlabel="Time (s)", ylabel="deg")
    ax[1, 0].plot(t, qbar / 1000)
    ax[1, 0].set(title="Dynamic pressure", xlabel="Time (s)", ylabel="kPa")
    ax[1, 1].plot(t, nav_err)
    ax[1, 1].set(title="Kalman filter position error", xlabel="Time (s)", ylabel="m")
    ax[1, 2].hist([r["impact_x"] / 1000 for r in nominal], bins=20, alpha=0.7, label="healthy")
    if gps_f:
        ax[1, 2].hist([r["impact_x"] / 1000 for r in gps_f], bins=10, alpha=0.7, label="GPS fault")
    if eng_f:
        ax[1, 2].hist([r["impact_x"] / 1000 for r in eng_f], bins=10, alpha=0.7, label="engine fault")
    ax[1, 2].set(title="Monte Carlo impact footprint", xlabel="Impact downrange (km)", ylabel="Runs")
    ax[1, 2].legend()
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nPlots saved to {path}")


# =============================================================================
# WHERE QUANTUM / QUANTUM-INSPIRED METHODS WOULD PLUG IN (not implemented)
# =============================================================================
#  guidance_pitch()        -> replace the fixed program with an optimiser
#                             (classical first: direct collocation / convex
#                             guidance; then test QAOA/VQE on a discretised
#                             sub-problem and compare against the classical one).
#  NavFilter               -> a quantum-inspired estimator would replace or
#                             augment predict()/update().
#  FDIR logic in fly()     -> a classifier (classical ML first, then a
#                             variational quantum circuit) trained on Monte
#                             Carlo telemetry produced by monte_carlo().
#  monte_carlo()           -> the data generator for training and for
#                             validating any of the above.
# A fair test always compares the quantum method against the best classical
# method on the same data; that comparison is what "demonstrable advantage" means.


if __name__ == "__main__":
    print("Nominal flight (no wind, no faults)...")
    nom = fly(Vehicle(), FlightConfig(seed=1), log=True)
    print(f"  apogee {nom['apogee']/1000:.2f} km, impact {nom['impact_x']/1000:.2f} km, "
          f"max-q {nom['max_qbar']/1000:.1f} kPa, flight time {nom['flight_time']:.0f} s")

    print("\nGPS fault demo (300 m bias at t = 40 s)...")
    r = fly(Vehicle(), FlightConfig(seed=2, gps_fault_time=40.0))
    print(f"  GPS isolated at t = {r['gps_isolated_time']}")

    print("\nEngine fault demo (thrust drops to 60% at t = 20 s)...")
    r = fly(Vehicle(), FlightConfig(seed=3, engine_fault_time=20.0))
    print(f"  Engine fault detected at t = {r['engine_fault_detected_time']}, "
          f"apogee {r['apogee']/1000:.2f} km")

    rows = monte_carlo(n_runs=200)
    nominal, gps_f, eng_f = summarise(rows)
    make_plots(nom["hist"], nominal, gps_f, eng_f)
