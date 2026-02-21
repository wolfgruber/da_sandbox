#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A model for a propelled object in Earth's atmosphere.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dataclasses import dataclass
from functools import partial
from typing import Callable

EPS = 1e-12


def normalize3(x, y, z, fallback=(1.0, 0.0, 0.0)):
    n = np.sqrt(x * x + y * y + z * z)
    if n < EPS:
        return fallback
    return x / n, y / n, z / n


def cross3(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def velocity_frame(vr, vth, vlam):
    v_hat_r, v_hat_th, v_hat_lam = normalize3(vr, vth, vlam)

    # "Up" in local spherical basis is radial.
    up_r, up_th, up_lam = 1.0, 0.0, 0.0
    b_r, b_th, b_lam = cross3(up_r, up_th, up_lam, v_hat_r, v_hat_th, v_hat_lam)
    b_hat_r, b_hat_th, b_hat_lam = normalize3(b_r, b_th, b_lam, fallback=(0.0, 0.0, 1.0))

    n_r, n_th, n_lam = cross3(v_hat_r, v_hat_th, v_hat_lam, b_hat_r, b_hat_th, b_hat_lam)
    n_hat_r, n_hat_th, n_hat_lam = normalize3(n_r, n_th, n_lam, fallback=(0.0, 1.0, 0.0))

    return (
        (v_hat_r, v_hat_th, v_hat_lam),
        (n_hat_r, n_hat_th, n_hat_lam),
        (b_hat_r, b_hat_th, b_hat_lam),
    )


def zero_engine_acceleration(t, y, v_hat, n_hat, b_hat):
    return 0.0, 0.0, 0.0


def _as_schedule(value_or_callable):
    if callable(value_or_callable):
        return value_or_callable
    return lambda t, y: float(value_or_callable)


@dataclass
class SteeredEngineAcceleration:
    accel_fn: Callable
    yaw_fn: Callable
    pitch_fn: Callable
    angle_scale: float

    def __call__(self, t, y, v_hat, n_hat, b_hat):
        a = self.accel_fn(t, y)
        yaw_rad = self.yaw_fn(t, y) * self.angle_scale
        pitch_rad = self.pitch_fn(t, y) * self.angle_scale

        c_pitch = np.cos(pitch_rad)
        v_gain = c_pitch * np.cos(yaw_rad)
        n_gain = np.sin(pitch_rad)
        b_gain = c_pitch * np.sin(yaw_rad)

        ar = a * (v_gain * v_hat[0] + n_gain * n_hat[0] + b_gain * b_hat[0])
        ath = a * (v_gain * v_hat[1] + n_gain * n_hat[1] + b_gain * b_hat[1])
        alam = a * (v_gain * v_hat[2] + n_gain * n_hat[2] + b_gain * b_hat[2])
        return ar, ath, alam


def steered_engine_acceleration(accel, yaw=0.0, pitch=0.0, angles_in_degrees=True):
    """
    Return engine callback with steering in local velocity frame.

    `accel`, `yaw`, `pitch` can be constants or callables `(t, y) -> float`.
    Direction is:
    - along velocity (`v_hat`) when yaw=pitch=0
    - pitch tilts toward `n_hat`
    - yaw tilts toward `b_hat`
    """
    accel_fn = _as_schedule(accel)
    yaw_fn = _as_schedule(yaw)
    pitch_fn = _as_schedule(pitch)
    angle_scale = np.pi / 180.0 if angles_in_degrees else 1.0

    return SteeredEngineAcceleration(accel_fn, yaw_fn, pitch_fn, angle_scale)


def flight_rhs(
    t,
    y,
    beta,
    mu,
    omega,
    rho0,
    scale_height,
    r_earth,
    engine_acceleration,
):
    r, th, lam, rd, thd, lamd = y

    sin_th = np.sin(th)
    cos_th = np.cos(th)
    rs = r * sin_th

    vr = rd
    vth = r * thd
    vlam = rs * lamd

    v2 = vr * vr + vth * vth + vlam * vlam
    v = np.sqrt(v2) + EPS

    v_hat, n_hat, b_hat = velocity_frame(vr, vth, vlam)

    rho = rho0 * np.exp(-(r - r_earth) / scale_height)
    drag = rho * v / (2.0 * beta)

    adr = -drag * vr
    adt = -drag * vth
    adl = -drag * vlam

    # Coriolis in local spherical basis, Earth rotates about +z axis.
    acr = 2.0 * omega * r * sin_th * sin_th * lamd
    act = -2.0 * omega * r * sin_th * cos_th * lamd
    acl = -2.0 * omega * (rd * sin_th + r * thd * cos_th)

    # Centrifugal acceleration for consistency in Earth-rotating frame.
    acen_r = omega * omega * r * sin_th * sin_th
    acen_t = omega * omega * r * sin_th * cos_th

    aer, aet, ael = engine_acceleration(t, y, v_hat, n_hat, b_hat)
    grav_r = -mu / (r * r)

    rdd = (
        r * thd * thd
        + r * lamd * lamd * sin_th * sin_th
        + grav_r
        + adr
        + acr
        + acen_r
        + aer
    )
    thdd = (r * lamd * lamd * sin_th * cos_th - 2.0 * rd * thd + adt + act + acen_t + aet) / r
    lamdd = (-2.0 * rd * sin_th * lamd - 2.0 * r * thd * cos_th * lamd + adl + acl + ael) / (rs + EPS)

    return np.array([rd, thd, lamd, rdd, thdd, lamdd], dtype=float)


def rk4_step(rhs, t, y, dt):
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_flight(
    y0,
    dt,
    t_final,
    beta,
    mu=3.986004418e14,
    omega=7.2921159e-5,
    rho0=0.125,
    scale_height=8500.0,
    r_earth=6371000.0,
    engine_acceleration=None,
):
    if engine_acceleration is None:
        engine_acceleration = zero_engine_acceleration

    n_steps = int(np.ceil(t_final / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    t[-1] = t_final

    y_hist = np.zeros((n_steps, 6), dtype=float)
    y_hist[0] = np.asarray(y0, dtype=float)

    rhs_local = partial(
        flight_rhs,
        beta=beta,
        mu=mu,
        omega=omega,
        rho0=rho0,
        scale_height=scale_height,
        r_earth=r_earth,
        engine_acceleration=engine_acceleration,
    )

    for i in range(n_steps - 1):
        dt_i = t[i + 1] - t[i]
        y_next = rk4_step(rhs_local, t[i], y_hist[i], dt_i)

        if y_next[0] < r_earth:
            y_next[0] = r_earth
            y_next[3:] = 0.0
            y_hist[i + 1] = y_next
            if i + 2 < n_steps:
                y_hist[i + 2 :] = y_next
            break

        y_hist[i + 1] = y_next

    return t, y_hist


def initial_conditions_from_launch(
    lat_deg,
    lon_deg,
    h0,
    speed,
    azimuth_deg,
    flight_path_deg,
    r_earth=6371000.0,
):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    az = np.radians(azimuth_deg)
    gamma = np.radians(flight_path_deg)

    r0 = r_earth + h0
    theta0 = np.pi / 2.0 - lat
    lambda0 = lon

    v_up = speed * np.sin(gamma)
    v_h = speed * np.cos(gamma)
    v_n = v_h * np.cos(az)
    v_e = v_h * np.sin(az)

    r_dot0 = v_up
    theta_dot0 = -v_n / r0
    lambda_dot0 = v_e / (r0 * max(np.sin(theta0), EPS))

    return np.array([r0, theta0, lambda0, r_dot0, theta_dot0, lambda_dot0], dtype=float)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def interpolate_angle_shortest(a0, a1, alpha):
    return a0 + alpha * wrap_to_pi(a1 - a0)


def touchdown_by_future_time(t, y_hist, t_future, r_earth=6371000.0, tol=1e-6):
    """
    Check whether touchdown occurs by `t_future`.

    Returns:
      - None, if no touchdown by `t_future`
      - dict with touchdown time and coordinates, if touchdown occurs by `t_future`
    """
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y_hist, dtype=float)
    if t_arr.ndim != 1 or y_arr.ndim != 2 or y_arr.shape[1] != 6 or y_arr.shape[0] != t_arr.size:
        raise ValueError("Expected t shape (N,) and y_hist shape (N, 6) with matching N.")

    if t_future < t_arr[0]:
        return None

    end_idx = int(np.searchsorted(t_arr, t_future, side="right") - 1)
    end_idx = max(0, min(end_idx, t_arr.size - 1))

    r = y_arr[:, 0]
    hit_candidates = np.where(r[: end_idx + 1] <= r_earth + tol)[0]
    if hit_candidates.size == 0:
        return None

    i = int(hit_candidates[0])
    if i == 0:
        t_touch = t_arr[0]
        theta_touch = y_arr[0, 1]
        lam_touch = y_arr[0, 2]
        state_touch = y_arr[0].copy()
    elif r[i - 1] > r_earth + tol:
        r0, r1 = r[i - 1], r[i]
        alpha = np.clip((r_earth - r0) / ((r1 - r0) + EPS), 0.0, 1.0)
        t_touch = t_arr[i - 1] + alpha * (t_arr[i] - t_arr[i - 1])
        theta_touch = y_arr[i - 1, 1] + alpha * (y_arr[i, 1] - y_arr[i - 1, 1])
        lam_touch = interpolate_angle_shortest(y_arr[i - 1, 2], y_arr[i, 2], alpha)
        state_touch = y_arr[i - 1] + alpha * (y_arr[i] - y_arr[i - 1])
        state_touch[0] = r_earth
        state_touch[1] = theta_touch
        state_touch[2] = lam_touch
    else:
        t_touch = t_arr[i]
        theta_touch = y_arr[i, 1]
        lam_touch = y_arr[i, 2]
        state_touch = y_arr[i].copy()

    if t_touch > t_future + tol:
        return None

    lat_touch = np.pi / 2.0 - theta_touch
    lon_touch = wrap_to_pi(lam_touch)
    return {
        "t_touch_s": float(t_touch),
        "theta_touch_rad": float(theta_touch),
        "lambda_touch_rad": float(lon_touch),
        "lat_touch_deg": float(np.degrees(lat_touch)),
        "lon_touch_deg": float(np.degrees(lon_touch)),
        "state_touch": state_touch,
        "index_after_touchdown": i,
    }


def rotate_2d(x, z, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    xr = c * x - s * z
    zr = s * x + c * z
    return xr, zr


def spherical_to_cartesian(r, theta, lam):
    x = r * np.sin(theta) * np.cos(lam)
    y = r * np.sin(theta) * np.sin(lam)
    z = r * np.cos(theta)
    return x, y, z


def build_projection_frame(traj_xyz):
    centroid = traj_xyz.mean(axis=0)
    centered = traj_xyz - centroid
    _, _, vt = np.linalg.svd(centered)
    e1, e2, normal = vt[0], vt[1], vt[2]

    north = np.array([0.0, 0.0, 1.0], dtype=float)
    north_proj = north - np.dot(north, normal) * normal
    north_norm = np.linalg.norm(north_proj)
    if north_norm < EPS:
        # Plane is near-horizontal; orientation is underdetermined.
        rot = 0.0
    else:
        north_proj /= north_norm
        north_x = np.dot(north_proj, e1)
        north_z = np.dot(north_proj, e2)
        rot = np.arctan2(north_x, north_z)
    return centroid, e1, e2, normal, rot


def project_points(points_xyz, centroid, e1, e2, rot):
    centered = points_xyz - centroid
    x = centered @ e1
    z = centered @ e2
    return rotate_2d(x, z, rot)


def check_coordinate_consistency():
    """
    Numerical sanity checks for local frame and engine steering transform.
    Returns max errors; values near 0 indicate consistency.
    """
    velocity_samples = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 2.0, 3.0),
        (-4.0, 1.0, 0.5),
    ]
    angle_samples = [(-20.0, -10.0), (0.0, 0.0), (15.0, 25.0), (60.0, -30.0)]

    max_norm_err = 0.0
    max_ortho_err = 0.0
    max_handed_err = 0.0
    max_engine_mag_err = 0.0

    for vr, vth, vlam in velocity_samples:
        v_hat, n_hat, b_hat = velocity_frame(vr, vth, vlam)
        max_norm_err = max(
            max_norm_err,
            abs(np.linalg.norm(v_hat) - 1.0),
            abs(np.linalg.norm(n_hat) - 1.0),
            abs(np.linalg.norm(b_hat) - 1.0),
        )
        max_ortho_err = max(
            max_ortho_err,
            abs(np.dot(v_hat, n_hat)),
            abs(np.dot(v_hat, b_hat)),
            abs(np.dot(n_hat, b_hat)),
        )
        handed = np.cross(np.array(v_hat), np.array(n_hat))
        max_handed_err = max(max_handed_err, np.linalg.norm(handed - np.array(b_hat)))

        for yaw_deg, pitch_deg in angle_samples:
            eng = steered_engine_acceleration(accel=1.0, yaw=yaw_deg, pitch=pitch_deg)
            ar, ath, alam = eng(0.0, np.zeros(6), v_hat, n_hat, b_hat)
            max_engine_mag_err = max(
                max_engine_mag_err,
                abs(np.sqrt(ar * ar + ath * ath + alam * alam) - 1.0),
            )

    return {
        "max_norm_error": max_norm_err,
        "max_orthogonality_error": max_ortho_err,
        "max_right_handed_error": max_handed_err,
        "max_engine_magnitude_error": max_engine_mag_err,
    }


def plot_trajectory_map(
    t,
    y_hist,
    r_earth=6371000.0,
    title="Flight Trajectory",
):
    r = y_hist[:, 0]
    theta = y_hist[:, 1]
    lam = y_hist[:, 2]

    lat = np.pi / 2.0 - theta
    lon = lam
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.LAND, alpha=0.3)
    ax_map.add_feature(cfeature.OCEAN, alpha=0.2)
    ax_map.plot(lon_deg, lat_deg, color="red", linewidth=2, transform=ccrs.Geodetic(), label="Trajectory")
    ax_map.scatter(lon_deg[0], lat_deg[0], color="green", s=60, transform=ccrs.PlateCarree(), label="Start")
    ax_map.scatter(lon_deg[-1], lat_deg[-1], color="blue", s=60, transform=ccrs.PlateCarree(), label="End")
    ax_map.set_title(title)
    ax_map.legend()

    ax_side = fig.add_subplot(gs[1, 0])

    x, y_cart, z = spherical_to_cartesian(r, theta, lam)
    traj = np.column_stack((x, y_cart, z))
    centroid, e1, e2, normal, rot = build_projection_frame(traj)

    x_plane, z_plane = project_points(traj, centroid, e1, e2, rot)

    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    circle = (np.outer(np.cos(angle), e1) + np.outer(np.sin(angle), e2)) * r_earth
    x_earth, z_earth = project_points(circle, centroid, e1, e2, rot)

    lon_eq = np.linspace(-np.pi, np.pi, 800)
    eq_points = np.column_stack(
        (
            r_earth * np.cos(lon_eq),
            r_earth * np.sin(lon_eq),
            np.zeros_like(lon_eq),
        )
    )
    eq_centered = eq_points - centroid
    eq_visible = (eq_centered @ normal) >= 0.0
    x_eq, z_eq = project_points(eq_points, centroid, e1, e2, rot)
    ax_side.plot(x_eq[eq_visible] / 1000.0, z_eq[eq_visible] / 1000.0, linestyle="--", color="black")

    meridians_deg = np.arange(-120, 181, 30)
    lat_vals = np.linspace(-np.pi / 2.0, np.pi / 2.0, 600)
    cos_lat = np.cos(lat_vals)
    sin_lat = np.sin(lat_vals)
    for lon_m_deg in meridians_deg:
        lon_m = np.radians(lon_m_deg)
        m_points = np.column_stack(
            (
                r_earth * cos_lat * np.cos(lon_m),
                r_earth * cos_lat * np.sin(lon_m),
                r_earth * sin_lat,
            )
        )
        m_centered = m_points - centroid
        visible = (m_centered @ normal) >= 0.0
        x_m, z_m = project_points(m_points, centroid, e1, e2, rot)
        ax_side.plot(x_m[visible] / 1000.0, z_m[visible] / 1000.0, linewidth=0.7, color="black")

    ax_side.plot(x_earth / 1000.0, z_earth / 1000.0, color="black")
    ax_side.plot(x_plane / 1000.0, z_plane / 1000.0, color="red", label="Trajectory")
    ax_side.scatter(x_plane[0] / 1000.0, z_plane[0] / 1000.0, color="green", s=60, label="Start")
    ax_side.scatter(x_plane[-1] / 1000.0, z_plane[-1] / 1000.0, color="blue", s=60, label="End")
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("In-plane distance (km)")
    ax_side.set_ylabel("North (km)")
    ax_side.legend()
    ax_side.set_title("True 3D Side Projection (North Up)")
    ax_side.grid(False)

    # Rates panel: dr/dt, dtheta/dt, dlambda/dt vs time.
    ax_rates = fig.add_subplot(gs[1, 1])
    t_hours = np.asarray(t) / 3600.0
    #ax_rates.plot(t_hours, y_hist[:, 0], label=r"$\dot{r}$ (m/s)")
    ax_rates.plot(t_hours, y_hist[:, 1], label=r"$\dot{\theta}$ (rad/s)")
    ax_rates.plot(t_hours, y_hist[:, 2], label=r"$\dot{\lambda}$ (rad/s)")
    ax_rates.set_xlabel("Time (h)")
    ax_rates.set_ylabel("Rate")
    ax_rates.set_title("State Rates vs Time")
    ax_rates.grid(True, alpha=0.3)
    ax_rates.legend()

    plt.tight_layout()
    plt.show()

# %%
def main():
    y0 = initial_conditions_from_launch(
        lat_deg=10,
        lon_deg=10,
        h0= 600 * 1000,
        speed=6000,
        azimuth_deg=15,
        flight_path_deg=45,
    )

    engine = steered_engine_acceleration(
        accel=0.0,    # m/s^2 engine magnitude
        yaw=0.0,      # deg
        pitch=0.0,    # deg
        angles_in_degrees=True,
    )

    t, y_hist = integrate_flight(
        y0=y0,
        dt=1,
        t_final=3600 * 1.0,
        beta=500.0,
        engine_acceleration=engine,
    )

    plot_trajectory_map(t, y_hist)


if __name__ == "__main__":
    main()

# %%
