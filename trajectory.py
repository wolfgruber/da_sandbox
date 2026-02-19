#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 21:58:14 2026

@author: ludo
"""


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numba import njit

#%%

def integrate_flight(
    y0, dt, t_final, beta,
    mu=3.98184378210000e14,   # Earth's gravitational parameter (m^3/s^2)
    Omega=7.2921159e-5,
    rho0=0.125,
    H=8500.0,
    R_earth=6371000.0,
    engine_acceleration=None,
):

    if engine_acceleration is None:
        def engine_acceleration(t, y, v_hat, n_hat, b_hat):
            return 0.0, 0.0, 0.0
        
    def rhs(t, y):
        r, th, lam, rd, thd, lamd = y

        sin_th = np.sin(th)
        cos_th = np.cos(th)
        rs = r * sin_th

        # Velocity components
        vr = rd
        vth = r * thd
        vlam = rs * lamd

        v2 = vr*vr + vth*vth + vlam*vlam
        v = np.sqrt(v2) + 1e-12

        # Unit velocity vector
        v_hat_r = vr / v
        v_hat_th = vth / v
        v_hat_lam = vlam / v

        # Normal (pitch) direction
        n_hat_r = -v_hat_th
        n_hat_th = v_hat_r
        n_hat_lam = 0.0

        # Binormal (roll) direction
        b_hat_r = -v_hat_lam
        b_hat_th = 0.0
        b_hat_lam = v_hat_r

        # Atmosphere
        rho = rho0 * np.exp(-(r - R_earth) / H)
        drag = rho * v / (2.0 * beta)

        # Drag
        adr = -drag * vr
        adt = -drag * vth
        adl = -drag * vlam

        # Coriolis
        acr = 2 * Omega * r * sin_th*sin_th * lamd
        act = -2 * Omega * r * sin_th*cos_th * lamd
        acl = -2 * Omega * (rd*sin_th + r*thd*cos_th)

        # Engine acceleration (steered)
        aer, aet, ael = engine_acceleration(
            t, y,
            (v_hat_r, v_hat_th, v_hat_lam),
            (n_hat_r, n_hat_th, n_hat_lam),
            (b_hat_r, b_hat_th, b_hat_lam),
        )

        # Newtonian gravity (1/r^2 radial law)
        grav_r = -mu / (r*r)
        
        rdd = (
            r*thd*thd
            + r*lamd*lamd*sin_th*sin_th
            + grav_r
            + adr + acr + aer
        )

        thdd = (
            r*lamd*lamd*sin_th*cos_th
            - 2*rd*thd
            + adt + act + aet
        ) / r

        lamdd = (
            -2*rd*sin_th*lamd
            - 2*r*thd*cos_th*lamd
            + adl + acl + ael
        ) / (rs + 1e-12)

        return np.array([rd, thd, lamd, rdd, thdd, lamdd])

    # Time array
    N = int(np.ceil(t_final / dt)) + 1
    t = np.linspace(0, t_final, N)

    Y = np.zeros((N, 6))
    Y[0] = y0

    # RK4 integration
    for i in range(N - 1):
        ti = t[i]
        yi = Y[i]

        k1 = rhs(ti, yi)
        k2 = rhs(ti + dt/2, yi + dt/2 * k1)
        k3 = rhs(ti + dt/2, yi + dt/2 * k2)
        k4 = rhs(ti + dt, yi + dt * k3)

        Y[i+1] = yi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        if Y[i+1,0] < R_earth:
            Y[i+1,0] = R_earth
            Y[i+1,3] = 0
            Y[i+1,4] = 0
            Y[i+1,5] = 0
            #g = 0
            

    return t, Y


def initial_conditions_from_launch(
    lat_deg,
    lon_deg,
    h0,
    speed,
    azimuth_deg,
    flight_path_deg,
    R_earth=6371000.0,
):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    az = np.radians(azimuth_deg)
    gamma = np.radians(flight_path_deg)

    r0 = R_earth + h0
    theta0 = np.pi/2 - lat
    lambda0 = lon

    vU = speed * np.sin(gamma)
    vH = speed * np.cos(gamma)

    vN = vH * np.cos(az)
    vE = vH * np.sin(az)

    r_dot0 = vU
    theta_dot0 = -vN / r0
    lambda_dot0 = vE / (r0 * np.sin(theta0))

    return np.array([r0, theta0, lambda0,
                     r_dot0, theta_dot0, lambda_dot0])
#%%

def plot_trajectory_map(
    Y,
    R_earth=6371000.0,
    title="Flight Trajectory",
):
    """
    Plot trajectory ground track on Cartopy map and
    altitude over curved Earth distance.

    Parameters
    ----------
    Y : ndarray (N x 6)
        State history from integrate_flight
    R_earth : float
        Earth radius
    """

    r = Y[:, 0]
    theta = Y[:, 1]
    lam = Y[:, 2]

    # Convert to geodetic-like coordinates
    lat = np.pi/2 - theta
    lon = lam

    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    # ---- Create figure ----
    fig = plt.figure(figsize=(10, 10))

    # =========================
    # Top panel: Map
    # =========================
    ax_map = plt.subplot(2, 1, 1,
                         projection=ccrs.PlateCarree())

    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.add_feature(cfeature.LAND, alpha=0.3)
    ax_map.add_feature(cfeature.OCEAN, alpha=0.2)

    ax_map.plot(lon_deg, lat_deg,
        color='red', linewidth=2,
        transform=ccrs.Geodetic(), label="Trajectory")

    ax_map.scatter(lon_deg[0], lat_deg[0],
        color='green', s=60,
        transform=ccrs.PlateCarree(), label="Start")

    ax_map.scatter(lon_deg[-1], lat_deg[-1],
        color='blue', s=60,
        transform=ccrs.PlateCarree(), label="End")

    ax_map.set_title(title)
    ax_map.legend()

    # =========================
    # Bottom panel: True 3D side projection (North up)
    # =========================
    ax_side = plt.subplot(2, 1, 2)

    # --- Convert spherical to Cartesian ---
    x = r * np.sin(theta) * np.cos(lam)
    y_cart = r * np.sin(theta) * np.sin(lam)
    z = r * np.cos(theta)

    traj = np.vstack((x, y_cart, z)).T

    # --- Best-fit plane (SVD) ---
    centroid = traj.mean(axis=0)
    traj_centered = traj - centroid

    U, S, Vt = np.linalg.svd(traj_centered)
    e1 = Vt[0]
    e2 = Vt[1]
    normal = Vt[2]

    # --- Project trajectory into plane ---
    x_plane = traj_centered @ e1
    z_plane = traj_centered @ e2

    # --- Earth cross section in same plane ---
    angle = np.linspace(0, 2*np.pi, 400)
    circle_points = (
        np.outer(np.cos(angle), e1) +
        np.outer(np.sin(angle), e2)
    ) * R_earth

    circle_centered = circle_points - centroid
    x_earth = circle_centered @ e1
    z_earth = circle_centered @ e2

    # ==================================================
    # Rotate so that global North (z-axis) is UP
    # ==================================================

    north_vec = np.array([0.0, 0.0, 1.0])

    # Project north into the flight plane
    north_proj = north_vec - np.dot(north_vec, normal) * normal
    north_proj /= np.linalg.norm(north_proj)

    # Coordinates of north direction in plane basis
    north_x = np.dot(north_proj, e1)
    north_z = np.dot(north_proj, e2)

    # Angle to rotate so north aligns with +z axis
    angle_rot = np.arctan2(north_x, north_z)

    c = np.cos(angle_rot)
    s = np.sin(angle_rot)

    def rotate(X, Z):
        Xr =  c*X - s*Z
        Zr =  s*X + c*Z
        return Xr, Zr

    x_plane, z_plane = rotate(x_plane, z_plane)
    x_earth, z_earth = rotate(x_earth, z_earth)
    
    # ==================================================
    # Equator (visible half only)
    # ==================================================
    lon_eq = np.linspace(-np.pi, np.pi, 800)
    x_eq = R_earth * np.cos(lon_eq)
    y_eq = R_earth * np.sin(lon_eq)
    z_eq = np.zeros_like(lon_eq)

    eq_points = np.vstack((x_eq, y_eq, z_eq)).T
    eq_centered = eq_points - centroid

    # Visibility test
    visible = (eq_centered @ normal) >= 0

    x_eq_plane = eq_centered @ e1
    z_eq_plane = eq_centered @ e2
    x_eq_plane, z_eq_plane = rotate(x_eq_plane, z_eq_plane)

    ax_side.plot(
        x_eq_plane[visible]/1000,
        z_eq_plane[visible]/1000,
        linestyle='--', color="black"
    )


    # ==================================================
    # Meridians (visible half only)
    # ==================================================
    meridians_deg = np.arange(-120, 181, 30)
    lat_vals = np.linspace(-np.pi/2, np.pi/2, 600)

    for lon_deg in meridians_deg:
        lon = np.radians(lon_deg)

        x_m = R_earth * np.cos(lat_vals) * np.cos(lon)
        y_m = R_earth * np.cos(lat_vals) * np.sin(lon)
        z_m = R_earth * np.sin(lat_vals)

        m_points = np.vstack((x_m, y_m, z_m)).T
        m_centered = m_points - centroid

        visible = (m_centered @ normal) >= 0

        x_m_plane = m_centered @ e1
        z_m_plane = m_centered @ e2
        x_m_plane, z_m_plane = rotate(x_m_plane, z_m_plane)

        ax_side.plot(
            x_m_plane[visible]/1000,
            z_m_plane[visible]/1000,
            linewidth=0.7, color="black"
        )


    # --- Plot ---
    ax_side.plot(x_earth/1000, z_earth/1000, color="black")
    ax_side.plot(x_plane/1000, z_plane/1000, color="red", label="Trajectory")
    ax_side.scatter(x_plane[0]/1000, z_plane[0]/1000,
        color='green', s=60, label="Start")
    ax_side.scatter(x_plane[-1]/1000, z_plane[-1]/1000,
        color='blue', s=60, label="End")

    ax_side.set_aspect('equal', adjustable='box')
    ax_side.set_xlabel("In-plane distance (km)")
    ax_side.set_ylabel("North (km)")
    ax_side.legend()
    ax_side.set_title("True 3D Side Projection (North Up)")
    ax_side.grid(False)

    plt.tight_layout()
    plt.show()
#%%
y0 = initial_conditions_from_launch(
    lat_deg=45, lon_deg=0, h0=600000,
    speed=8000, azimuth_deg=150, flight_path_deg=20)

t, Y = integrate_flight(
    y0,
    dt=5,
    t_final=3600 * 10,
    beta=500
)

plot_trajectory_map(Y)
