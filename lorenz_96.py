# %%
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# %%
def EnKF(x_ens, obs, B, H, sigma_obs=0.01):
    background = x_ens[:,-1,:]
    n_obs = len(obs)
    R = np.eye(n_obs) * sigma_obs**2
    
    # compute the departure or innovation d
    # d = y_o - H x_f with added observation noise
    obs_noise = np.random.normal(0, sigma_obs, size=(n_ens, n_obs))
    H = H
    Hx = H @ background.T
    departure = obs + obs_noise - Hx.T
    #print("Background: {}".format(background))
    #print("Departure: {}".format(departure))
    
    # compute the kalman gain K
    # K = B H^T (H B H^T + R)^-1
    K = B @ H.T @ np.linalg.pinv(H @ B @ H.T + R)
    
    # compute the formula
    # x_a = x_f + K (y_o - H x_f)
    # with K = B H^T (H B H^T + R)^-1
    analysis = background + (K @ departure.T).T
    #print("Increment: {}".format(analysis-background))
    #print("Analysis: {}".format(analysis))

    return analysis

# %%
# These are our constants
n_x = 50  # Number of variables
F = 8  # Forcing
n_ens = 20
dt = 0.01
dt_fraction = 10
dt_coarse = dt * dt_fraction

rng = np.random.default_rng(0)


def L96(x, t):
    """Lorenz 96 model with constant forcing"""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 


def init(n_x, F):
    x_t_0 = rng.normal(0, F*0.6, size=n_x) # Initial state
    x_ens_0 = x_t_0[np.newaxis,:] + rng.normal(0, F*0.06, size=(n_ens, n_x)) # Initial state

    t_true = np.arange(0.0, 1+dt/2, dt)
    t_coarse = np.arange(0.0, 1+dt_coarse/2, dt_coarse)

    x_t = odeint(L96, x_t_0, t_true)
    #x_t = np.concatenate([np.reshape(x_t_0, (1,n_x)), x_t], axis=0)

    x_ens = np.empty((n_ens, len(t_coarse), n_x))
    for ens in range(n_ens):
        x_ens[ens,:,:] = odeint(L96, x_ens_0[ens,:], t_coarse)
    #x_ens = np.concatenate([np.reshape(x_ens_0, (n_ens,1,n_x)),
    #                        x_ens], axis=1)

    return t_true, x_t, t_coarse, x_ens


def integrate_by(x_t, x_ens, time):

    t_true = np.arange(0, time+dt/2, dt)
    t_coarse = np.arange(0, time+dt_coarse/2, dt_coarse)

    x_t_new = odeint(L96, x_t[-1,:], t_true)
    x_t = np.concatenate([x_t, x_t_new[1:,:]], axis=0)

    x_ens_new = np.empty((n_ens, len(t_coarse), n_x))
    for ens in range(n_ens):
        x_ens_new[ens,:,:] = odeint(L96, x_ens[ens,-1,:], t_coarse)
    x_ens = np.concatenate([x_ens, x_ens_new[:,1:,:]], axis=1)

    # generate full t
    t_true = np.linspace(0, x_t.shape[0]*dt, x_t.shape[0])
    t_coarse = np.linspace(0, x_ens.shape[1]*dt_coarse, x_ens.shape[1])

    return t_true, x_t, t_coarse, x_ens


def make_observations(x_t, loc, true_obs_err=F*0.1):
     return x_t[-1,loc] + rng.normal(0, true_obs_err, size=len(loc))


def model_equivalent(x_ens, loc):
     # simple observation operator
     return x_t[:,-1,loc]


def copute_ens_covar(x_ens):
        error = x_ens[:,-1,:] - np.mean(x_ens, axis=1)
        B = error.T @ error / (n_ens - 1)
        return B


def plot_spaghetti(t_true, x_t, t_coarse, x_ens, n=1):
    for ens in range(n_ens):
        plt.plot(t_coarse, x_ens[ens,:,n],
                 color="black", linewidth=0.1)
    plt.plot(t_coarse, np.mean(x_ens[:,:,n], axis=0),
             linewidth=1, color="black")
    plt.plot(t_coarse, np.median(x_ens[:,:,n], axis=0),
             linewidth=1, color="tab:blue")
    plt.plot(t_true, x_t[:,n],
             linewidth=1, color="tab:red")
    plt.xlabel("Time")
    plt.ylabel("X{:02d}".format(n))
    plt.show()

# run it
t_true, x_t, t_coarse, x_ens = init(n_x=n_x, F=F)
#t_true, x_t, t_coarse, x_ens = integrate_by(x_t, x_ens, time=1)

for i in range(10):
    obs_loc = np.arange(0, n_x, 10).astype(int)
    obs = make_observations(x_t, obs_loc, true_obs_err=0)
    B = copute_ens_covar(x_ens)
    H = np.zeros((len(obs), n_x))
    for obl in range(len(obs)):
        H[obl,obs_loc[obl]] = 1
    #analysis = EnKF(x_ens=x_ens, obs=obs, B=B, H=H, sigma_obs=0.0001)
    #x_ens[:,-1,:] = analysis

    t_true, x_t, t_coarse, x_ens = integrate_by(x_t, x_ens, time=1)

plot_spaghetti(t_true, x_t, t_coarse, x_ens)

# %%
B = copute_ens_covar(x_ens)
vx = max(-np.min(B), np.max(B))
plt.pcolormesh(B, vmin=-vx, vmax=vx, cmap="coolwarm")
plt.colorbar()
# %%
plt.plot(t2, np.abs(x[::10,1]-x2[:, 1])/F)
# %%
plt.pcolormesh(x)
# %%
