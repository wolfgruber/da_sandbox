#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%%
# General functions
def ddx(f, dx):
    return (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2 * dx)


def ddy(f, dy):
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dy)


def shallow_water(state, dx):
    h, u, v = np.split(state, 3, axis=-1)
    h = h.squeeze()
    u = u.squeeze()
    v = v.squeeze()
    
    g = 9.81
    f = 1e-4
    k = 1e-4 + np.random.rand() * 1e-5
    # compute derivatives
    dhdt = - 1 * (ddx(u, dx) + ddy(v, dx))
    dudt = - g * ddx(h, dx) + f*v - k*u
    dvdt = - g * ddy(h, dx) - f*u - k*v
    return np.stack([dhdt, dudt, dvdt], axis=-1).astype(np.float32)

def RK_integration(state, dt, dx):
    k1 = shallow_water(state, dx)
    k2 = shallow_water(state + 0.5*dt*k1, dx)
    k3 = shallow_water(state + 0.5*dt*k2, dx)
    k4 = shallow_water(state + dt*k3, dx)
    return (state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)).astype(np.float32)

def cycle(model, assimilation, t0, tf, interval, show=False):
    t = t0
    while t < tf:
        model.run_until(t+interval, show)
        assimilation.analysis()
        t += interval

#%%
# Physical model, observation and data assimilation classes

X = 100
Y = 100

class Truth:
    dx = 1.0
    dy = dx
    def __init__(self):
        self.dt = 0.4
        self.nx = int(X // self.dx)
        self.ny = int(Y // self.dy)
        self.x_1d = np.linspace(0, self.nx*self.dx, self.nx)
        self.y_1d = np.linspace(0, self.ny*self.dy, self.ny)
        self.y_2d, self.x_2d = np.meshgrid(self.y_1d, self.x_1d)
        self.h_max = 0.1
        self.u0 = np.sin(2*np.pi*self.x_2d/self.nx) * np.sin(2*np.pi*self.y_2d/self.ny) * self.h_max #np.random.random(size=(self.nx, self.ny)) * 0.01
        self.v0 = np.random.random(size=(self.nx, self.ny)) * self.h_max
        self.h0 = 0.1 * np.sin(2*np.pi*self.x_2d/self.nx) * np.sin(2*np.pi*self.y_2d/self.ny) * self.h_max
        self.state = np.stack([self.h0, self.u0, self.v0], axis=-1).astype(np.float32)
        self.history = self.state.reshape((1, self.nx, self.ny, 3))
        self.time = 0
        self.times = np.array([0])
        return
    
    def step(self):
        self.state = RK_integration(self.state, self.dt, self.dx).astype(np.float32)
        self.history = np.concat([self.history, self.state.reshape((1, self.nx, self.ny, 3))], axis=0)
        self.time = np.round(self.time + self.dt, 2)
        self.times = np.hstack([self.times, self.time])
        return
    
    def run_until(self, t, show=False):
        while not self.time + self.dt/2 > t:
            self.step()
            if self.time % 1 == 0 and show:
                self.plot_state()
        return
    
    def get_value(self, t, x, y, var):
        idx_x = np.argmin(np.abs(self.x_1d - x))
        idx_y = np.argmin(np.abs(self.y_1d - y))
        idx_t = np.argmin(np.abs(self.times - t))
        idx_var = ['h', 'u', 'v'].index(var)
        return self.history[idx_t, idx_x, idx_y, idx_var]
    
    def get_timeline(self, x, y, var):
        idx_x = np.argmin(np.abs(self.x_1d - x))
        idx_y = np.argmin(np.abs(self.y_1d - y))
        idx_var = ['h', 'u', 'v'].index(var)
        return self.history[:, idx_x, idx_y, idx_var]
    
    def plot_timeline(self, x, y, var, **kwargs):
        idx_x = int(np.round(x / self.dx))
        idx_y = int(np.round(y / self.dy))
        idx_var = ['h', 'u', 'v'].index(var)
        args_dict = {'linewidth': 2, "color": 'tab:green'}
        args_dict.update(kwargs)
        plt.plot(self.times, self.history[:, idx_x, idx_y, idx_var], **args_dict)

    def plot_state(self, t=None):
        if t is None:
            idx_t = -1
        else:
            idx_t = np.argmin(np.abs(self.times - t))
        cax = plt.pcolormesh(self.x_2d, self.y_2d,
                             self.history[idx_t, :, :, 0],
                             vmin=-self.h_max, vmax=self.h_max)
        sample = self.nx // 10
        plt.quiver(self.x_2d[sample//2::sample,sample//2::sample],
                   self.y_2d[sample//2::sample,sample//2::sample],
                   self.history[idx_t, sample//2::sample, sample//2::sample, 1],
                   self.history[idx_t, sample//2::sample, sample//2::sample, 2])
        plt.colorbar(cax)
        plt.show()



class Model(Truth):
    def __init__(self, ens_size=10, dx=5):
        self.ens_size = ens_size
        self.dt = 1
        self.dx = dx
        self.dy = dx
        self.nx = int(X // self.dx)
        self.ny = int(Y // self.dy)
        self.x_1d = np.arange(0, self.nx*self.dx, self.dx)
        self.y_1d = np.arange(0, self.ny*self.dy, self.dy)
        self.y_2d, self.x_2d = np.meshgrid(self.y_1d, self.x_1d)
        self.u0 = np.zeros((self.ens_size, self.nx, self.ny))
        self.v0 = np.zeros((self.ens_size, self.nx, self.ny))
        self.h_max = 0.1
        self.h0 = np.zeros((self.ens_size, self.nx, self.ny)) * self.h_max
        self.state = np.stack([self.h0, self.u0, self.v0], axis=-1)
        self.history = self.state.reshape((1, self.ens_size, self.nx, self.ny, 3))
        self.time = 0
        self.times = np.array([0])
        return
    
    def initiate(self, truth, time=None, epsilon=0.01):
        if time is None:
            time = 0
        self.state = np.zeros((self.ens_size, self.nx, self.ny, 3))
        for i in range(self.nx):
            for j in range(self.ny):
                self.state[:, i, j, 0] = np.full(self.ens_size, truth.get_value(time, i * self.dx, j * self.dy, 'h'))
                self.state[:, i, j, 1] = np.full(self.ens_size, truth.get_value(time, i * self.dx, j * self.dy, 'u'))
                self.state[:, i, j, 2] = np.full(self.ens_size, truth.get_value(time, i * self.dx, j * self.dy, 'v'))
        self.state += np.random.normal(0, epsilon, size=(self.ens_size, self.nx, self.ny, 3))
        self.history = self.state.reshape((1, self.ens_size, self.nx, self.ny, 3))
        self.time = time
        self.times = np.array([time])
        return
    
    def blank_start(self):
        self.state = np.random.random((self.ens_size, self.nx, self.ny, 3)) * self.h_max
        self.history = self.state.reshape((1, self.ens_size, self.nx, self.ny, 3))
        self.time = 0
        self.times = np.array([0])
        return
    
    def step(self):
        self.state = RK_integration(self.state, self.dt, self.dx)
        self.history = np.concatenate([self.history, self.state.reshape((1, self.ens_size, self.nx, self.ny, 3))], axis=0)
        self.time = self.time + self.dt
        self.times = np.hstack([self.times, self.time])
        return
    
    def get_B(self):
        error = self.state - np.mean(self.state, axis=0)
        error = error.reshape((self.ens_size, self.nx * self.ny * 3))
        B = error.T @ error / (self.ens_size - 1)
        return B
    
    def get_P_clima(self):
        # poor climatological bg error cov matrix computed from the timeline of
        # the det member
        error = self.history[:,0] - np.mean(self.history[:,0], axis=0)
        error = error.reshape((len(self.times), self.nx * self.ny * 3))
        P = error.T @ error / (len(self.times) - 1)
        return P
    
    def get_H(self, obs):
        H = np.zeros((obs.n_obs, self.nx * self.ny * 3))
        for i in range(obs.n_obs):
            x, y = obs.coordinates[i]
            idx_x = np.argmin(np.abs(self.x_1d - x))
            idx_y = np.argmin(np.abs(self.y_1d - y))
            idx_var = ['h', 'u', 'v'].index(obs.var[i])
            H[i, np.ravel_multi_index((idx_x, idx_y, idx_var), (self.nx, self.ny, 3))] = 1
        return H
    
    def get_value(self, t, x, y, var):
        idx_x = int(np.round(x / self.dx))
        idx_y = int(np.round(y / self.dy))
        idx_t = int(np.round(t / self.dt))
        idx_var = ['h', 'u', 'v'].index(var)
        return self.history[idx_t, :, idx_x, idx_y, idx_var]
    
    def get_mean_value(self, t, x, y, var):
        idx_x = int(np.round(x / self.dx))
        idx_y = int(np.round(y / self.dy))
        idx_t = int(np.round(t / self.dt))
        idx_var = ['h', 'u', 'v'].index(var)
        return np.mean(self.history[idx_t, :, idx_x, idx_y, idx_var])
    
    def get_mean_timeline(self, x, y, var):
        idx_x = int(np.round(x / self.dx))
        idx_y = int(np.round(y / self.dy))
        idx_var = ['h', 'u', 'v'].index(var)
        return np.mean(self.history[:, :, idx_x, idx_y, idx_var], axis=1)
    
    def plot_mean_timeline(self, x, y, var, **kwargs):
        idx_x = int(np.round(x / self.dx))
        idx_y = int(np.round(y / self.dy))
        idx_var = ['h', 'u', 'v'].index(var)
        args_dict = {'linewidth': 2, "color": 'tab:red'}
        args_dict.update(kwargs)
        plt.plot(self.times, np.mean(self.history[:, :, idx_x, idx_y, idx_var], axis=1), **args_dict)
        
    def plot_det_timeline(self, x, y, var, **kwargs):
        # lets say ensemble member 0 is the det forecast
        idx_x = int(np.round(x / self.dx))
        idx_y = int(np.round(y / self.dy))
        idx_var = ['h', 'u', 'v'].index(var)
        args_dict = {'linewidth': 2, "color": 'tab:blue'}
        args_dict.update(kwargs)
        plt.plot(self.times, self.history[:, 0, idx_x, idx_y, idx_var], **args_dict)

    def plot_spaghetti_timeline(self, x, y, var, **kwargs):
        idx_x = int(np.round(x / self.dx))
        idx_y = int(np.round(y / self.dy))
        idx_var = ['h', 'u', 'v'].index(var)
        args_dict = {'linewidth': 0.1, "color": 'k'}
        args_dict.update(kwargs)
        for i in range(self.ens_size):
            plt.plot(self.times, self.history[:, i, idx_x, idx_y, idx_var], **args_dict)

    def plot_state(self, t=None):
        if t is None:
            idx_t = -1
        else:
            idx_t = np.argmin(np.abs(self.times - t))
        cax = plt.pcolormesh(self.x_2d, self.y_2d,
                             self.history[idx_t,:,:,:,0].mean(axis=0),
                             vmin=-self.h_max, vmax=self.h_max)
        sample = self.nx // 10
        plt.quiver(self.x_2d[sample//2::sample,sample//2::sample],
                   self.y_2d[sample//2::sample,sample//2::sample],
                   self.history[idx_t,:,sample//2::sample,sample//2::sample,1].mean(axis=0),
                   self.history[idx_t,:,sample//2::sample,sample//2::sample,2].mean(axis=0))
        plt.colorbar(cax)
        plt.show()

    def evaluate(self, truth):
        truth_history = np.empty((len(self.times), self.nx, self.ny, 3))
        for t in range(len(self.times)):
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(3):
                        truth_history[t, i, j, k] = truth.get_value(self.times[t], i * self.dx, j * self.dy, ['h', 'u', 'v'][k])
        error = np.sqrt(np.mean((np.mean(self.history, axis=1) - truth_history)**2, axis=(1, 2)))
        error_h, error_u, error_v = np.split(error, 3, axis=-1)
        error_h = error_h.squeeze()
        error_u = error_u.squeeze()
        error_v = error_v.squeeze()
        return error_h, error_u, error_v
    
    def plot_error(self, truth):
        error_h, error_u, error_v = self.evaluate(truth)
        plt.plot(self.times, error_h, label='h')
        plt.plot(self.times, error_u, label='u')
        plt.plot(self.times, error_v, label='v')
        plt.legend()

    def plot_spread(self):
        spread = np.std(self.history, axis=1)
        spread = np.mean(spread, axis=(1,2))
        spread_h, spread_u, spread_v = np.split(spread, 3, axis=-1)
        spread_h = spread_h.squeeze()
        spread_u = spread_u.squeeze()
        spread_v = spread_v.squeeze()
        plt.plot(self.times, spread_h, label='h')
        plt.plot(self.times, spread_u, label='u')
        plt.plot(self.times, spread_v, label='v')
        plt.legend()
    


class Observation:
    def __init__(self, truth, coordinates, var, noise=0.01):
        self.truth = truth
        self.noise = noise
        self.coordinates = coordinates # np.array of coordinates
        self.var = var # list of variables
        self.n_obs = len(coordinates)
        return
    
    def get_value(self, t):
        obs = []
        index_t = int(np.round(t / self.truth.dt))
        for i in range(self.n_obs):
            x, y = self.coordinates[i]
            obs.append((self.truth.get_value(t, x, y, self.var[i])
                        + np.random.normal(0, self.noise)).astype(np.float32))
        return np.array(obs), self.coordinates, self.var
    

class EnKF:
    def __init__(self, model, obs, sigma_obs=0.01):
        self.model = model
        self.obs = obs
        self.R = np.eye(obs.n_obs) * sigma_obs**2
        self.H = model.get_H(obs)
        self.n_obs = self.obs.n_obs
        self.nx = self.model.nx
        self.ny = self.model.ny
        self.ens_size = self.model.ens_size

    def compute_departure(self):
        # compute the departure or innovation d
        # d = y_o - H x_f with added observation noise
        time = self.model.time
        obs, coordinates, var = self.obs.get_value(time)
        obs_noise = np.random.normal(0, self.obs.noise, size=(self.ens_size, self.n_obs))
        H = self.H
        Hx = H @ self.model.state.reshape((self.ens_size, self.nx * self.ny * 3)).T
        self.departure = obs + obs_noise - Hx.T
        return self.departure

    def compute_kalman_gain(self):
        # compute the kalman gain K
        # K = B H^T (H B H^T + R)^-1
        B = self.model.get_B()
        H = self.H
        R = self.R
        K = B @ H.T @ np.linalg.pinv(H @ B @ H.T + R)
        self.K = K
        return self.K
    
    def analysis(self):
        # compute the formula
        # x_a = x_f + K (y_o - H x_f)
        # with K = B H^T (H B H^T + R)^-1
        self.compute_kalman_gain()
        self.compute_departure()
        analysis = self.model.state + np.reshape((self.K @ self.departure.T).T, (self.ens_size, self.nx, self.ny, 3))
        self.model.state = analysis
        self.model.history = np.concatenate([self.model.history, analysis.reshape((1, self.ens_size, self.nx, self.ny, 3))], axis=0)
        self.model.times = np.hstack([self.model.times, self.model.time])


class LETKF():
    def __init__(self, model, obs, sigma_obs=0.01):
        self.model = model
        self.obs = obs
        self.R = np.eye(obs.n_obs) * sigma_obs**2
        self.H = model.get_H(obs)
        self.n_obs = self.obs.n_obs
        self.nx = self.model.nx
        self.ny = self.model.ny
        self.ens_size = self.model.ens_size

    def compute_model_equivalents(self):
        # compute the model equivalents of the observations
        # H x_f
        H = self.H
        Hx = H @ self.model.state.reshape((self.ens_size, self.nx * self.ny * 3)).T
        self.Hx = Hx.T
        return self.Hx
        
    def compute_P_tilde(self):
        # compute the P_tile matrix
        # P_tile = (N-1)^-1 (H B H^T + R)^-1
        B = self.model.get_B()
        H = self.H
        R = self.R
        N = self.ens_size
        P_tilde = np.linalg.pinv(H @ B @ H.T + R) / (N - 1)
        self.P_tilde = P_tilde
        return self.P_tilde
    
    
class var3D():
    def __init__(self, model, obs, sigma_obs=0.01):
        self.model = model
        self.obs = obs
        self.R = np.eye(obs.n_obs) * sigma_obs**2
        self.H = model.get_H(obs)
        self.n_obs = self.obs.n_obs
        self.nx = self.model.nx
        self.ny = self.model.ny
        self.ens_size = self.model.ens_size
        
    def cost_function(self):
        #TODO: move to grad(J)=0 and use solve
        B_inv = np.linalg.pinv(self.model.get_P_clima())
        R_inv = np.linalg.pinv(self.R)
        x_b = self.model.state[0].flatten()
        time = self.model.time
        obs, coordinates, var = self.obs.get_value(time)
        def J(x):
            Hx = self.H @ x.T
            print('x: ', x[:3])
            print()
            cost = ( (x_b - x) @ B_inv @ (x_b - x)
                   + (obs - Hx) @ R_inv @ (obs - Hx))
            return cost
        return J
        
    def analysis(self):
        J = self.cost_function()
        print('cost before: {:.2f}'.format(J(self.model.state[0].flatten())))
        analysis = sp.optimize.minimize(J, self.model.state[0].flatten())
        self.model.state[0] = np.reshape(analysis.x, (self.nx, self.ny, 3))
        print('cost after: {:.2f}'.format(J(self.model.state[0].flatten())))
        self.model.history = np.concatenate([self.model.history, self.model.state.reshape((1, self.ens_size, self.nx, self.ny, 3))], axis=0)
        self.model.times = np.hstack([self.model.times, self.model.time])


#%%
# Playground
# define truth

truth = Truth()
truth.run_until(50, show=False)
truth.plot_timeline(25, 50, 'h')
plt.show()
#truth.plot_state()

#%%
# define model

model = Model(ens_size=1)
obs = Observation(truth, np.stack([model.x_2d.flatten(), model.y_2d.flatten()], axis=1),
                  model.nx * model.ny * ['h'], noise=0.001)
assimilation = var3D(model, obs, sigma_obs=0.01)
#model.blank_start()
model.initiate(truth, time=0, epsilon=0.0)

cycle(model, assimilation, 0, 50, 10, show=True)
#model.run_until(100, show=True)

model.plot_spaghetti_timeline(25, 50, 'h')
model.plot_mean_timeline(25, 50, 'h')
truth.plot_timeline(25, 50, 'h')
plt.show()

#model.plot_state()

#%%
# run analysis


model.plot_error(truth)
plt.show()


model.plot_spread()
plt.semilogy()
plt.show()

#%%
B = model.get_B()
plt.pcolormesh(B, cmap='bwr')
plt.colorbar()
plt.show()


P = model.get_P_clima()
plt.pcolormesh(P, cmap='bwr')
plt.colorbar()
plt.show()

#%%
idx = 200
plt.pcolormesh(np.reshape(P[idx,idx%3::3], (20,20)), cmap='bwr')
plt.colorbar()
