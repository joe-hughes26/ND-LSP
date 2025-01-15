#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:08:53 2024

@author: jhughes
"""
import ndlsp as nd
import numpy as np
import matplotlib.pyplot as plt


#%% 1D example

n = 150
t = np.random.rand(n)
f0 = 5
phi0 = 0
amp = 3

y = amp*np.sin(2 * np.pi * f0 * t + phi0) + np.random.randn(n) * 0.5
y -= np.mean(y)

#now run the LSP
fsamp = np.linspace(0.1, 10, 60)
X = np.reshape(t, [1, n])
A, phi = nd.lsp_nd(X, y, [fsamp])

#plot the results
f, (ax1, ax2) = plt.subplots(2,1, figsize = (6,6))
ax1.plot(t, y, '.', label = 'Data')
ax1.set_xlabel('Time []')
ax1.set_ylabel('Value []')

ax2.plot(fsamp, A)
ax2.set_xlabel('Frequency []')
ax2.set_ylabel('amplitude []')
ax2.plot([f0, f0], [0, amp], 'r--', label = f'Known max: Amp = {amp}, phi = {(phi0 * 180/np.pi):.1f} degrees')

#try the reconstruction
inds = [np.argmax(A)]

ax2.plot(fsamp[inds], A[inds], 'r*', label = f'Retrived max: Amp = {A[inds[0]]:.2f}, phi = {phi[inds[0]] * 180/np.pi:.1f} degrees')
ax2.legend()

grids = [np.linspace(0, 1, 200)]
yre = nd.reconstruct(A, phi, [fsamp], inds, grids)

ax1.plot(grids[0], yre, 'k--', label = 'Reconstruction')
ax1.legend()

plt.tight_layout()


#%% 2D example
n = 2000
x = np.random.rand(n)
y = np.random.rand(n)
X = np.vstack((x, y))

k1 = np.array([3, 4])
k2 = np.array([-3, 1])
z = 1 * np.sin(2*np.pi * (np.sum(k1[:, None] * X, axis = 0))) + \
    2 * np.sin(2*np.pi * (np.sum(k2[:, None] * X, axis = 0))) + \
    0.5 * np.random.randn(n)  

#now run the LSP
fx = np.linspace(-5, 5, 40)
fy = np.linspace(-5, 5, 41)
fs = [fx, fy]

A, phi = nd.lsp_nd(X, z, fs)

#plot the results - move this to a function eventually
f, (ax1, ax2) = plt.subplots(2,1, figsize = (6,8))
im = ax1.scatter(x, y, c=z)
cb = plt.colorbar(im, ax = ax1)
cb.set_label('Value []')
ax1.set_xlabel('X []')
ax1.set_ylabel('Y []')

im = ax2.contourf(fx, fy, A.T)
cb = plt.colorbar(im, ax = ax2)
plt.plot(k1[0], k1[1], 'rx')
plt.plot(*k2, 'rx')
cb.set_label('Amplitude []')
ax2.set_xlabel('X frequency []')
ax2.set_ylabel('Y frequency []')

#%% test the 95th percentile
A95 = nd.findQuantile(X, z, fs, q = 0.95, N = 20)

# and do automated detection
av, (fxv, fyv) = nd.findPeaks(A, A95, fs, factor = 3)

f = plt.figure()
plt.contourf(fx, fy, A.T)
plt.plot(fxv, fyv, 'm.')
cb = plt.colorbar()
cb.set_label('Amplitude []')
plt.xlabel('X frequency []')
plt.ylabel('Y frequency []')

#%% test the time handling

n = 1000
t = np.random.rand(n) * 3
x = np.random.rand(n) * 10
X = np.vstack((x, t))

fx0 = 0.25; ft0 = 1.5
z = np.sin(2*np.pi * (fx0*x - ft0*t)) + \
    0.1 * np.random.randn(n) 

#now run the LSP
ft = np.linspace(0, 3, 40)
fx = np.linspace(-1, 1, 41)
fs = [fx, ft]

A, phi = nd.lsp_nd(X, z, fs, timeAx= 1)

f, (ax1, ax2) = plt.subplots(2,1, figsize = (6,8))
im = ax1.scatter(t, x, c=z)
cb = plt.colorbar(im, ax = ax1)
cb.set_label('Value []')
ax1.set_xlabel('t []')
ax1.set_ylabel('x []')

im = ax2.contourf(ft, fx, A)
ax2.plot(ft0, fx0, 'rx')
cb = plt.colorbar(im, ax = ax2)
cb.set_label('Amplitude []')
ax2.set_xlabel('time frequency []')
ax2.set_ylabel('x frequency []')


#%% test 3d and plots

n = 2000
x = np.random.rand(n)
y = np.random.rand(n)
t = np.linspace(0, 10, n)
X = np.vstack((x, y, t))

k1 = np.array([3, 4]); f1 = 0.2
k2 = np.array([-3, 1]); f2 = 0.3
Y = 1 * np.sin(2*np.pi * (np.sum(k1[:, None] * X[:-1, :], axis = 0) - f1*t)) + \
    2 * np.sin(2*np.pi * (np.sum(k2[:, None] * X[:-1, :], axis = 0) - f2*t)) + \
    0.5 * np.random.randn(n)  

#now run the LSP
fx = np.linspace(-5, 5, 40)
fy = np.linspace(-5, 5, 41)
ft = np.linspace(0, 2, 22)
fs = [fx, fy, ft]

A, phi = nd.lsp_nd(X, Y, fs, timeAx = 2)

#%% now demo the plot

fnames = ('x freq []', 'y freq []', 't freq []')
f0 = np.array([*k2, f2])
title = 'test'

fig = nd.makeStairPlot(A, fs, f0, fnames, title = title)


#%% try it in 4D

n = 1000
x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)
t = np.linspace(0, 10, n)
X = np.vstack((x, y, z, t))

k1 = np.array([3, 4, 2]); f1 = 0.2
k2 = np.array([-3, 1, -1]); f2 = 0.3
Y = 1 * np.sin(2*np.pi * (np.sum(k1[:, None] * X[:-1, :], axis = 0) - f1*t)) + \
    2 * np.sin(2*np.pi * (np.sum(k2[:, None] * X[:-1, :], axis = 0) - f2*t)) + \
    0.5 * np.random.randn(n)  

#now run the LSP
fx = np.linspace(-5, 5, 30)
fy = np.linspace(-5, 5, 31)
fz = np.linspace(-5, 5, 32)
ft = np.linspace(0, 2, 22)
fs = [fx, fy, fz, ft]

A, phi = nd.lsp_nd(X, Y, fs, timeAx = 3)

#%%
fnames = ('x freq []', 'y freq []', 'z freq []', 't freq []')
f0 = np.array([*k1, f1])
title = 'test 4d'

fig = nd.makeStairPlot(A, fs, f0, fnames, title = title)

#%% try the inverse problem - given a f, make the field

inds = [np.argmin(np.abs(f0[i] - fs[i])) for i in range(len(fs))]
f00 = np.array([fs[i][inds[i]] for i in range(len(fs))])


gridx = np.linspace(0, 1, 100)
gridy = np.linspace(0, 1, 99)
gridz = np.linspace(0, 1, 98)
gridt = np.linspace(0, 1, 97)

grids = np.meshgrid(gridx, gridy, gridz, gridt, indexing='ij')
arg = np.zeros(grids[0].shape)

for i in range(len(fs)):
    arg += grids[i] * f00[i]
    
yre = np.cos(arg + phi[*inds]) # is it sine??






