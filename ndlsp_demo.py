#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:08:53 2024

@author: jhughes
"""
import numpy as np
import matplotlib.pyplot as plt


def lsp_nd(X, Y, fs):
    '''
    Assume n measurements of m dependant variables and 1 independant varible.
    Further assume m freqency vectors for the m dependant variables each with
    m_i elements.
    Inputs: X - a vertical stack of the dependant variables. shape = [m, n]
            Y - a np array of the independant variable. shape = n
            fs - a list of the desired frequency arrays. all in units of 1/X.
                 the list has m elements, and each element contains an array
                 with m_i elements
    Outputs: A - the amplitude of the signal at the frequency. shape =
                 [m_1, m_2, m_3, ...]
            phi - the phase of the signal at the frequency. shape =
                 [m_1, m_2, m_3, ...]
    '''
    m,n = X.shape
    xshape = [m, *np.ones(m, dtype=int), n]
    yshape = [*np.ones(m, dtype=int), n]
    
    W = 2*np.pi * np.stack(np.meshgrid(*fs, indexing='ij'))
    Xs = np.reshape(X, xshape)
    
    arg = np.sum( W[..., None] * Xs, axis = 0)
    t1 = np.sum(np.sin(2 * arg), axis = -1)
    b1 = np.sum(np.cos(2 * arg), axis = -1)
    tau = np.arctan2(t1, b1)/2
    
    Ys = np.reshape(Y, yshape)
    t2 = np.sum(Ys * np.cos(arg - tau[..., None]), axis = -1)
    b2 = np.sum(np.cos(arg - tau[..., None])**2, axis = -1)
    a = t2/b2
    
    t3 = np.sum(Ys * np.sin(arg - tau[..., None]), axis = -1)
    b3 = np.sum(np.sin(arg - tau[..., None])**2, axis = -1)
    b = t3/b3
    
    A2 = a**2 + b**2
    A = np.sqrt(A2)
    phi = np.arctan2(b, a)
    
    return(A, phi)

#%% 1D example

n = 100
t = np.random.rand(n)
f0 = 5
y = np.sin(2 * np.pi * f0 * t) + np.random.randn(n) * 0.5

#now run the LSP
fsamp = np.linspace(0.1, 10, 40)
X = np.reshape(t, [1, n])
A, phi = lsp_nd(X, y, [fsamp])

#plot the results
f, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(t, y, '.')
ax1.set_xlabel('Time []')
ax1.set_ylabel('Value []')

ax2.plot(fsamp, A)
ax2.set_xlabel('Frequency []')
ax2.set_ylabel('amplitude []')

#%% 2D example
n = 1000

x = np.random.rand(n)
y = np.random.rand(n)
z = np.sin(2 * np.pi*(3*x + 4*y)) + np.random.randn(n) * 0.5

#now run the LSP
fsampx = np.linspace(0.1, 10, 40)
fsampy = np.linspace(0.1, 10, 30)
F = [fsampx, fsampy]
X = np.vstack((x, y))
A, phi = lsp_nd(X, z, F)

#plot the results
f, (ax1, ax2) = plt.subplots(2,1, figsize = (6,8))
im = ax1.scatter(x, y, c=z)
cb = plt.colorbar(im, ax = ax1)
cb.set_label('Value []')
ax1.set_xlabel('X []')
ax1.set_ylabel('Y []')

im = ax2.contourf(fsampx, fsampy, A.T)
cb = plt.colorbar(im, ax = ax2)
cb.set_label('Amplitude []')
ax2.set_xlabel('X frequency []')
ax2.set_ylabel('Y frequency []')





