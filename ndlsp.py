#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:28:54 2024

@author: jhughes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from copy import deepcopy

def lsp_nd(X, Y, fs0, timeAx = None):
    '''
    Assume n measurements of m dependant variables and 1 independant varible.
    Further assume m freqency vectors for the m dependant variables each with
    m_i elements.
    Inputs: X - a vertical stack of the dependant variables. shape = [m, n]
            Y - a np array of the independant variable. shape = n
            fs - a list of the desired frequency arrays. all in units of 1/X.
                 the list has m elements, and each element contains an array
                 with m_i elements
            timeAx - if one of your dimensions is time, provide it here as an 
            integer. if none of your dimensions are time, leave as None
    Outputs: A - the amplitude of the signal at the frequency. shape =
                 [m_1, m_2, m_3, ...]
            phi - the phase of the signal at the frequency. shape =
                 [m_1, m_2, m_3, ...]
    '''
    if timeAx is not None:
        fs = deepcopy(fs0)
        fs[timeAx] *= -1
    else:
        fs = deepcopy(fs0)
        
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
    phi = np.arctan2(b, a) + tau - np.pi/2
    
    return(A, phi)

def findQuantile(X, Y, fs, q = 0.95, N = 20):
    '''
    This function finds the qth quantile of the grid noise by running the ND LSP
    on shuffled inputs. Useful for determinding significance of peaks. Assume n 
    measurements of m dependant variables and 1 independant varible. Further 
    assume m freqency vectors for the m dependant variables each with
    m_i elements.
    
    Inputs: X - a vertical stack of the dependant variables. shape = [m, n]
            Y - a np array of the independant variable. shape = n
            fs - a list of the desired frequency arrays. all in units of 1/X.
                 the list has m elements, and each element contains an array
                 with m_i elements
            q - the quantile, default is 95%
            N - number of instances to run for. default is 20 so that the 19th 
                corresponds to the 95th percentile
            
    Outputs: A - the amplitude of the shuffled signal at the frequency. shape =
                 [m_1, m_2, m_3, ...]

    '''
    
    #copy Y so that it is unaffected by shuffling
    Yr = np.copy(Y)
    
    #preallocate A
    ss = [f.size for f in fs] # size of the frequencies
    Ar = np.zeros([*ss, N]) * np.nan
    
    #run the nd lsp N times, storing the output
    for i in range(N):
        print(f'Working on random instance {i+1} out of {N}')
        np.random.shuffle(Yr)
        Ar[..., i], _ = lsp_nd(X, Yr, fs)
        
    Aq = np.quantile(Ar, q, axis = -1)
    
    return(Aq)

def findPeaks(A, Aq, fs, factor = 2):
    '''
    This function finds statistically significant peaks in the ND LSP output.
    
    Inputs: A - amplitude as a function of frequency. made with lsp_nd()
            Aq - amplitude of the SHUFFLED signal as a funtion of frequency. made
            with findQuantile()
            fs - a list of the desired frequency arrays. all in units of 1/X.
                 the list has m elements, and each element contains an array
                 with m_i elements
            factor - fudge factor to make this less sensitive when the grid is 
            close to uniform
            
    Outputs: av - 1D array of the significant amplitudes
             fsv - list of all the significant frequencies. unpack this in the 
             order of fs

    '''
    
    n = len(fs)
    
    #create a kernel that is 1 everywhere but the center
    kernel = np.ones([3] * n)
    kernel[*np.ones(n).astype(int)] = 0

    local_max = maximum_filter(A, footprint = kernel, mode='nearest')
    valid = np.nonzero((A > factor * Aq) * (A > local_max))
    #mark a point as valid if it's larger than the grid noise AND is a local max
    
    #find the amplitude of these valid points and sort them
    av0 = A[valid]
    si = np.argsort(-av0)
    av = av0[si]
    
    #now find the frequencies of these valid points
    Fs = np.meshgrid(*fs, indexing='ij')
    fsv = [(x[valid])[si] for x in Fs]

    return(av, fsv)

def makeStairPlot(A, fs, f0, fnames, title = None, Aq = None):
    #find the indices coresponding to f0
    n = len(fs)
    inds = [np.argmin(np.abs(f0[ii] - fs[ii])) for ii in range(n)]
    Amax = np.nanmax(A)

    #make the figure
    fig, axs = plt.subplots(n-1, n-1, sharey='row', figsize = (8,8))
    levs = np.linspace(0, Amax, 15)
    kwd = {'extend':'max', 'cmap':'inferno'}

    for ii in range(n-1):
        fyi = fs[ii]
        
        for jj in range(ii, n-1):
            
            fxj = fs[jj+1]
            #axis ii and jj+1 are free
            inds2 = inds.copy()
            inds2[ii] = slice(None)
            inds2[jj+1] = slice(None)
            
            a = axs[ii,jj]
            im = a.contourf(fxj, fyi, A[tuple(inds2)], levs, **kwd)
            a.plot(f0[jj+1], f0[ii], 'x', color = 'cyan')
            if Aq is not None:
                a.contourf(fxj, fyi, A[tuple(inds2)] / Aq[tuple(inds2)], [0, 1],
                           hatches = ['//'], cmap='gray', alpha = 0.25)
            
            if jj == ii:
                a.set_ylabel(fnames[ii])
                a.set_xlabel(fnames[jj+1])
            
        for kk in range(ii):
            axs[ii,kk].remove()
            
    cax = fig.add_axes([0.15, 0.2, 0.3, 0.02])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal', 
                      ticks = np.around(levs[::3], 2))
    cb.set_label('Amplitude []')

    s = r'Central point: $1/f$ =' + f' {[np.around(1/f,2) for f in f0]}'
    plt.text(-1, -5, s)
    
    if title is not None: fig.suptitle(title)
    
    return(fig)

def reconstruct(A, phi, fs, inds, grids):
    
    '''
    This function returns a reconstruction of 1 wave specified by inds.
    
    Inputs: A - amplitude as a function of frequency. made with lsp_nd()
            phi - phase as a function of frequency. made with lsp_nd()
            fs - a list of the desired frequency arrays. all in units of 1/X.
            inds - iterable of length = A.ndim. describes the wave you want to reconstruct
            grids - iterable of length = A.dim. contains the points to sample at
            
    Outputs: yre - the signal. same shape of each elements of grids

    '''
    
    N = len(fs)
    #assert(N == A.ndim == phi.ndim == len(grids) == len(inds) )
    
    f0 = np.array([fs[i][inds[i]] for i in range(N)]) # find the peak frequency to measure

    Grids = np.meshgrid(*grids, indexing='ij') #expand the grids - unsure if this fails in 1D
    
    arg = np.zeros(Grids[0].shape) #expand the grids and iterate over them
    for i in range(N):
        arg += 2*np.pi * Grids[i] * f0[i]
        
    yre = A[inds] * np.sin(arg + phi[*inds]) # is it sine or cosine??
    
    return yre