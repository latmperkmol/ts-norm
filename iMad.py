#!/usr/bin/env python
#******************************************************************************
#  Name:     iMad.py
#  Purpose:  Perform IR-MAD change detection on bitemporal, multispectral
#            imagery 
#  Usage:             
#    python iMad.py 
#
#  Copyright (c) 2013, Mort Canty
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Modified by Nicholas Leach, 2019

import numpy as np
from scipy import linalg, stats
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_UInt16, GDT_Int16, GDT_Float32, GDT_Int32
import os, sys, time
import math


def run_MAD(image1, image2, outfile_name, band_pos1=(1,2,3,4), band_pos2=(1,2,3,4), penalty=0.0,
            datatype_out=GDT_Float32, outdir=None):
    """
    Tweaked version of iMad which eschews GUI.
    General requirements still required. Input images must have same spatial and spectral dimensions.
    All no-data values must also match for proper output.

    :param image1: Image which will receive radiometric calibration (target image). Include path.
    :param image2: Image with desired radiometry (reference image). Include path.
    :param outfile_name: Name for MAD image output. DO NOT INCLUDE PATH.
    :param band_pos1: Bands in image 1 to be radiometrically calibrated.
    :param band_pos2: Bands in image 2 which correspond to bands selected in band_pos1.
    :param penalty: (float) regularization parameter to penalize large weights
    :return:
    """
    gdal.AllRegister()
    path = os.path.split(image1)[0]
    if path:
        os.chdir(path)
#  first image
    file1 = image1
    if file1:
        inDataset1 = gdal.Open(file1,GA_ReadOnly)
        cols = inDataset1.RasterXSize
        rows = inDataset1.RasterYSize
        bands = inDataset1.RasterCount
    else:
        return
    pos1 =  band_pos1
    if not pos1:
        return
    dims = [0,0,cols,rows]
    if dims:
        x10,y10,cols1,rows1 = dims
    else:
        return
#  second image
    file2 = image2
    if file2:
        inDataset2 = gdal.Open(file2,GA_ReadOnly)
        cols = inDataset2.RasterXSize
        rows = inDataset2.RasterYSize
        bands = inDataset2.RasterCount
    else:
        return
    pos2 =  band_pos2
    if not pos2:
        return
    dims=[0,0,cols,rows]
    if dims:
        x20,y20,cols,rows = dims
    else:
        return
#  penalization
    lam = penalty
    if lam is None:
        return
#  outfile
    if outdir:
        dir_target = outdir
    else:
        dir_target = os.path.split(image1)[0]
    outfile, fmt = os.path.join(dir_target, outfile_name), "GTiff"
    if not outfile:
        return
#  match dimensions
    bands = len(pos2)
    if (rows1 != rows) or (cols1 != cols) or (len(pos1) != bands):
        sys.stderr.write("Size mismatch")
        sys.exit(1)
    print('=========================')
    print('       iMAD')
    print('=========================')
    print(time.asctime())
    print('time1: '+file1)
    print('time2: '+file2)
    print('Delta    [canonical correlations]')
#  iteration of MAD
    cpm = Cpm(2*bands)
    delta = 1.0
    oldrho = np.zeros(bands)
    itr = 0
    tile = np.zeros((cols,2*bands))  # tile is 2D array with 'cols' rows and '2*bands' columns
    sigMADs = 0
    means1 = 0
    means2 = 0
    A = 0
    B = 0
    rasterBands1 = []
    rasterBands2 = []
    for b in pos1:
        rasterBands1.append(inDataset1.GetRasterBand(b))
    for b in pos2:
        rasterBands2.append(inDataset2.GetRasterBand(b))
    # NL: continue while largest absolute change in canonical correlations is greater than largest permitted change
    while (delta > 0.001) and (itr < 100):
        # spectral tiling for statistics
        for row in range(rows):
            for k in range(bands):
                tile[:,k] = rasterBands1[k].ReadAsArray(x10,y10+row,cols,1)
                tile[:,bands+k] = rasterBands2[k].ReadAsArray(x20,y20+row,cols,1)
            # eliminate no-data pixels (assuming all zeroes)
            # NL: find no-data(0) pixels that exist in both images
            tst1 = np.sum(tile[:,0:bands],axis=1)      # NL: array w sum of rows in 'tile'
            tst2 = np.sum(tile[:,bands::],axis=1)      # NL: array w sum of rows in 'tile' corresponding to second image
            idx1 = set(np.where((tst1>0))[0])      # NL: index of first non-zero item in tst1
            idx2 = set(np.where((tst2>0))[0])
            # NL: create a list of all the rows where there are NOT no-data values in both images??
            # (not exactly right since tst is the sum of row locations)
            idx = list(idx1.intersection(idx2))
            if itr>0:
                mads = np.asarray((tile[:,0:bands]-means1)*A - (tile[:,bands::]-means2)*B)
                chisqr = np.sum((mads/sigMADs)**2,axis=1)
                wts = 1-stats.chi2.cdf(chisqr,[bands])
                cpm.update(tile[idx, :], wts[idx])    # NL: update means and covariance using no-data values??
            else:
                cpm.update(tile[idx, :])
#     weighted covariance matrices and means
        S = cpm.covariance()
        means = cpm.means()
#     reset prov means object (equations 6,7,8 from 2007 paper with lasso regularization)
        cpm.__init__(2*bands)
        s11 = S[0:bands,0:bands]
        s11 = (1-lam)*s11 + lam*np.eye(bands)
        s22 = S[bands:,bands:]
        s22 = (1-lam)*s22 + lam*np.eye(bands)
        s12 = S[0:bands,bands:]
        s21 = S[bands:,0:bands]
        c1 = s12*linalg.inv(s22)*s21
        b1 = s11
        c2 = s21*linalg.inv(s11)*s12
        b2 = s22
#     solution of generalized eigenproblems
        if bands>1:
            mu2a,A = geneiv(c1,b1)
            mu2b,B = geneiv(c2,b2)
#          sort a
            idx = np.argsort(mu2a)
            A = A[:,idx]
#          sort b
            idx = np.argsort(mu2b)
            B = B[:,idx]
            mu2 = mu2b[idx]
        else:
            mu2 = c1/b1
            A = 1/np.sqrt(b1)
            B = 1/np.sqrt(b2)
#      canonical correlations
        mu = np.sqrt(mu2)
        a2 = np.diag(A.T*A)
        b2 = np.diag(B.T*B)
        sigma = np.sqrt( (2-lam*(a2+b2))/(1-lam)-2*mu )
        rho=mu*(1-lam)/np.sqrt( (1-lam*a2)*(1-lam*b2) )
#      stopping criterion
        delta = max(abs(rho-oldrho))
        print(delta, rho)
        oldrho = rho
#      tile the sigmas and means
        sigMADs = np.tile(sigma,(cols,1))
        means1 = np.tile(means[0:bands],(cols,1))
        means2 = np.tile(means[bands::],(cols,1))
#      ensure sum of positive correlations between X and U is positive
        D = np.diag(1/np.sqrt(np.diag(s11)))
        s = np.ravel(np.sum(D*s11*A,axis=0))
        A = A*np.diag(s/np.abs(s))
#      ensure positive correlation between each pair of canonical variates
        cov = np.diag(A.T*s12*B)
        B = B*np.diag(cov/np.abs(cov))
        itr += 1
# write results to disk
    driver = gdal.GetDriverByName(fmt)
    outDataset = driver.Create(outfile, cols, rows, bands+1, datatype_out)  # want to write a float!!
    projection = inDataset1.GetProjection()
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x10*gt[1]
        gt[3] = gt[3] + y10*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)
    outBands = []
    for k in range(bands+1):
        outBands.append(outDataset.GetRasterBand(k+1))
    for row in range(rows):
        for k in range(bands):
            tile[:,k] = rasterBands1[k].ReadAsArray(x10,y10+row,cols,1)
            tile[:,bands+k] = rasterBands2[k].ReadAsArray(x20,y20+row,cols,1)
        mads = np.asarray((tile[:,0:bands]-means1)*A - (tile[:,bands::]-means2)*B)  # MADs
        chisqr = np.sum((mads/sigMADs)**2,axis=1)  # under no-change hypothesis, this sum is approx. chi2 distributed
        for k in range(bands):
            outBands[k].WriteArray(np.reshape(mads[:,k],(1,cols)),0,row)
        outBands[bands].WriteArray(np.reshape(chisqr,(1,cols)),0,row)       # output: first outbands are MADs for corresponding bands. Final "band" is the chisquared image.
    for outBand in outBands:
        outBand.FlushCache()
    outDataset = None
    inDataset1 = None
    inDataset2 = None
    print('result written to: '+outfile)
    print('--------done---------------------')
    return


class Cpm(object):
    """Provisional means algorithm"""

    def __init__(self, N):
        self.mn = np.zeros(N)
        self.cov = np.zeros((N, N))
        self.sw = 0.0000001

    def update(self, Xs, Ws=None):
        # testing a version of update that eschews the provisional means dll
        # it will be less efficient but easier to troubleshoot and implement
        # reference Nick Leach's python implementation provisional_means
        n, N = np.shape(Xs)
        if Ws is None:
            Ws = np.ones(n)
        sw = self.sw
        mn = self.mn
        cov = self.cov
        n, N = np.shape(Xs)
        sw, mn, cov = provisional_means(Xs, Ws, N, n, sw, mn, cov)
        self.sw = sw
        self.mn = mn
        self.cov = cov

    def covariance(self):
        c = np.mat(self.cov / (self.sw - 1.0))
        d = np.diag(np.diag(c))
        return c + c.T - d

    def means(self):
        return self.mn


def choldc(A):
    # Cholesky-Banachiewicz algorithm,
    # A is a numpy matrix
    L = A - A
    for i in range(len(L)):
        for j in range(i):
            sm = 0.0
            for k in range(j):
                sm += L[i,k]*L[j,k]
            L[i,j] = (A[i,j]-sm)/L[j,j]
        sm = 0.0
        for k in range(i):
            sm += L[i,k]*L[i,k]
        L[i,i] = math.sqrt(A[i,i]-sm)
    return L


def geneiv(A,B):
    # solves A*x = lambda*B*x for numpy matrices A and B,
    # returns eigenvectors in columns
    Li = np.linalg.inv(choldc(B))
    C = Li*A*(Li.transpose())
    C = np.asmatrix((C + C.transpose())*0.5, np.float32)
    eivs,V = np.linalg.eig(C)
    return eivs, Li.transpose()*V


def provisional_means(Xs, Ws, NN, n, sw, mn, cov):
    """
    Provisional means algorithm, adapted from C++ code written by Mort Canty
    In original script, Xs is pointer, Ws is pointer, NN is int, n is int, sw is double, mn is pointer, cov is pointer.
    Written by Nick Leach
    :param Xs: (numpy array) input array with data values. Must be 1D??
    :param Ws: (numpy array) weights of the input values??
    :param NN: (int) columns of Xs
    :param n: (int) rows of Xs
    :param sw: (float, I think) offset parameter or something. Probably stands for "sum of the weights"
    :param mn: (numpy array) empty 1D numpy array with length N. Will store means.
    :param cov: (numpy array) empty 2D numpy array with dimensions n, N
    :return:
    """
    d = np.zeros(n*NN)  # empty array
    Xs = Xs.flatten()
    cov_shape = cov.shape
    cov = cov.flatten()
    for i in range(0, n):
        w = Ws[i]  # equivalent to dereferencing the Ws array in C
        sw += Ws[i]  # sw = sw+Ws (essentially adding a small offset to Ws the first time. sw may change later)
        r = w/sw  # r is the ith weight divided by the sum of the weights
        # update the means mn
        for j in range(0, NN):
            d[j] = Xs[i*NN+j] - mn[j]
            mn[j] += d[j]*r
        # update the covariance cov
        for j in range(0, NN):
            for k in range(j, NN):
                cov[j*NN+k] += d[j]*d[k]*(1-r)*w
    cov = np.reshape(cov, cov_shape)
    return sw, mn, cov
