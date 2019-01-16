#!/usr/bin/env python
#******************************************************************************
#  Name:     radcal.py
#  Purpose:  Automatic radiometric normalization
#  Usage:             
#       python radcal.py  
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

import auxil.auxil as auxil
import sys, os, time
import numpy as np 
from scipy import stats
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32, GDT_Int32, GDT_UInt16
import matplotlib.pyplot as plt
import json
 
def run_radcal(image1, image2, outfile_name, iMAD_img, full_target_scene, band_pos1=[1,2,3,4], band_pos2=[1,2,3,4],
               nochange_thresh=0.95, view_plots=True, add_nodata=True):
    """

    :param image1: Image which will receive radiometric calibration (target image). Include path.
    :param image2: Image with desired radiometry (reference image). Include path.
    :param outfile_name: Name for radcal image output. DO NOT INCLUDE PATH.
    :param iMAD_img: Output image from iMAD. Include path.
    :param full_target_scene: The full-resolution, full-size image to receive radiometric corrections.
    :param band_pos1: positions of the bands to be calibrated in image 1.
    :param band_pos2: positions of the bands to be calibrated in image 2.
    :param nochange_thresh:
    :return:
    """

    gdal.AllRegister()
    path = os.path.split(image1)[0]
    if path:
        os.chdir(path)      
#  reference image    
    file1 = image2
    if file1:                  
        inDataset1 = gdal.Open(file1,GA_ReadOnly)     
        cols = inDataset1.RasterXSize
        rows = inDataset1.RasterYSize    
        bands = inDataset1.RasterCount
    else:
        return
    pos1 = band_pos2
    if not pos1:
        return   
    dims = [0,0,cols,rows]
    if dims:
        x10,y10,cols1,rows1 = dims
    else:
        return 
#  target image     
    file2 = image1
    if file2:                  
        inDataset2 = gdal.Open(file2,GA_ReadOnly)     
        cols = inDataset2.RasterXSize
        rows = inDataset2.RasterYSize    
        bands = inDataset2.RasterCount
    else:
        return   
    pos2 = band_pos1
    if not pos2:
        return 
    dims=[0,0,cols,rows]
    if dims:
        x20,y20,cols2,rows2 = dims
    else:
        return  
#  match dimensions       
    bands = len(pos2)
    if (rows1 != rows2) or (cols1 != cols2) or (len(pos1) != bands):
        sys.stderr.write("Size mismatch")
        sys.exit(1)             
#  iMAD image     
    file3 = iMAD_img
    if file3:                  
        inDataset3 = gdal.Open(file3,GA_ReadOnly)     
        cols = inDataset3.RasterXSize
        rows = inDataset3.RasterYSize    
        imadbands = inDataset3.RasterCount
    else:
        return   
    dims = [0, 0, cols, rows]
    if dims:
        x30,y30,cols,rows = dims
    else:
        return     
    if (rows1 != rows) or (cols1 != cols):
        sys.stderr.write("Size mismatch")
        sys.exit(1)    
#  outfile
    outfile, fmt = outfile_name, "GTiff"
    if not outfile:
        return    
#  full scene
    fsfile = full_target_scene
#  no-change threshold    
    ncpThresh = nochange_thresh
    if ncpThresh is None:
        return                 
    chisqr = inDataset3.GetRasterBand(imadbands).ReadAsArray(x30,y30,cols,rows).ravel()
    ncp = 1 - stats.chi2.cdf(chisqr,[imadbands-1])
    idx = np.where(ncp>ncpThresh)[0]
#  split train/test in ratio 2:1 
    tmp = np.asarray(range(len(idx)))
    tst = idx[np.where(np.mod(tmp,3) == 0)]
    trn = idx[np.where(np.mod(tmp,3) > 0)]
    
    print('=========================================')
    print('             RADCAL')
    print('=========================================')
    print(time.asctime())
    print('reference: '+file1)
    print('target   : '+file2)
    print('no-change probability threshold: '+str(ncpThresh))
    print('no-change pixels (train): '+str(len(trn)))
    print('no-change pixels (test): '+str(len(tst)))
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,bands,GDT_UInt16)
    projection = inDataset1.GetProjection()
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x10*gt[1]
        gt[3] = gt[3] + y10*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)      
    aa = []
    bb = []  
    i = 1
    log = []
    for k in pos1:
        x = inDataset1.GetRasterBand(k).ReadAsArray(x10,y10,cols,rows).astype(float).ravel()
        y = inDataset2.GetRasterBand(k).ReadAsArray(x20,y20,cols,rows).astype(float).ravel() 
        b, a, R = auxil.orthoregress(y[trn],x[trn])  # trn is the vector of training points
        mean_tgt, mean_ref, mean_nrm = np.mean(y[tst]), np.mean(x[tst]), np.mean(a+b*y[tst])
        t_test = stats.ttest_rel(x[tst], a+b*y[tst])
        var_tgt, var_ref, var_nrm = np.var(y[tst]), np.var(x[tst]), np.var(a+b*y[tst])
        F_test = auxil.fv_test(x[tst], a+b*y[tst])
        print('--------------------')
        print('spectral band:      ', k)
        print('slope:              ', b)
        print('intercept:          ', a)
        print('correlation:        ', R)
        print('means(tgt,ref,nrm): ', mean_tgt, mean_ref, mean_nrm)
        print('t-test, p-value:    ', t_test)
        print('vars(tgt,ref,nrm)   ', var_tgt, var_ref, var_nrm)
        print('F-test, p-value:    ', F_test)
        aa.append(a)
        bb.append(b)
        fit_info = {'k':k, 'b':b, 'a':a, 'R':R, 'mean_tgt':mean_tgt, 'mean_ref':mean_ref, 'mean_nrm':mean_nrm,
                    't_test':t_test, 'var_tgt':var_tgt, 'var_ref':var_ref, 'var_nrm':var_nrm, 'F_test':F_test}
        log.append(fit_info)
        outBand = outDataset.GetRasterBand(i)
        outBand.WriteArray(np.resize(a+b*y,(rows,cols)),0,0) 
        outBand.FlushCache()
        if i <= 10:
            plt.figure(i)    
            ymax = max(y[idx]) 
            xmax = max(x[idx])      
            plt.plot(y[idx],x[idx],'k.',[0,ymax],[a,a+b*ymax],'k-')
            plt.axis([0,ymax,0,xmax])
            plt.title('Band '+str(k))
            plt.xlabel('Target')
            plt.ylabel('Reference')
            if view_plots:
                plt.show()
        i += 1
    # write out a log with radcal fit information
    log_outpath = os.path.join(path, 'radcal_parameters.json')
    with open(log_outpath, "w") as write_file:
        json.dump(log, write_file)

    outDataset = None
    print('result written to: '+outfile)
    if fsfile is not None:
        path = os.path.dirname(fsfile)
        basename = os.path.basename(fsfile)
        root, ext = os.path.splitext(basename)
        fsoutfile = path+'/'+root+'_norm'+ext        
        print('normalizing '+fsfile+'...')
        fsDataset = gdal.Open(fsfile,GA_ReadOnly)
        cols = fsDataset.RasterXSize
        rows = fsDataset.RasterYSize    
        driver = fsDataset.GetDriver()
        outDataset = driver.Create(fsoutfile,cols,rows,bands,GDT_UInt16)
        projection = fsDataset.GetProjection()
        geotransform = fsDataset.GetGeoTransform()
        if geotransform is not None:
            outDataset.SetGeoTransform(geotransform)
        if projection is not None:
            outDataset.SetProjection(projection) 
        j = 0
        for k in pos2:
            inBand = fsDataset.GetRasterBand(k)
            outBand = outDataset.GetRasterBand(j+1)
            for i in range(rows):
                y = inBand.ReadAsArray(0,i,cols,1)
                outBand.WriteArray(aa[j]+bb[j]*y,0,i)   # this is where the operations happen. Operates by row
            outBand.FlushCache() 
            j += 1      
        outDataset = None    
        print('result written to: '+fsoutfile)

    print('-------done-----------------------------')
    if fsfile is not None:
        return fsoutfile
    else:
        return