"""
Purpose: The Big Kahuna. Connects all the other scripts and functions that have been written up to this point.
Given AOI(s) and a target date, this will:
    Download spatiotemporally proximate Landsat images around the AOI
        Merge into 4-band images (bands 2-5 for L8?)
    Download spatiotemporally proximate Planet images around the AOI
    Generate Landsat proxy images (??)
    Calibrate each Planet image to the appropriate (proxy) Landsat image using radcal
        Includes registration and radiometric calibration
    Calculate vegetation indices for each calibrated Planet image
    Calculate some zonal statistics for each given AOI
    Despike these statistics
    Fit with series of linear segments
"""

import numpy as np
import numpy.ma as ma
import gdal
import os
import custom_utils as cu
import datetime
from lmfit import Model
from functools import partial
from contextlib import contextmanager
from multiprocessing import Pool
import time
import warnings


def get_inputs():
    d = raw_input("Date of interest (YYYY-MM-DD): ")
    date_of_interest = datetime.date(int(d[0:4]), int(d[5:7]), int(d[8:10]))
    aoi = raw_input("Filepath of area of interest (.shp): ")
    direc = raw_input("Directory for outputs: ")
    ref_dir = raw_input("Directory of (radiometric) reference scenes: ")
    planet_dir = raw_input("Directory of Planet images: ")
    bqa_dir = raw_input("Directory of Landsat quality masks: ")   # will assume their names have not been changed
    trim = raw_input("Do the Landsat images in your directory need to be cropped? y/n: ")
    return date_of_interest, aoi, direc, trim, ref_dir, planet_dir, bqa_dir


def radiometric_calibration(target_img, reg_ref_img, rad_ref_img):
    calibrated_img_path = cu.main(target_img, reg_ref_img, rad_ref_img, allowDownsample=True,
                                            allowRegistration=True, view_radcal_fits=False)
    return calibrated_img_path


def harmonic_model(x, amp, period, hor_off, vert_off):
    return amp*np.sin(x*2*np.pi/period - hor_off) + vert_off


def read_rasters_as_arrays(rasterlist):
    """
    Given a list of filepaths to rasters, read each of them into a numpy array
    Return a list, where each item in the list is a tuple with one array for each band
    :param rasterlist: a list containing the filepath of all rasters to load into arrays (cropped to appropriate size)
    :return:
    """
    gdal.AllRegister()
    arrays = []
    # each item in list 'landsat_arrays' is a tuple of 4 arrays (1 for each band)
    # are all ordered by date
    for img in rasterlist:
        src = gdal.Open(img)
        bands = src.RasterCount
        temp_tuple = []
        for band in range(bands):
            temp_tuple.append(np.array(src.GetRasterBand(band + 1).ReadAsArray()))
        arrays.append(temp_tuple)
        src = None
    return arrays


def fit_model_to_pixel_ts(pixel, model, dates, initial_params=0., interp_dates=[], min_valid_points=5.):
    """
    Generate a set of parameters using a non-linear least-squares minimization method
    :param model: (function) the model chosen to fit the data
    :param pixel: (np masked array) a vector with the value of a pixel through time
    :param dates: (list or np array) a vector with the dates of each raster
    :param initial_params: (dictionary) keys are names of params, values are values
    :param interp_dates: (list or np array) a vector with the dates to evaluate each model (i.e. dates of Planet images)
    :return: array containing the model fit evaluated at the dates with Planet acquisitions
    """
    # in case pixel gets passed in a a list instead of a masked array:
    pixel = ma.asarray(pixel)
    # make sure there are enough usable values to actually fit a model
    if len(pixel[~pixel.mask]) <= min_valid_points:
        #print "Only found " + str(len(pixel[~pixel.mask])) + " usable pixels for an input pixel-location."
        # do a bad thing: just call the pixel constant through time with the average value of the available pixels
        if len(pixel[~pixel.mask]) == 0:
            return 0
        else:
            avg_value = np.average(pixel)
            if interp_dates == []:
                interp_dates = np.linspace(dates[0], dates[-1], dates[-1] + 1)
            constant_pixel_val = np.full(len(interp_dates), avg_value)
            return constant_pixel_val

    fit_model = Model(model, nan_policy='propagate')       # should this be "omit" instead??
    param_dict = dict()
    if initial_params != 0.:
        params = fit_model.make_params(**initial_params)
    else:
        # if initial arguments aren't given, grab the input arguments and prompt user for guesses
        import inspect
        param_names = inspect.getargspec(model)[0][1:]
        for p in param_names:
            param_dict[p] = float(raw_input("Initial value for " + p + " : "))
        params = fit_model.make_params(**param_dict)

    result = fit_model.fit(pixel, params, x=dates, nan_policy="omit")
    # make a vector with all the dates in the range. model will interpolate these values
    if interp_dates == []:
        interp_dates = np.linspace(dates[0], dates[-1], dates[-1]+1)
    interpolated_model = model(np.asarray(interp_dates), **result.best_values)
    return interpolated_model


def landsat_data_masker(landsat_bqa_path, mask_threshold=2720):
    landsat_bqa_array = cu.img_to_array(landsat_bqa_path)[0]
    # True = bad data. False = good data.
    landsat_bqa_mask = landsat_bqa_array > mask_threshold     # this could be refined to use bit order
    return landsat_bqa_mask


@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def main():
    doi, aoi, out_directory, trim, landsat_img_dir, planet_img_dir, bqa_dir = get_inputs()

    gdal.UseExceptions()
    # run the Landsat downloader here
        # create a list of all the Landsat images
    # run the Planet downloader next
        # create a list of all the Planet images

    landsat_img_paths = []
    landsat_img_dates = []
    for dirpath, dirnames, filenames in os.walk(landsat_img_dir):
        for filename in [f for f in filenames if f.endswith(".tif")]:
            landsat_img_paths.append(os.path.join(landsat_img_dir, filename))
            landsat_img_dates.append(datetime.date(int(filename[17:21]), int(filename[21:23]), int(filename[23:25])))
    # generate a vector containing the relative dates of each Landsat image as ints, starting with 0 for first image
    rel_date_ls = []
    for d in landsat_img_dates:
        rel_date_ls.append((d - landsat_img_dates[0]).days)

    planet_img_paths = []
    planet_img_dates = []
    exclude = {"EVI", "NDVI", "final_tifs"}
    for dirpath, dirnames, filenames in os.walk(planet_img_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for f in [f for f in filenames if f.endswith(".tif")]:
            planet_img_paths.append(os.path.join(planet_img_dir, f))
            planet_img_dates.append(datetime.date(int(f[0:4]), int(f[5:7]), int(f[8:10])))

    # TODO: if requested, trim all the Landsat images in the directory to the extent of the first Planet image

    # landsat_arrays[d] = list of 4 arrays (one for each band) acquired on date d (sort of, just ordered 0-last)
    # landsat_arrays[d][b] = single image at date d, band b
    landsat_arrays = read_rasters_as_arrays(landsat_img_paths)
    rows = int(landsat_arrays[0][0].shape[0])
    cols = int(landsat_arrays[0][0].shape[1])
    bands = len(landsat_arrays[0])
    dates = len(landsat_arrays)

    # build a list of the BQA bands. Will search directory recursively for files ending in "BQA.TIF"
    bqa_paths = []
    bqa_masks = []
    for dirpath, dirnames, filenames in os.walk(bqa_dir, topdown=True):
        for f in [f for f in filenames if f.endswith("BQA.TIF")]:
            bqa_paths.append(os.path.join(dirpath, f))
            bqa_masks.append(landsat_data_masker(os.path.join(dirpath, f)))

    # apply bad data mask to landsat arrays
    for d in range(0, dates):
        for b in range(0, bands):
            landsat_arrays[d][b] = ma.masked_array(landsat_arrays[d][b], mask=bqa_masks[d])

    # TODO: better way of getting the initial guesses only once (probably best to prompt user for guesses)
    initial_params = {'amp':3000, 'period':365, 'hor_off':0, 'vert_off':8000}

    # use the first Planet image in the list as the reference for registration
    reg_ref = planet_img_paths[0]
    # determine the dates of the Planet images in relation to the Landsat time series
    # take first Landsat image as day 0
    rel_date_planet = []
    for d in planet_img_dates:
        rel_date_planet.append((d - landsat_img_dates[0]).days)

    start = time.time()
    # currently, pixels becomes a list, where each item is an array with a single pixel's values through time (??)
    pixels = []
    temp_row = []
    temp_arr = []

    # TODO: fix this up? really janky and could probably be improved and switched to np arrays
    for b in range(0, bands):
        for j in range(0, rows):
            for i in range(0, cols):
                # append all dates in given location
                #item[band][row,col]
                temp_row.append([item[b][j, i] for item in landsat_arrays])  # all the rows for given column
            temp_arr.append(temp_row)  # builds array row-by-row
            temp_row = []
        pixels.append(temp_arr)
        temp_arr = []

    # for each pixel in the row, calculate the model, then interpolate values for Planet observation
    print "Beginning modeling... "
    pixel_models = []
    temp2 = []
    np.warnings.filterwarnings('once', '.*converting a masked element to nan.*')
    for b in range(0, bands):
        for i in range(0, rows):
            with poolcontext(processes=8) as pool:
                temp = (pool.map(partial(fit_model_to_pixel_ts, model=harmonic_model, dates=rel_date_ls,
                                        initial_params=initial_params, interp_dates=rel_date_planet), pixels[b][i]))
            temp2.append(temp)
        pixel_models.append(temp2)
        temp2 = []
    end = time.time()
    print "Modeling process took %5.2f seconds " % (end-start)

    # Build a usable Landsat image
    proxy_images = []
    for d in range(0, len(rel_date_planet)):
        temp_arr = None
        for b in range(0, bands):
            img = np.asarray([item[d] for item in pixel_models[b][0]])
            for row in range(1, rows):
                # build array row-by-row. This is grossly inefficient. Must be a more efficient way to do this
                img = np.vstack([img, np.asarray([item[d] for item in pixel_models[b][row]])])
            # right here, we have one complete array in a single band. Will then loop and do the next band
            try:    # if this is the first band, assign it to temp_arr. Otherwise, stack it with temp_arr (adding band)
                temp_arr = np.dstack((temp_arr, img))
            except ValueError:
                temp_arr = img
        proxy_images.append(temp_arr)

    # Save the Landsat images to disk
    # TODO: in order for this to work with the cu.array_to_img method, will need to build 3D arrays
    proxy_images_paths = []
    i = 0
    for proxy_img in proxy_images:
        fname = "landsat_proxy_d" + str(rel_date_planet[i]) + ".tif"
        print("Generating proxy image " + str(i+1))
        fpath = os.path.join(out_directory, fname)
        proxy_images_paths.append(fpath)
        cu.array_to_img(proxy_img, fpath, landsat_img_paths[0])     # use information from first landsat image in stack
        i += 1

    return


if __name__ == '__main__':
    main()
