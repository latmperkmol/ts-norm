
# Usage: from command line in appropriate folder;
# >> python custom_utils.py
# Will prompt for names and file locations of images.

import future
import past
import six
from builtins import input
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import json
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import numpy.ma as ma
import gdal, osr
from gdalconst import GA_ReadOnly
from subprocess import call
import os
import time
import arosics
from iMad import run_MAD
from radcal import run_radcal


def process_udm(udm_path, src_nodata=1.0):
    """
    Takes a default Planet Unusable Data Mask and compresses it to one bit, where everything originally = 1.0 or greater
        becomes nodata. Currently still saves as an 8bit file though.
        NB: Important!! This edits the UDM in place!
    :param udm_path:
    :param src_nodata:
    :return:
    """
    file_name, directory = os.path.split(udm_path)
    src = rasterio.open(udm_path, "r+")
    src_data = src.read(1)
    updated_src = np.where(src_data >= 1, 1, src_data)
    src.write(updated_src, 1)
    src.nodata = src_nodata
    src.close()
    return file_name


def udm_merger(rasterpaths, outpath, src_nodata=1.0):
    """
    Mosaics UDMs.
    Assumes that the UDMs already have been processed via process_udm so that the nodata values are set appropriately.
    Expects
    :param rasterpaths: (tuple) udm filepaths to be merged
    :param outpath: (string)
    :param src_nodata:
    :return:
    """
    from rasterio.merge import merge
    src_files = []
    for r in rasterpaths:
        src = rasterio.open(r, "r+")
        src.nodata = src_nodata
        src_files.append(src)
    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans})
    with rasterio.open(outpath, "w", **out_meta) as dest:
        dest.write(mosaic)
    return outpath


def save_VIs(rasterpath, outdir=None, nodataval = 0.0, save_ndvi=True, save_evi=False):
    """
    Generates two new GeoTiffs: the NDVI and EVI for the input raster.

    :param rasterpath: path to the raster
    :param outdir: (str) directory to save outputs
    :param nodataval: no-data value in the input images
    :return:
    """
    ndvi, evi, img_src, cols, rows = calc_VIs(rasterpath, nodataval)

    # grab the directory and original file name, then build new filenames
    dir_src, filename_src = os.path.split(rasterpath)[0], os.path.split(rasterpath)[1]
    if outdir:
        filename_ndvi_src = os.path.join(outdir, filename_src[:-4] + '-NDVI.tif')
    else:
        filename_ndvi_src = os.path.join(dir_src, filename_src[:-4] + '-NDVI.tif')
    filename_evi_src = os.path.join(dir_src, filename_src[:-4] + '-EVI.tif')

    # TODO: rewrite these two loops to use rasterio to reduce file locks and improve readability
    if save_ndvi:
        ndvi_dst = gdal.GetDriverByName('GTiff').Create(filename_ndvi_src, cols, rows, 1, gdal.GDT_Float32)
        ndvi_dst.GetRasterBand(1).WriteArray(ndvi[:, :])
        ndvi_dst.SetGeoTransform(img_src.GetGeoTransform())
        ndvi_dst.SetProjection(img_src.GetProjection())
        ndvi_dst.FlushCache()
        ndvi_dst = None
    if save_evi:
        evi_dst = gdal.GetDriverByName('GTiff').Create(filename_evi_src, cols, rows, 1, gdal.GDT_Float32)
        evi_dst.GetRasterBand(1).WriteArray(evi[:, :])
        evi_dst.SetGeoTransform(img_src.GetGeoTransform())
        evi_dst.SetProjection(img_src.GetProjection())
        evi_dst.FlushCache()
        evi_dst = None
    # Close up shop
    img_src = None
    src = None
    return


def calc_VIs(rasterpath, nodataval=0.0):
    """
    Generates numpy arrays of the NDVI and EVI
    # NB: assumes that blue is band 1, red is band 3, and NIR is band 4 (indexed from 1)
    :param rasterpath: filepath of the raster
    :param nodataval: no-data value in the input images
    :return:
    """
    nodataval = int(nodataval)
    print("Calculating vegetation indices... ")
    # Set EVI coefficients
    G, C1, C2, L = 2.5, 6.0, 7.5, 1.0

    # Register drivers with gdal and get image properties
    gdal.AllRegister()
    img_src = gdal.Open(rasterpath)
    cols = img_src.RasterXSize
    rows = img_src.RasterYSize
    bands = img_src.RasterCount

    # Read in bands from original image
    blue_band = np.array(img_src.GetRasterBand(1).ReadAsArray()).astype(np.float)
    red_band = np.array(img_src.GetRasterBand(3).ReadAsArray()).astype(np.float)
    NIR_band = np.array(img_src.GetRasterBand(4).ReadAsArray()).astype(np.float)

    # Create masked versions of these arrays
    blue_band_ma = ma.masked_values(blue_band, nodataval)
    red_band_ma = ma.masked_values(red_band, nodataval)
    NIR_band_ma = ma.masked_values(NIR_band, nodataval)

    # allow division by 0
    np.seterr(divide='ignore', invalid='ignore')

    ndvi = (NIR_band_ma - red_band_ma)/(NIR_band_ma + red_band_ma)
    evi = (NIR_band_ma - red_band_ma)/(NIR_band_ma + C1*red_band_ma - C2*blue_band_ma + L)

    # the new arrays aren't masked, but have masked values filled with the set fill_value (nodataval)
    ndvi_filled = np.where(ndvi.mask, ndvi.fill_value, ndvi.data)
    evi_filled = np.where(evi.mask, evi.fill_value, evi.data)

    return ndvi_filled, evi_filled, img_src, cols, rows


def vegetation_mask(ndvi, threshold=0.0):
    """
    Create a boolean array selecting all pixels with NDVI above a set a threshold
    :param ndvi: numpy masked array of the NDVI values
    :param threshold: minimum NDVI value required to send a pixel into MAD
    :return: veg_mask: boolean numpy array. 1 -> do not use for MAD. 0 -> usable for MAD.
    """
    print("Generating vegetation mask... ")
    rows, cols = ndvi.shape
    empty = np.zeros([rows, cols])
    ones = empty+1

    # where ndvi >= threshold, veg_mask = True. Elsewhere, veg_mask = 0.
    veg_mask = np.where(ndvi >= threshold, ones, empty)
    # require that at least 10000 pixels are usable
    iteration = 0
    while np.sum(veg_mask) < 10000:
        threshold -= 0.10
        print("Couldn't find enough usable pixels for veg mask. Reducing NDVI threshold to %3.2f " % threshold)
        veg_mask = np.where(ndvi>=threshold, ones, empty)
        iteration += 1
        if iteration > 20:
            print("Couldn't find enough pixels with sufficient NDVI")
            quit()
    # mask = ~veg_mask.astype(bool)
    return veg_mask


def calc_img_stats(img_tif):
    """
    Calculate some basic statistics of the image.
    :param img_tif:
    :return:
    """
    # Load .tif image
    with rasterio.open(img_tif) as src:
        array = src.read()
    stats = []
    for band in array:
        stats.append({
            'min': band.min(),
            'mean': band.mean(),
            'median': np.median(band),
            'max': band.max(),
            'var': np.var(band)
        })
    return stats


def check_for_clouds(dir=".", tolerance=0.5):
    """
    Check the metadata file for the input image to see if it good enough to run through iMad.

    :param dir: (string) directory containing JSON files with image metadata
    :return: usable: (boolean) True for a usable image, False otherwise.
    """
    cloud = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith("metadata.json")]:
            with open(os.path.join(dirpath, filename)) as file:
                metadata = json.load(file)
                cloud.append(metadata["properties"]["cloud_cover"])
    if not cloud:
        warnings.warn("Couldn't find the Planet metadata.json file. Assuming cloud cover is at an acceptable level. ")
        return True
    cloud_frac = sum(cloud)/len(cloud)
    if cloud_frac <= tolerance and cloud_frac >= 0.0:
        return True
    else:
        return False


def register_image2(target_img, reference_img, no_data_val=0.0, outdir=None):
    """
    Using AROSICS, perform co-registration. Generates and saves a CSV of the tie-points and their shift vectors.
    Also creates a visualization of how the co-registered image has shifted.
    :param target_img: image to have geospatial referencing updated
    :param reference_img: image with target geospatial referencing
    :param no_data_val: (float) no-data value for the target and reference images
    :return warped_out: filepath of coregistered version of target_image
    """
    # this function uses arosics instead of arcpy to be open source. Results differ slightly but look comparable
    if outdir:
        direc = outdir
    else:
        direc = os.path.split(target_img)[0]

    if target_img.endswith('.tiff'):
        warped_out = os.path.join(direc, os.path.split(target_img[:-5])[1] + "_aligned.tif")
    elif target_img.endswith('.tif'):
        warped_out = os.path.join(direc, os.path.split(target_img[:-4])[1] + "_aligned.tif")
    else:
        print("Input image must be a GeoTiff. Exiting.")
        return
    registered_img = arosics.COREG_LOCAL(reference_img, target_img, 300, path_out=warped_out,
                                         nodata=(no_data_val, no_data_val), fmt_out="GTiff",
                                         projectDir=os.path.split(target_img)[0])
    tie_points = registered_img.CoRegPoints_table
    coreg_csv_name = os.path.split(target_img)[1] + "coreg_points.csv"
    coreg_visual_name = os.path.split(target_img)[1] + "coreg_visual.png"
    tie_points.to_csv(os.path.join(direc, coreg_csv_name), sep=",", index=False)
    registered_img.view_CoRegPoints(savefigPath=os.path.join(direc, coreg_visual_name))
    registered_img.correct_shifts()
    return warped_out


def trim_to_image(input_big, input_target, allow_downsample=True, outdir=None):
    """
    Trim an image to the dimensions of another (in same projection). Trimmed image is output without overwriting original.
    The larger image, input_big, is trimmed to the dimensions of input_target.

    :param input_big: (string) location of a large file to be cropped to smaller dimensions
    :param input_target: (string) location of a smaller file with target dimensions. May be downsampled if it is higher resolution that input_big.
    :param allow_downsample: (bool) True if images are different resolutions.
    :return conductedDownsample: (bool) True if image was downsampled
    :return downsampled_target: (str) filepath of the downsampled version of input_target
    :return outfile: (str) filepath of the cropped version of input_big
    :return input_target: (tuple) first, whether or not downsample was conducted (bool)
        second, path to the downsampled file (str)
        third, filepath of the trimmed image
        fourth,
    """

    # open the big file, grab its resolution
    data_big = gdal.Open(input_big, GA_ReadOnly)
    geoTransform = data_big.GetGeoTransform()
    xres_big = geoTransform[1]
    yres_big = geoTransform[5]

    # open the file with your target dimensions, calculate bounding box
    data_target = gdal.Open(input_target, GA_ReadOnly)
    geoTransform = data_target.GetGeoTransform()
    xres_target = geoTransform[1]
    yres_target = geoTransform[5]
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data_target.RasterXSize
    miny = maxy + geoTransform[5] * data_target.RasterYSize

    # name the output file for the cropped big image. take path from input_target or specified output directory
    if outdir:
        dir_target = outdir
    else:
        dir_target = os.path.split(input_target)[0]
    if input_big.endswith('.tiff'):
        outfile = os.path.join(dir_target, os.path.split(input_big)[1][:-5] + "_trimmed.tif")
    else:
        outfile = os.path.join(dir_target, os.path.split(input_big)[1][:-4] + "_trimmed.tif")

    # downsample the target image if it is higher resolution than the big image
    downsampled_target = None
    if allow_downsample:
        if (abs(xres_big) != abs(xres_target)) or (abs(yres_big) != abs(yres_target)):
            print("Downsampling target image...")
            # generate name for downsampled version of image
            if input_target.lower().endswith('tiff'):
                downsampled_target = os.path.join(dir_target, os.path.split(input_target)[1][:-5] + "_downsample.tif")
            elif input_target.lower().endswith('tif'):
                downsampled_target = os.path.join(dir_target, os.path.split(input_target)[1][:-4] + "_downsample.tif")
            else:
                print("It looks like the file extension was " + input_target[:-4])
                print("Was expecting .tif or .tiff. Try again.")
                quit()
            # perform downsampling
            # TODO: swap this command to rasterio
            """
            scale_factor_x = xres_big / xres_target  # will be a value larger than 1
            scale_factor_y = yres_big / yres_target

            with rasterio.open(input_big) as ref_ds:
                ref_transform = ref_ds.transform
                target_transform = ref_transform * rasterio.Affine(ref_transform.a*scale_factor_x,
                                                                   ref_transform.b, ref_transform.c,
                                                                   ref_transform.d, ref_transform.d*scale_factor_y,
                                                                   ref_transform.f)
                ref_arr = ref_ds.read()
                kwargs = ref_ds.meta
                kwargs['transform'] = target_transform
                with rasterio.open(downsampled_target, 'w', **kwargs) as dst:
                    for i, band in enumerate(ref_arr, 1):
                        dest = np.zeros_like(band)
                        reproject(band, dest, src_transform=ref_transform, src_crs=ref_ds.crs,
                                  dst_transform=target_transform, dst_crs=ref_ds.crs, resampling=Resampling.bilinear)
                        dst.write(dest, indexes=i)
            """
            call('gdalwarp -tr ' + str(abs(xres_big)) + ' ' + str(abs(yres_big)) + ' -r average ' + input_target + ' '
                 + downsampled_target, shell=True)
            # reset data, grab new extent
            data_target = gdal.Open(downsampled_target, GA_ReadOnly)
            geoTransform = data_target.GetGeoTransform()
            minx = geoTransform[0]
            maxy = geoTransform[3]
            maxx = minx + geoTransform[1] * data_target.RasterXSize
            miny = maxy + geoTransform[5] * data_target.RasterYSize
            # set a flag to show that downsampling occurred
            conductedDownsample = True
        else:
            conductedDownsample = False
    else:
        downsampled_target = input_target
        conductedDownsample = False

    print("Cropping...")
    call('gdal_translate -projwin ' + ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) + ' -a_nodata 0.0' +
         ' -of GTiff ' + input_big + ' ' + outfile, shell=True)
    return conductedDownsample, downsampled_target, outfile, input_target


def perform_downsample(src_image, target_res, outfile="downsample.tif"):
    """
    Quick and dirty wrapper to call CL gdalwarp for downsampling
    :param src_image:
    :param target_res:
    :param outfile:
    :return:
    """
    # TODO: change this whole function to rasterio
    # open source image, grab resolution
    src_DS = gdal.Open(src_image, GA_ReadOnly)
    geotransform = src_DS.GetGeoTransform()
    src_xres = geotransform[1]
    src_yres = geotransform[5]
    call('gdalwarp -tr ' + str(abs(target_res)) + ' ' + str(abs(target_res)) + ' -r average ' + src_image + ' '
         + outfile, shell=True)
    return outfile


def set_no_data(planet_img, cropped_img, outfile="out.tif", outdir=None,  src_nodata=0.0, dst_nodata=0.0, save_mask=False,
                datatype_out=gdal.GDT_UInt16):
    """
    Takes in two images- one with no-data values around the perimeter, one without. Applies matching no-data values to the second image and saves as new output.

    :param planet_img: (string) filepath of Planet image containing some no-data values around the perimeter. Also supports numpy array.
    :param cropped_img: (string) Must have the same bounding box as planet_img, but can have data everywhere. No-data values from planet_img will be added to cropped_img (no overwrite)
    :param outfile: (string) name of output file. DO NOT ADD LOCATION. File location will be added later.
    :param outdir: (str) directory to save output file
    :param src_nodata: (float) the no-data value in the incoming image.
    :param dst_nodata: (float) the no-data value in the output image.
    :param save_mask: (bool) save the mask as a GeoTiff
    :param datatype_out: (gdal datatype) desired filetype for the output file
    :return planet image: unchanged from input
    :return outfile: (string) filepath of image with no-data values applied
    """

    # array version:
    if (type(planet_img) == np.ndarray) or (type(planet_img) == np.ma.core.MaskedArray):
        # this is the case where the inputs are arrays
        # this is essentially just for the vegetation mask (single band) at the moment
        gdal.AllRegister()
        img_nodata_source = planet_img      # the file with some no-data pixels that will be copied over.
        img_nodata_target = gdal.Open(cropped_img, gdal.GA_Update)  # file which will have no-data pixels added.
        rows, cols = img_nodata_source.shape
        cols2 = img_nodata_target.RasterXSize
        rows2 = img_nodata_target.RasterYSize
        bands = img_nodata_target.RasterCount
        if (cols != cols2) or (rows != rows2):
            print("size mismatch. use images with same dimensions.")
            return

        target_arr = np.zeros([rows, cols, bands])     # intializing. will become the output image
        target_arr = ma.masked_array(target_arr, fill_value=0.0)
        src_nodata_arr = np.zeros([rows, cols])  # Fill this array with the source no-data value
        if src_nodata != 0.0:
            src_nodata_arr.fill(src_nodata)
        dst_nodata_arr = np.zeros([rows, cols])  # Build array full of the the target no-data value.
        if dst_nodata != 0.0:
            dst_nodata_arr.fill(dst_nodata)

        print("Building no-data array...")
        no_data_mask = (img_nodata_source == src_nodata)  # Pixels with data marked as 0.0. Pixels with no data marked as 1.0.
        no_data_mask = no_data_mask.astype(bool)

        print("Applying no-data array...")
        for band in range(bands):
            target_arr[:, :, band] = np.array(img_nodata_target.GetRasterBand(band + 1).ReadAsArray())
            target_arr[:, :, band] = np.where(no_data_mask[:, :], dst_nodata_arr[:, :], target_arr[:, :, band])
            target_arr[:, :, band] = ma.array(target_arr[:, :, band], mask=no_data_mask, fill_value=dst_nodata)

        print("Writing and saving...")
        if outdir:
            dir_target = outdir
        else:
            dir_target = os.path.split(cropped_img)[0]
        outfile = os.path.join(dir_target, outfile)
        target_DS = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows, bands, datatype_out)
        for band in range(bands):
            target_DS.GetRasterBand(band+1).WriteArray(target_arr[:,:,band])
        target_DS.SetGeoTransform(img_nodata_target.GetGeoTransform())
        target_DS.SetProjection(img_nodata_target.GetProjection())
        target_DS.FlushCache()
        target_DS = None
        img_nodata_target = None
        print("Done setting no-data!")
        return img_nodata_source, outfile, no_data_mask

    # disk-based file version
    else:
        # this is the case where the inputs are files on disk
        gdal.AllRegister()
        planet = gdal.Open(planet_img)
        target = gdal.Open(cropped_img, gdal.GA_Update)
        bands = target.RasterCount
        cols = planet.RasterXSize
        rows = planet.RasterYSize
        cols2 = target.RasterXSize
        rows2 = target.RasterYSize

        if outdir:
            dir_target = outdir
        else:
            dir_target = os.path.split(cropped_img)[0]
        outfile = os.path.join(dir_target, outfile)
        target_DS = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows, bands, datatype_out)

        if (cols != cols2) or (rows != rows2):
            print("size mismatch. use images with same dimensions.")
            return

        planet_arr = np.zeros([rows, cols, bands])
        target_arr = np.zeros([rows, cols, bands])
        target_arr = ma.masked_array(target_arr, fill_value=0.0)
        no_data_mask = np.zeros([rows, cols, bands])
        src_nodata_arr = np.zeros([rows, cols])      # Fill this array with the no-data value
        if src_nodata != 0.0:
            src_nodata_arr.fill(src_nodata)
        dst_nodata_arr = np.zeros([rows, cols])     # Build array full of the the target no-data value.
        if dst_nodata != 0.0:
            dst_nodata_arr.fill(dst_nodata)

        print("Building no-data array...")
        for band in range(bands):
            planet_arr[:, :, band] = np.array(planet.GetRasterBand(band+1).ReadAsArray())
            no_data_mask[:, :, band] = planet_arr[:, :, band] == src_nodata
            # Pixels with data marked as 0.0. Pixels with no data marked as 1.0. Could flip this by using: planet_arr[:,:,band] != src_nodata
            # TODO flip this so that it makes more sense...

        new_mask = no_data_mask.sum(axis=2)
        new_mask = new_mask != 0     # no-data where new_mask = True, i.e. setup for masked arrays

        print("Applying no-data array...")
        for band in range(bands):
            target_arr[:, :, band] = np.array(target.GetRasterBand(band+1).ReadAsArray())
            target_arr[:, :, band] = np.where(no_data_mask[:, :, band], dst_nodata_arr[:, :], target_arr[:, :, band])
            target_arr[:, :, band] = ma.array(target_arr[:, :, band], mask=new_mask, fill_value=dst_nodata)

        print("Writing and saving...")
        for band in range(bands):
            target_DS.GetRasterBand(band+1).WriteArray(target_arr[:, :, band])

        if save_mask:
            print("Saving no-data mask...")
            mask_outfile = os.path.join(dir_target, "no_data_mask.tif")
            mask_DS = gdal.GetDriverByName('GTiff').Create(mask_outfile, cols, rows, 1, gdal.GDT_Byte)
            mask_DS.GetRasterBand(1).WriteArray(new_mask)
            mask_DS.SetGeoTransform(planet.GetGeoTransform())
            mask_DS.SetProjection(planet.GetProjection())
            mask_DS.FlushCache()
            mask_DS = None

        target_DS.SetGeoTransform(planet.GetGeoTransform())
        target_DS.SetProjection(planet.GetProjection())
        target_DS.FlushCache()
        target_DS = None
        target = None
        planet = None
        print("Done setting no-data!")

        return planet_img, outfile, new_mask


def scale_image(img_path, nodata, outdir=None, datatype_out=gdal.GDT_UInt16):
    """
    Scale the range of values for an image from 0-100.
    First, sets the 2% lowest values to the 2% mark and the 2% highest values to the 98% mark.
        # ALTERNATE VERSION: replaces the brightest values with nodata
    Then takes the output, sets the lowest value to 0 and the highest to 100.
    Writes a float.
    :param img_path:
    :param nodata: no-data value in input image
    :param datatype_out: GDAL data type to save outputs, e.g. gdal.GDT_Float32.
    :return:
    """
    gdal.AllRegister()
    img_src = gdal.Open(img_path)
    cols = img_src.RasterXSize
    rows = img_src.RasterYSize
    bands = img_src.RasterCount
    count_thresh = 0.02

    #bands_array = np.zeros([rows, cols, bands])    # maybe restore later
    bands_array = []
    nodata_array = np.zeros([rows, cols])    # will store locations where a pixel is no-data. If no-data, then 1.0.
    for band in range(bands):
        temp_array = np.array(img_src.GetRasterBand(band+1).ReadAsArray()).astype(np.float)
        size = temp_array.size
        upper_cutoff = np.sort(temp_array, axis=None)[int(size*(1-count_thresh))]
        #lower_cutoff = np.sort(temp_array, axis=None)[int(size*count_thresh)]
        #np.place(temp_array, temp_array>=upper_cutoff, upper_cutoff)
        np.place(temp_array, temp_array >= upper_cutoff, nodata)
        #np.place(temp_array, temp_array<=lower_cutoff, lower_cutoff)
        #bands_array[:,:,band] = temp_array/upper_cutoff*100.
        #bands_array[:, :, band] = temp_array / temp_array.max() * 100.     # maybe restore later
        bands_array.append(temp_array / temp_array.max()*100.)
        nodata_array = np.where(temp_array == nodata, 1.0, nodata_array)

    # if a pixel is no-data in one band, set to no-data in all
    for band in range(bands):
        # np.place(bands_array[:,:,band], nodata_array==1.0, nodata)    # maybe restore later
        np.place(bands_array[band], nodata_array==1.0, nodata)

    if outdir:
        dir_target = outdir
    else:
        dir_target = os.path.split(img_path)[0]
    src_filename = os.path.split(img_path)[1]
    outfile = os.path.join(dir_target, src_filename[:-4] + "_scaled.tif")
    target_DS = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows, bands, datatype_out)
    for band in range(bands):
        target_DS.GetRasterBand(band + 1).WriteArray(bands_array[band])
    target_DS.SetGeoTransform(img_src.GetGeoTransform())
    target_DS.SetProjection(img_src.GetProjection())
    target_DS.FlushCache()
    target_DS = None
    img_src = None
    return


def img_to_array(img_path):
    """
    Returns a list of arrays for a multiband image. Returns a single numpy array for a single band image.
    :param img_path:
    :return:
    """
    img_src = gdal.Open(img_path)
    bands = img_src.RasterCount
    if bands > 1:
        bands_array = []
        for band in range(bands):
            bands_array.append(np.array(img_src.GetRasterBand(band+1).ReadAsArray()).astype(np.float))
    else:
        bands_array = np.array(img_src.GetRasterBand(1).ReadAsArray()).astype(np.float)
    img_src = None
    return bands_array


def array_to_img(array, outpath, ref_img, datatype_out=gdal.GDT_UInt16):
    """
    Life would be easier if this was a list of arrays for each band, but let's assume it is a 3D numpy array
    :param array: (numpy array) 2D or 3D with pixel values
    :param outpath: (string) filepath for image to be saved
    :param ref_img: filepath to an image where function can grab geotransform and projection. Easier than inputting.
    :param datatype_out: GDAL data type to save outputs, e.g. gdal.GDT_Float32.
    :return:
    """

    # check if it is a 2D or a 3D array first
    shape = array.shape
    rows = shape[0]
    cols = shape[1]
    if len(shape) >= 3:
        bands = shape[2]
    else:
        bands = 1

    target_DS = gdal.GetDriverByName('GTiff').Create(outpath, cols, rows, bands, datatype_out)
    if bands > 1:
        for band in range(bands):
            target_DS.GetRasterBand(band + 1).WriteArray(array[:, :, band])
    else:
        target_DS.GetRasterBand(1).WriteArray(array)

    # write it to memory
    # get info from the reference image
    ref_src = gdal.Open(ref_img)
    target_DS.SetGeoTransform(ref_src.GetGeoTransform())
    target_DS.SetProjection(ref_src.GetProjection())
    target_DS.FlushCache()
    target_DS = None
    ref_src = None
    return


# THIS FUNCTION IS NO LONGER IN USE.
def diff_images(img1_path, img2_path, outfile=False):
    """
    Goal: look for pixels that are substantially different between the Landsat composite and final output
        Principle is that some changes are significant and some are unimportant, but all will be all obvious
        Start by finding the changes, then go on to sort them later
    Inputs: paths to the two images to be diffed.
    Assumes bands are in the same order
    :param img1_path:
    :param img2_path:
    :param outpath: full desired pathname of output image
    :return:
    """
    # Step 1: read in images
    img1_src = gdal.Open(img1_path)
    img2_src = gdal.Open(img2_path)
    rows1 = img1_src.RasterXSize
    rows2 = img2_src.RasterXSize
    cols1 = img1_src.RasterYSize
    cols2 = img2_src.RasterYSize
    bands1 = img1_src.RasterCount
    bands2 = img2_src.RasterCount
    bands = min([bands1, bands2])
    # Step 2: make sure the images are the same dimensions. If not, upsample to the finer resolution.
    if (rows1 != rows2) or (cols1 != cols2):
        print("Images have different dimensions. Attempting to fix that... ")
        if (rows1 > rows2) or (cols1 > cols2):
            # first arg is the one that gets downsampled
            # returns filepath
            img1_path = trim_to_image(img1_path, img2_path, allow_downsample=True)[1]

        else:
            img2_path = trim_to_image(img2_path, img1_path, allow_downsample=True)[1]
    # Step 3: load images into arrays. Loads each band into a numpy array. Arrays are stored in a list.
    img1_array = img_to_array(img1_path)    # filetype: list
    img2_array = img_to_array(img2_path)    # filetype: list
    # Step 4: take differences, band by band
    img_delta = []
    for band in range(bands):
        img_delta.append((img1_array[band] - img2_array[band]))     # filetype: list
    # Step 5: write out as a new array
    if outfile == False:
        outpath = os.path.split(img1_path)[0]
        outfile = outpath + '\\' + "diffed_img.tif"
    target_DS = gdal.GetDriverByName('GTiff').Create(outfile, min([rows1, rows2]), min([cols1, cols2]), bands, gdal.GDT_Float32)
    for band in range(bands):
        target_DS.GetRasterBand(band + 1).WriteArray(img_delta[band])
    target_DS.SetGeoTransform(img1_src.GetGeoTransform())
    target_DS.SetProjection(img1_src.GetProjection())
    target_DS.FlushCache()
    target_DS = None
    img1_src = None
    img2_src = None
    return "Diff image saved at " + outfile


def main(image_ref, image_reg_ref, image_targ, allowDownsample, allowRegistration, view_radcal_fits, src_nodataval=0.0,
         dst_nodataval=0.0, udm=None, ndvi_thresh=0.0, nochange_thresh=0.95, outdir=None, datatype_out=gdal.GDT_UInt16):
    """
    Purpose: radiometrically calibrate a target image to a reference image.
    Optionally update the georeferencing in the target image.
    :param image_ref: (str) filepath of radiometry reference image
    :param image_reg_ref: (str) filepath of geolocation reference image
    :param image_targ: (str) filepath of target image
    :param allowDownsample: (bool) whether the target image needs to be downsampled. True if reference and target are different resolutions.
    :param allowRegistration: (bool) whether the target image needs registration
    :param view_radcal_fits: (bool) whether the radcal fits should be displayed
    :param src_nodataval: (float) no-data value in the input images
    :param dst_nodataval: (float) no-data value to be applied to the output images
    :param udm: (list, tuple, or string) filepath of Unusable Data Mask(s) which will be applied to the final image
    :param outdir: (str) folder to save all outputs. Not yet functional.
    :param datatype_out: GDAL data type to save outputs, e.g. gdal.GDT_Float32. Not yet functional.
    :return: outpath_final: (str) path to final output image.
    """
    start = time.time()
    # Step 0: check image metadata to see if it has an acceptable level of cloud.
    image_targ_dir = os.path.split(image_targ)[0]
    below_cloud_thresh = check_for_clouds(image_targ_dir, tolerance=0.5)
    if below_cloud_thresh == False:
        print("Image is above cloud threshold. Either use a different image or increase threshold.")
        cloudOverride = input("Override cloud threshold? y/n: ")
        assert isinstance(cloudOverride, str)
        if cloudOverride == "y":
            print("Cloud threshold overridden. Proceeding with processing. ")
        elif cloudOverride == "n":
            print("Exiting. ")
            return
        else:
            print("Must choose y or n. Try again.")
            return
    # create output directory if it does not yet exist
    if outdir:
        if not os.path.isdir(outdir):
            print("Making the output directory " + outdir)
            os.mkdir(outdir)
    else:
        print("Not making an output directory. ")

    # Step 0.5: check to make sure all input images are in the same projection
    # TODO: switch from gdal to rasterio to reduce image lock issues?
    rad_ref_DS = gdal.Open(image_ref)
    reg_ref_DS = gdal.Open(image_reg_ref)
    target_DS = gdal.Open(image_targ)
    rad_ref_prj = rad_ref_DS.GetProjection()
    reg_ref_prj = reg_ref_DS.GetProjection()
    target_prj = target_DS.GetProjection()
    rad_ref_srs = osr.SpatialReference(wkt=rad_ref_prj).GetAttrValue('authority', 1)
    reg_ref_srs = osr.SpatialReference(wkt=reg_ref_prj).GetAttrValue('authority', 1)
    target_srs = osr.SpatialReference(wkt=target_prj).GetAttrValue('authority', 1)
    if (rad_ref_srs == reg_ref_srs == target_srs):
        print("All projections are the same. Good. ")
    else:
        warnings.warn("Oh no! The projections are different! Attemping to fix that. ")
        print("Projection of the radiometric reference: " + str('EPSG:'+rad_ref_srs))
        print("Projection of the registration reference: " + str('EPSG:'+reg_ref_srs))
        print("Projection of the target image: " + str('EPSG:'+target_srs))
        # TODO: rewrite these using rasterio?
        if rad_ref_srs != reg_ref_srs:
            if outdir:
                dir_target = outdir
            else:
                dir_target = os.path.split(image_reg_ref)[0]
            reproj_reg_ref = os.path.join(dir_target, "reg_ref_reprojected.tif")
            call('gdalwarp -t_srs EPSG:' + rad_ref_srs + ' ' + image_reg_ref + ' ' + reproj_reg_ref)
            image_reg_ref = reproj_reg_ref
        if rad_ref_srs != target_srs:
            if outdir:
                dir_target = outdir
            else:
                dir_target = os.path.split(image_targ)[0]
            reproj_target = os.path.join(dir_target, image_targ[:-4] + "_reproj.tif")
            call('gdalwarp -overwrite -s_srs EPSG:' + target_srs + ' -t_srs EPSG:' + rad_ref_srs + ' ' + image_targ + ' '
                 + reproj_target)
            image_targ = reproj_target
    # while the files are open, also grab resolution and number of bands
    bands_ref = rad_ref_DS.RasterCount
    bands_targ = target_DS.RasterCount
    res_ref_x = abs(rad_ref_DS.GetGeoTransform()[1])
    res_ref_y = abs(rad_ref_DS.GetGeoTransform()[5])
    res_targ_x = abs(target_DS.GetGeoTransform()[1])
    res_targ_y = abs(target_DS.GetGeoTransform()[5])
    # close files
    del rad_ref_DS
    del reg_ref_DS
    del target_DS
    # Step 1: grab a reference image snip that will be used to align the target image.
    trim_out = trim_to_image(image_reg_ref, image_targ, allow_downsample=False, outdir=outdir)

    # Step 2: align the Planet image to that snip (if permitted).
    if allowRegistration:
        image2_aligned = register_image2(image_targ, trim_out[2], outdir=outdir)
        # out is a full-resolution image. 2nd arg is cropped reference image.
    else:
        image2_aligned = image_targ  # keep in mind that this has a path attached

    # Step 3: create a new Landsat snip to match aligned Planet image.
    trim_out = trim_to_image(image_ref, image2_aligned, allowDownsample, outdir=outdir)
    # note on next line: downsampled_img may or may not exist, depending if downsampling occurred.
    # trim_out[1:3] are all strings which include file location.
    downsampleFlag, downsampled_img, cropped_img, original_planet_img = trim_out  # downsampled_img has good alignment
    if downsampleFlag == False:
        downsampled_img = original_planet_img

    # Step 4: generate a veg mask at Planet resolution, downsample and apply, but save full res for later
    VI_calc_out_full_res = calc_VIs(image2_aligned, nodataval=dst_nodataval)
    ndvi_full_res = VI_calc_out_full_res[0]
    veg_mask_full_res_arr = vegetation_mask(ndvi_full_res, threshold=ndvi_thresh)
    if outdir:
        dir_veg_mask = outdir
    else:
        dir_veg_mask = image_targ_dir
    veg_mask_full_res_img = os.path.join(dir_veg_mask, "veg_mask_full_res.tif")
    array_to_img(veg_mask_full_res_arr, veg_mask_full_res_img, image2_aligned)
    # to make this a neat duplicate of the previous method, set_no_data expects veg mask to be an array, not a file
    # resolution in the following call assumes square pixels
    veg_mask_downsamp = perform_downsample(veg_mask_full_res_img, res_ref_x,
                                           outfile=os.path.join(dir_veg_mask, "veg_mask_downsample.tif"))
    veg_mask_downsamp_arr = img_to_array(veg_mask_downsamp)
    planet_downsamp_vegmasked = set_no_data(veg_mask_downsamp_arr, downsampled_img, outfile="planet_with_veg_mask.tif",
                                            outdir=outdir, src_nodata=src_nodataval, dst_nodata=dst_nodataval)[1]

    # Step 5: add no data values into Landsat image so that it will play nice with iMad.py and radcal.py
    landsat_nodata = os.path.split(cropped_img)[1][:-4] + "_nodata.tif"
    no_data_out_novegmask = set_no_data(downsampled_img, cropped_img, landsat_nodata, outdir=outdir,
                                        dst_nodata=dst_nodataval)  # need version that hasn't been masked too.
    no_data_out_vegmask = set_no_data(planet_downsamp_vegmasked, cropped_img, landsat_nodata, outdir=outdir,
                                      dst_nodata=dst_nodataval)

    # at this point, we have a Planet image at Landsat resolution and a Landsat image with Planet no-data values.
    # everything we need for MAD + radcal to run
    # planet_img = downsampled version of planet at this point
    planet_img_novegmask = no_data_out_novegmask[0]  # note that planet_img and landsat_img include paths
    landsat_img_novegmask = no_data_out_novegmask[1]
    planet_img_vegmask = no_data_out_vegmask[0]  # note that planet_img and landsat_img include paths
    landsat_img_vegmask = no_data_out_vegmask[1]

    end = time.time()
    print("===============================")
    print("Time elapsed: " + str(int(end-start)) + " seconds")
    print("===============================")
    print("Beginning MAD...")
    outfile_MAD = os.path.split(planet_img_novegmask)[1][:-4] + "_MAD.tif"
    outfile_RAD = os.path.split(planet_img_novegmask)[1][:-4] + "_RAD.tif"
    outfile_final = os.path.split(image_targ)[1][:-4] + "_FINAL.tif"
    run_MAD(planet_img_vegmask, landsat_img_vegmask, outfile_MAD, outdir=outdir)
    # image2_aligned is the full resolution Planet scene
    print("Beginning radcal...")
    normalized_fsoutfile = run_radcal(planet_img_novegmask, landsat_img_novegmask, outfile_RAD, outfile_MAD,
                                      image2_aligned, view_plots=view_radcal_fits, outdir=outdir,
                                      nochange_thresh=nochange_thresh)
    # Step 6: re-apply no-data values to the radiometrically corrected full-resolution planet image
    # necessary since radcal applies the correction to the no-data areas
    # TODO add a projection-checking step to the UDM application.
    if udm:
        # Documentation on Planet UDM doesn't seem to agree with actual values:
        # https://assets.planet.com/docs/Planet_Combined_Imagery_Product_Specs_letter_screen.pdf
        if (type(udm) == list) or (type(udm) == tuple):
            for raster in udm:
                # expects each item in the list udm to be a file PATH, not just filename. Updates files in place.
                process_udm(raster, src_nodata=1.0)
            merged_udm = os.path.join(os.path.split(udm[0])[0], "merged_udm.tif")
            udm_merger(udm, merged_udm)
            udm_as_arr = img_to_array(merged_udm)
            final_images = set_no_data(udm_as_arr, normalized_fsoutfile, outfile_final, src_nodata=1.0,
                                       dst_nodata=dst_nodataval, save_mask=True)
        elif type(udm) == str:
            # assume that if we got a string, it is a single default UDM filepath that still must be processed.
            process_udm(udm, src_nodata=1.0)
            udm_as_arr = img_to_array(udm)
            final_images = set_no_data(udm_as_arr, normalized_fsoutfile, outfile_final, src_nodata=1.0,
                                       dst_nodata=dst_nodataval, save_mask=True)
        else:
            warnings.warn("Got an unexpected type {} for 'udm'. Proceeding without using udm. ".format(type(udm)))
    else:
        final_images = set_no_data(image2_aligned, normalized_fsoutfile, outfile_final, dst_nodata=dst_nodataval,
                                   save_mask=True, outdir=outdir)

    end = time.time()
    print("===============================")
    print("All done!")
    print("Final image at: ")
    print(outfile_final)
    print("Total time elapsed: " + str(int(end-start)) + " seconds")
    if outdir:
        dir_out_final = outdir
    else:
        dir_out_final = os.path.split(image_ref)[0]
    outpath_final = os.path.join(dir_out_final, outfile_final)

    return outpath_final


if __name__ == '__main__':
    image1 = input("Location of image with reference radiometry: ")
    assert isinstance(image1, str)
    image_reg_ref = input("Location of image with desired georeferencing: ")
    assert isinstance(image_reg_ref, str)
    image2 = input("Location of target image to be radiometrically normalized : ")
    assert isinstance(image_reg_ref, str)
    output_dir = input("Location of directory to save outputs? (will be created if does not exist): ")
    assert isinstance(output_dir, str)
    allowDownsample = input("Allow target image to be downsampled if needed? y/n: ")
    assert isinstance(allowDownsample, str)
    if allowDownsample == "y":
        allowDownsample = True
    elif allowDownsample == "n":
        allowDownsample = False
    else:
        print("Must choose y or n. Try again.")
        quit()
    udms = input("Apply a usable data mask? Filepath if yes, otherwise n: ")
    assert isinstance(udms, str)
    if udms == "n":
        udms = False
    allowRegistration = input("Allow target image to be re-registered if needed? y/n: ")
    assert isinstance(allowRegistration, str)
    if allowRegistration == "y":
        allowRegistration = True
    elif allowRegistration == "n":
        allowRegistration = False
    else:
        print("Must choose y or n. Try again.")
        quit()
    view_radcal_fits = input("View radiometric calibration fit? y/n: ")
    assert isinstance(view_radcal_fits, str)
    if view_radcal_fits == "y":
        view_radcal_fits = True
    elif view_radcal_fits == "n":
        view_radcal_fits = False
    else:
        print("Must choose y or n. Try again.")
        quit()
    main(image1, image_reg_ref, image2, allowDownsample, allowRegistration, view_radcal_fits, udm=udms,
         outdir=output_dir)
