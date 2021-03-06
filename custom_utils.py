
# Usage: from command line in appropriate folder;
# >> python custom_utils.py
# Will prompt for names and file locations of images.


import six
from builtins import input
import warnings
import json
import math
from rasterio.warp import reproject, Resampling
import rasterio.crs
import numpy as np
import numpy.ma as ma
import gdal, osr
import os
import rasterio
import rasterio.mask
import rasterio.features
import fiona
from geopandas import GeoDataFrame
import pandas as pd
from shapely.geometry import shape
from gdalconst import GA_ReadOnly
from subprocess import call
import time
from iMad import run_MAD
from radcal import run_radcal

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def process_udm(udm_path, src_nodata=1.0):
    """
    Takes a default Planet Unusable Data Mask and compresses it to one bit, where everything originally = 1.0 or greater
        becomes nodata. Currently still saves as an 8bit file though.
        NB: this edits the UDM in place
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


def save_veg_indices(rasterpath, outdir=None, save_ndvi=True, save_evi=False):
    """
    Generates two new GeoTiffs: the NDVI and EVI for the input raster.

    :param rasterpath: path to the raster
    :param outdir: (str) directory to save outputs
    :param save_ndvi: (bool) whether or not to save the NDVI to disk
    :param save_evi: (bool) whether or not to save the EVI to disk
    :return:
    """
    # grab the input raster's meta information
    with rasterio.open(rasterpath, 'r') as src:
        meta = src.meta

    ndvi, evi, img_src, cols, rows = calc_VIs(rasterpath, meta['nodata'])

    # grab the directory and original file name, then build new filenames
    dir_src, filename_src = os.path.split(rasterpath)[0], os.path.split(rasterpath)[1]
    if outdir:
        filename_ndvi_src = os.path.join(outdir, filename_src[:-4] + '-NDVI.tif')
    else:
        filename_ndvi_src = os.path.join(dir_src, filename_src[:-4] + '-NDVI.tif')
    filename_evi_src = os.path.join(dir_src, filename_src[:-4] + '-EVI.tif')

    # update meta for vegetation indices
    vi_meta = meta.copy()
    vi_meta['count'] = 1
    vi_meta['dtype'] = 'float32'
    if save_ndvi:
        with rasterio.open(filename_ndvi_src, 'w', **vi_meta) as dst:
            dst.write_band(1, ndvi)
    if save_evi:
        with rasterio.open(filename_evi_src, 'w', **vi_meta) as dst:
            dst.write_band(1, evi)
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


def register_image2(target_img, reference_img, no_data_val=0.0, outdir=None):
    """
    Using AROSICS, perform co-registration. Generates and saves a CSV of the tie-points and their shift vectors.
    Also creates a visualization of how the co-registered image has shifted.
    :param target_img: image to have geospatial referencing updated
    :param reference_img: image with target geospatial referencing
    :param no_data_val: (float) no-data value for the target and reference images
    :return warped_out: filepath of coregistered version of target_image
    """
    import arosics
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
                                         projectDir=outdir, r_b4match=4, s_b4match=4,
                                         calc_corners=False, align_grids=False)
    coreg_csv_name = os.path.split(target_img)[1][:-4] + "_coreg_points.csv"
    try:
        tie_points = registered_img.CoRegPoints_table
        tie_points.to_csv(os.path.join(direc, coreg_csv_name), sep=",", index=False)
    except ImportError:
        pass
    registered_img.correct_shifts()  # move this to proceed calculating the points table?
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
    # TODO: separate this into two functions: one to trim, one to downsample
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
            print("Downsampling target image " + os.path.split(input_target)[1] + "...")
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
            scale_factor_x = xres_big / xres_target  # will be a value larger than 1
            scale_factor_y = yres_big / yres_target

            with rasterio.open(input_target) as target_ds:
                target_transform = target_ds.transform
                target_arr = target_ds.read()
                kwargs = target_ds.meta
                target_crs = target_ds.crs
                new_transform = rasterio.Affine(target_transform.a*scale_factor_x,
                                                target_transform.b, target_transform.c,
                                                target_transform.d, target_transform.e*scale_factor_y,
                                                target_transform.f)
            kwargs['transform'] = new_transform
            kwargs['height'] = math.ceil(target_arr[0].shape[0]/scale_factor_x)
            kwargs['width'] = math.ceil(target_arr[0].shape[1]/scale_factor_y)
            with rasterio.open(downsampled_target, 'w', **kwargs) as dst:
                for i, band in enumerate(target_arr, 1):  # i -> 1,2,3,4;  target_arr is in the reference array
                    dest = np.empty(shape=(math.ceil(band.shape[0]/scale_factor_x), math.ceil(band.shape[1]/scale_factor_y)),
                                    dtype=band.dtype)
                    reproject(band, dest, src_transform=target_transform, src_crs=target_crs,
                              dst_transform=new_transform, dst_crs=target_crs, resampling=Resampling.bilinear)
                    dst.write(dest, indexes=i)
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
    call('rio clip ' + input_big + ' ' + outfile + ' --like ' + downsampled_target, shell=True)
    return conductedDownsample, downsampled_target, outfile, input_target


def clip_to_shapefile(raster, shapefile, force_dims=None, outname="clipped_raster.tif", outdir=None):
    """
    Clip the input raster to the given shapefile.

    :param raster: (str) path to raster to clip
    :param shapefile: (str) path to shapefile with features to use for clipping
    :param force_dims: (tuple) height, width to enforce on the output image. Unstable.
    :param outname: (str) name of the output raster
    :param outdir: (str) if given, save the output this folder
    :return:
    """
    if outdir:
        # if outdir is specified, save the clipped raster there
        outpath = os.path.join(outdir, outname)
    else:
        # otherwise, save to the same folder as the input raster
        outpath = os.path.join(os.path.split(raster)[0], outname)
    # load in the features from shapefile
    with fiona.open(shapefile, 'r') as src:
        features = [feature['geometry'] for feature in src]
    # create clipped raster data and transform
    with rasterio.open(raster, 'r') as src:
        out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
        out_meta = src.meta.copy()
    # update metadata with new height, width, and transform
    if force_dims:
        out_meta.update({"height": force_dims[0], "width": force_dims[1], "transform": out_transform})
        # save to outpath
        with rasterio.open(outpath, 'w', **out_meta) as dst:
            dst.write(out_image[:, :force_dims[0], :force_dims[1]])
    else:
        out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        # save to outpath
        with rasterio.open(outpath, 'w', **out_meta) as dst:
            dst.write(out_image)
    return outpath


def make_shapefile_from_raster(raster, outname="vectorized.shp", outdir=None):
    """
    Generate a shapefile with a single feature outlining the extent of the input raster.
    There is probably a better way to do this, but this works...

    :param raster: (str) path to raster to vectorize
    :param outname: (str) name of the generated shapefile
    :param outdir: (str) if given, save the output to this folder
    :return:
    """
    if outdir:
        # if outdir is specified, save the clipped raster there
        outpath = os.path.join(outdir, outname)
    else:
        # otherwise, save to the same folder as the input raster
        outpath = os.path.join(os.path.split(raster)[0], outname)
    d = dict()
    d['val'] = []
    geometry = []
    with rasterio.open(raster, 'r') as src:
        empty = np.zeros_like(src.read(1))
        for shp, val in rasterio.features.shapes(source=empty, transform=src.transform):
            d['val'].append(val)
            geometry.append(shape(shp))
        raster_crs = src.crs
    df = pd.DataFrame(data=d)
    geo_df = GeoDataFrame(df, crs={'init': raster_crs['init']}, geometry=geometry)
    geo_df['area'] = geo_df.area
    geo_df.to_file(outpath, driver="ESRI Shapefile")
    return outpath


def update_projection(src_image, dst_image, outfile="reprojected.tif", outdir=None):
    """
    Reproject dst_image using the CRS from src_image.

    :param src_image: Use the CRS from this image
    :param dst_image: Update this image with the CRS from the other image
    :param outfile: Save updated image with this name
    :param outdir: Save image here. If not given, it will get saved in same place as dst_image
    :return:
    """
    if outdir:
        outfile = os.path.join(outdir, os.path.split(outfile)[1])
    else:
        outfile = os.path.join(os.path.split(dst_image)[0], os.path.split(outfile)[1])
    with rasterio.open(src_image) as src_ds:
        src_crs = src_ds.crs
    with rasterio.open(dst_image, 'r') as dst_ds:
        dst_crs = dst_ds.crs
        dst_transform = dst_ds.transform
        dst_arr = dst_ds.read()
        kwargs = dst_ds.meta
        with rasterio.open(outfile, 'w', **kwargs) as out_ds:
            for i, band in enumerate(dst_arr, 1):
                dest = np.zeros_like(dst_arr[0])
                reproject(band, dest, src_crs=src_crs, dst_crs=dst_crs, src_transform=dst_transform,
                          dst_transform=dst_transform)
                out_ds.write(dest, i)
    return outfile


def perform_downsample(src_image, scale_factor_x, scale_factor_y, outfile="downsampled.tif", outdir=None):
    """

    :param src_image: (string) filepath of image to downsample
    :param scale_factor_x: (float) scale to adjust image. >1 = downsample.(resolution of reference) / (resolution of target)
    :param scale_factor_y: (float) >1 = downsample. (resolution of reference) / (resolution of target)
    :param outfile: (string) filename
    :param outdir: (string) directory to save output image
    :return:
    """
    if outdir:
        outfile = os.path.join(outdir, os.path.split(outfile)[1])
    else:
        outfile = os.path.join(os.path.split(src_image)[0], os.path.split(outfile)[1])
    with rasterio.open(src_image) as src_ds:
        initial_transform = src_ds.transform
        target_arr = src_ds.read()
        kwargs = src_ds.meta
        target_crs = src_ds.crs
        new_transform = rasterio.Affine(initial_transform.a * scale_factor_x,
                                        initial_transform.b, initial_transform.c,
                                        initial_transform.d, initial_transform.e * scale_factor_y,
                                        initial_transform.f)
    kwargs['transform'] = new_transform
    kwargs['height'] = round(target_arr[0].shape[0] / scale_factor_x)
    kwargs['width'] = round(target_arr[0].shape[1] / scale_factor_y)
    # Note - may be disagreement in Python 2 between different dtypes (long vs long long)
    with rasterio.open(outfile, 'w', **kwargs) as dst:
        for i, band in enumerate(target_arr, 1):  # i -> 1,2,3,4;  target_arr is in the reference array
            dest = np.empty(shape=(int(round(band.shape[0] / scale_factor_x)), int(round(band.shape[1] / scale_factor_y))),
                            dtype=band.dtype)
            reproject(band, dest, src_transform=initial_transform, src_crs=target_crs,
                      dst_transform=new_transform, dst_crs=target_crs, resampling=Resampling.bilinear)
            dst.write(dest, indexes=i)
    return outfile


def set_no_data(img_w_nodata, cropped_img, outfile="out.tif", outdir=None, src_nodata=0.0, dst_nodata=0.0, save_mask=False,
                datatype_out=gdal.GDT_UInt16):
    """
    Takes in two images- one with no-data values around the perimeter, one without. Applies matching no-data values to the second image and saves as new output.

    :param img_w_nodata: (string) filepath of target image containing some no-data values around the perimeter. Also supports numpy array.
    :param cropped_img: (string) Must have the same bounding box as img_w_nodata, but can have data everywhere. No-data values from img_w_nodata will be added to cropped_img (no overwrite)
    :param outfile: (string) name of output file. DO NOT ADD LOCATION. File location will be added later.
    :param outdir: (str) directory to save output file
    :param src_nodata: (float) the no-data value in the incoming image.
    :param dst_nodata: (float) the no-data value in the output image.
    :param save_mask: (bool) save the mask as a GeoTiff
    :param datatype_out: (gdal datatype) desired filetype for the output file
    :return img_nodata_source: unchanged from input
    :return outfile: (string) filepath of image with no-data values applied
    :return no_data_mask:
    """

    # array version:
    if (type(img_w_nodata) == np.ndarray) or (type(img_w_nodata) == np.ma.core.MaskedArray):
        # this is the case where the inputs are arrays
        # this is essentially just for the vegetation mask (single band) at the moment
        # TODO: switch this to rasterio also
        gdal.AllRegister()
        img_nodata_source = img_w_nodata      # the file with some no-data pixels that will be copied over.
        img_nodata_target = gdal.Open(cropped_img, gdal.GA_Update)  # file which will have no-data pixels added.
        if len(img_nodata_source.shape) == 3:
            _, rows, cols = img_nodata_source.shape
        else:
            rows, cols = img_nodata_source.shape
        cols2 = img_nodata_target.RasterXSize
        rows2 = img_nodata_target.RasterYSize
        bands = img_nodata_target.RasterCount
        if (cols != cols2) or (rows != rows2):
            warnings.warn("size mismatch. use images with same dimensions.")

        rows = int(np.min((rows, rows2)))
        cols = int(np.min((cols, cols2)))
        target_arr = np.zeros([np.max((rows, rows2)), np.max((cols, cols2)), bands])  # intializing. will become the output image
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
            target_arr[:rows, :cols, band] = np.where(no_data_mask[:rows, :cols], dst_nodata_arr[:rows, :cols],
                                              target_arr[:rows, :cols, band])
            target_arr[:rows, :cols, band] = ma.array(target_arr[:rows, :cols, band],
                                                      mask=no_data_mask, fill_value=dst_nodata)

        print("Writing and saving...")
        if outdir:
            dir_target = outdir
        else:
            dir_target = os.path.split(cropped_img)[0]
        outfile = os.path.join(dir_target, outfile)
        target_DS = gdal.GetDriverByName('GTiff').Create(outfile, int(cols), int(rows), bands, datatype_out)
        for band in range(bands):
            target_DS.GetRasterBand(band+1).WriteArray(target_arr[:rows, :cols, band])
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
        planet = gdal.Open(img_w_nodata)
        target = gdal.Open(cropped_img, gdal.GA_Update)
        bands = target.RasterCount
        cols = planet.RasterXSize
        rows = planet.RasterYSize
        cols2 = target.RasterXSize
        rows2 = target.RasterYSize
        rows = int(np.min((rows, rows2)))
        cols = int(np.min((cols, cols2)))

        if outdir:
            dir_target = outdir
        else:
            dir_target = os.path.split(cropped_img)[0]
        outfile = os.path.join(dir_target, outfile)
        target_DS = gdal.GetDriverByName('GTiff').Create(outfile, cols, rows, bands, datatype_out)

        planet_arr = np.zeros([rows, cols, bands])
        target_arr = np.zeros([np.max((rows, rows2)), np.max((cols, cols2)), bands])
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
            planet_arr[:, :, band] = np.array(planet.GetRasterBand(band+1).ReadAsArray())[:rows, :cols]
            no_data_mask[:, :, band] = planet_arr[:, :, band] == src_nodata
            # Pixels with data marked as 0.0. Pixels with no data marked as 1.0. Could flip this by using: planet_arr[:,:,band] != src_nodata
            # TODO flip this so that it makes more sense...

        new_mask = no_data_mask.sum(axis=2)
        new_mask = new_mask != 0     # no-data where new_mask = True, i.e. setup for masked arrays

        print("Applying no-data array...")
        for band in range(bands):
            target_arr[:, :, band] = np.array(target.GetRasterBand(band+1).ReadAsArray())
            target_arr[:rows, :cols, band] = np.where(no_data_mask[:rows, :cols, band], dst_nodata_arr[:rows, :cols],
                                              target_arr[:rows, :cols, band])
            target_arr[:rows, :cols, band] = ma.array(target_arr[:rows, :cols, band],
                                                      mask=new_mask, fill_value=dst_nodata)

        print("Writing and saving...")
        for band in range(bands):
            target_DS.GetRasterBand(band+1).WriteArray(target_arr[:rows, :cols, band])

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

        return img_w_nodata, outfile, new_mask


def img_to_array(img_path):
    """
    Returns a list of arrays for a multiband image. Returns a single numpy array for a single band image.
    :param img_path:
    :return:
    """
    with rasterio.open(img_path, 'r') as src:
        bands_array = src.read()

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


# TODO: swap this whole function to rasterio
def projection_match(image_1, image_2, outdir=None):
    """
    Check if image1 and image2 are in the same spatial reference system.
    If they are not, the SRS from image 1 is applied to image 2. The reprojected image is saved either in the folder
    where image2 lives, or in outdir.
    :param image_1: (str) path
    :param image_2: (str) path
    :param outdir: (str) optional output folder to save the reprojected image in.
    :return:
    """
    image1_ds = gdal.Open(image_1)
    image2_ds = gdal.Open(image_2)
    image1_proj = image1_ds.GetProjection()
    image2_proj = image2_ds.GetProjection()
    image1_srs = osr.SpatialReference(wkt=image1_proj).GetAttrValue('authority', 1)
    image2_srs = osr.SpatialReference(wkt=image2_proj).GetAttrValue('authority', 1)
    image1_ds = None
    image2_ds = None
    if image1_srs == image2_srs:
        print("Image projections are identical")
        return image_2
    else:
        warnings.warn("The projections are different. Attemping to fix that. ")
        print("Assigning projection from " + os.path.split(image_1)[1] + " to " + os.path.split(image_2)[1])
        if outdir:
            dir_target = outdir
        else:
            dir_target = os.path.split(image_2)[0]
        image2_reprojected = image_2[:-4] + "_reprojected.tif"
        update_projection(image_1, image_2, image2_reprojected, outdir=dir_target)
        print("reprojected image at " + os.path.join(dir_target, image2_reprojected))
        return os.path.join(dir_target, image2_reprojected)


def main(image_ref, image_reg_ref, image_targ, allow_reg=False, view_radcal_fits=True,
         src_nodataval=0.0, dst_nodataval=0.0, udm=None, ndvi_thresh=0.0, nochange_thresh=0.95, outdir=None,
         datatype_out=gdal.GDT_UInt16, is_ps=True):
    """
    Purpose: radiometrically calibrate a target image to a reference image.
    Optionally update the georeferencing in the target image.
    :param image_ref: (str) filepath of radiometry reference image
    :param image_reg_ref: (str) filepath of geolocation reference image
    :param image_targ: (str) filepath of target image
    :param allow_reg: (bool) whether the target image needs registration
    :param view_radcal_fits: (bool) whether the radcal fits should be displayed
    :param src_nodataval: (float) no-data value in the input images
    :param dst_nodataval: (float) no-data value to be applied to the output images
    :param udm: (list, tuple, or string) filepath of Unusable Data Mask(s) which will be applied to the final image
    :param ndvi_thresh: (float) values -1.0 to 1.0 are valid. Any pixels with NDVI below set threshold will be masked.
    :param nochange_thresh: (float) values 0 to 1.0 (exclusive) are valid. Chi2 threshold for invariance in MAD.
    :param outdir: (str) folder to save all outputs. Will be created if it does not exist.
    :param datatype_out: GDAL data type to save outputs, e.g. gdal.GDT_Float32.
    :param is_ps: (bool) if the target image is from PlanetScope, PS-specific functions will be applied
    :return: outpath_final: (str) path to final output image.
    """
    start = time.time()
    # Step 0: check image metadata to see if it has an acceptable level of cloud.
    image_targ_dir = os.path.split(image_targ)[0]
    if is_ps:
        try:
            import planetscope_utilities as psu
        except ImportError:
            warnings.warn("is_ps == True, but cannot import planetscope_utilities. Proceeding without PS functions.")
            is_ps = False
    if is_ps:
        below_cloud_thresh = psu.check_for_clouds(image_targ_dir, tolerance=0.5)
        if not below_cloud_thresh:
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

    # Step 0.5: check to make sure all input images are in the same projection
    with rasterio.open(image_ref) as src:
        rad_ref_meta = src.meta
    with rasterio.open(image_reg_ref) as src:
        reg_ref_meta = src.meta
    with rasterio.open(image_targ) as src:
        image_targ_meta = src.meta
    rad_ref_crs = rasterio.crs.CRS(rad_ref_meta['crs'])
    reg_ref_crs = rasterio.crs.CRS(reg_ref_meta['crs'])
    target_crs = rasterio.crs.CRS(image_targ_meta['crs'])
    # TODO this may fail if a crs is not already set up as an epsg
    if not rad_ref_crs.to_epsg() == reg_ref_crs.to_epsg() == target_crs.to_epsg():
        warnings.warn("The projects are different. Attempting to fix that. ")
        print("Projection of the radiometric reference: " + rad_ref_crs.to_string())
        print("Projection of the registration reference: " + reg_ref_crs.to_string())
        print("Projection of the target image: " + target_crs.to_string())
        if rad_ref_crs.to_string() != reg_ref_crs.to_string():
            if outdir:
                dir_target = outdir
            else:
                dir_target = os.path.split(image_reg_ref)[0]
            reproj_reg_ref = os.path.join(dir_target, 'reg_ref_reprojected.tif')
            update_projection(image_ref, image_reg_ref, 'reg_ref_reprojected.tif', outdir=dir_target)
            image_reg_ref = reproj_reg_ref
        if rad_ref_crs.to_string() != target_crs.to_string():
            if outdir:
                dir_target = outdir
            else:
                dir_target = os.path.split(image_targ)[0]
            reproj_target = os.path.join(dir_target, os.path.split(image_targ)[1][:-4] + '_reproj.tif')
            update_projection(image_ref, image_targ, reproj_target, outdir=dir_target)
            image_targ = reproj_target

    # Grab the spatial resolution of the images and check if they match for later
    res_ref_x = abs(rad_ref_meta['transform'][0])
    res_ref_y = abs(rad_ref_meta['transform'][4])
    res_targ_x = abs(image_targ_meta['transform'][0])
    res_targ_y = abs(image_targ_meta['transform'][4])
    if (res_ref_x != res_targ_x) or (res_ref_y != res_targ_y):
        print("Reference image and target image have different spatial resolutions")
        different_resolutions = True
        allowDownsample = True
    else:
        different_resolutions = False
        allowDownsample = False

    # Step 1: grab a reference image snip that will be used to align the target image.
    trim_out = trim_to_image(image_reg_ref, image_targ, allow_downsample=False, outdir=outdir)

    # Step 2: align the Planet image to that snip (if permitted).
    if allow_reg:
        image2_aligned = register_image2(image_targ, trim_out[2], outdir=outdir)
        # out is a full-resolution image. 2nd arg is cropped reference image.
    else:
        image2_aligned = image_targ  # keep in mind that this has a path attached

    # TODO: check generalizability for diff combinations of downsample, register, etc.
    # Step 3: create a new snip of the radiometric reference image to match aligned the target image.
    if image_ref != image_reg_ref:
        # note: this also makes a downsampled version of the target image
        trim_out = trim_to_image(image_ref, image2_aligned, outdir=outdir)
    else:
        # if the radiometric and geolocation references are the same image
        if outdir:
            dir_target = outdir
        else:
            dir_target = os.path.split(image_targ)[0]
        trim_out = (allowDownsample, image2_aligned,
                    os.path.join(dir_target, os.path.split(image_ref)[1][:-4] + "_trimmed.tif"),
                    image2_aligned)
        # need to also make a downsampled version of the target image
        if different_resolutions:
            downsampled_img = perform_downsample(image_targ, res_ref_x / res_targ_x, res_ref_y / res_targ_y,
                                                     os.path.split(image_targ)[1][:-4] + "_downsample.tif",
                                                 outdir=outdir)

    # note on next line: downsampled_img may or may not exist, depending if downsampling occurred.
    # trim_out[1:3] are all strings which include file location.
    _, downsampled_img, cropped_img, original_target_img = trim_out  # downsampled_img has good alignment
    downsampled_img = perform_downsample(image_targ, res_ref_x / res_targ_x, res_ref_y / res_targ_y,
                                             os.path.split(image_targ)[1][:-4] + "_downsample.tif", outdir=outdir)
    if not allowDownsample:
        downsampled_img = original_target_img

    # Step 3.5: check if the radiometric reference image snip is actually smaller than the target image.
    # if so, trim the target image down to the size of the reference image snip.
    with rasterio.open(cropped_img, 'r') as src:
        cropped_dimensions = (src.height, src.width)
    with rasterio.open(downsampled_img, 'r') as src:
        downsampled_dimensions = (src.height, src.width)
    if (cropped_dimensions[0] < downsampled_dimensions[0]) or (cropped_dimensions[1] < downsampled_dimensions[1]):
        warnings.warn("Part of the target image's spatial extent falls outside the reference image. Clipping.")
        shapefile_name = "rad_ref_extent.shp"
        clipped_downsampled_target_name = downsampled_img[:-4] + "_clip.tif"
        clipped_fullres_target_name = image2_aligned[:-4] + "_clip.tif"
        shapefile_path = make_shapefile_from_raster(cropped_img, shapefile_name, outdir=outdir)
        downsampled_img = clip_to_shapefile(downsampled_img, shapefile_path, force_dims=cropped_dimensions,
                                            outname=clipped_downsampled_target_name, outdir=outdir)
        image2_aligned = clip_to_shapefile(image2_aligned, shapefile_path, force_dims=cropped_dimensions,
                                           outname=clipped_fullres_target_name, outdir=outdir)

    # Step 4: generate a veg mask at target image resolution, downsample and apply, but save full res for later
    # TODO: swap this to a mask layer rather than assigning no-data values
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
    if allowDownsample:
        veg_mask_downsamp = perform_downsample(veg_mask_full_res_img, res_ref_x / res_targ_x, res_ref_y / res_targ_y,
                                                   "veg_mask_downsample.tif", outdir=outdir)
    else:
        veg_mask_downsamp = veg_mask_full_res_img
    veg_mask_downsamp_arr = img_to_array(veg_mask_downsamp)
    target_downsamp_vegmasked = set_no_data(veg_mask_downsamp_arr, downsampled_img, outfile="target_with_veg_mask.tif",
                                            outdir=outdir, src_nodata=dst_nodataval, dst_nodata=dst_nodataval,
                                            datatype_out=datatype_out)[1]

    # Step 5: add no data values into Landsat image so that it will play nice with iMad.py and radcal.py
    reference_nodata = os.path.split(cropped_img)[1][:-4] + "_nodata.tif"
    no_data_out_novegmask = set_no_data(downsampled_img, cropped_img, reference_nodata, outdir=outdir,
                                        dst_nodata=dst_nodataval)  # need version that hasn't been masked too.
    no_data_out_vegmask = set_no_data(target_downsamp_vegmasked, cropped_img, reference_nodata, outdir=outdir,
                                      dst_nodata=dst_nodataval)

    # at this point, we have a target image at reference image resolution and a ref image with target no-data values.
    # everything we need for MAD + radcal to run
    # target_img = downsampled version of planet at this point
    target_img_novegmask = no_data_out_novegmask[0]  # note that target_img and landsat_img include paths
    reference_img_novegmask = no_data_out_novegmask[1]
    target_img_vegmask = no_data_out_vegmask[0]  # note that target_img and landsat_img include paths
    reference_img_vegmask = no_data_out_vegmask[1]

    end = time.time()
    print("===============================")
    print("Time elapsed: " + str(int(end-start)) + " seconds")
    print("===============================")
    print("Beginning MAD...")
    outfile_MAD = os.path.split(target_img_novegmask)[1][:-4] + "_MAD.tif"
    outfile_RAD = os.path.split(target_img_novegmask)[1][:-4] + "_RAD.tif"
    outfile_final = os.path.split(image_targ)[1][:-4] + "_FINAL.tif"
    run_MAD(target_img_vegmask, reference_img_vegmask, outfile_MAD, outdir=outdir)
    # image2_aligned is the full resolution target scene
    print("Beginning radcal...")
    print("Target image: " + os.path.split(target_img_novegmask)[1])
    print("Reference image: " + os.path.split(reference_img_novegmask)[1])
    normalized_fsoutfile = run_radcal(target_img_novegmask, reference_img_novegmask, outfile_RAD, outfile_MAD,
                                      image2_aligned, view_plots=view_radcal_fits, outdir=outdir,
                                      nochange_thresh=nochange_thresh)
    # Step 6: re-apply no-data values to the radiometrically corrected full-resolution planet image
    # necessary since radcal applies the correction to the no-data areas
    if udm:
        # Based on the PlanetScope UDM
        # Documentation on Planet UDM doesn't seem to agree with actual values:
        # https://assets.planet.com/docs/Planet_Combined_Imagery_Product_Specs_letter_screen.pdf
        if (type(udm) == list) or (type(udm) == tuple):
            for raster in udm:
                # expects each item in the list udm to be a file PATH, not just filename. Updates files in place.
                process_udm(raster, src_nodata=1.0)
            if outdir:
                merged_udm = os.path.join(outdir, "merged_udm.tif")
            else:
                merged_udm = os.path.join(os.path.split(udm[0])[0], "merged_udm.tif")
            udm_merger(udm, merged_udm)
            # check if the merged UDM is in the same projection as the final image
            merged_udm = projection_match(normalized_fsoutfile, merged_udm)
            udm_as_arr = img_to_array(merged_udm)
            final_images = set_no_data(udm_as_arr, normalized_fsoutfile, outfile_final, src_nodata=1.0,
                                       dst_nodata=dst_nodataval, save_mask=True)
        elif type(udm) == str:
            # assume that if we got a string, it is a single default UDM filepath that still must be processed.
            process_udm(udm, src_nodata=1.0)
            udm = projection_match(normalized_fsoutfile, udm)
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
    image2 = input("Location of target image to be radiometrically normalized: ")
    assert isinstance(image_reg_ref, str)
    udm = input("Location of UDM to apply to target image? n for none: ")
    assert isinstance(udm, str)
    if udm == "n":
        udm = None
    output_dir = input("Location of directory to save outputs? (will be created if does not exist): ")
    assert isinstance(output_dir, str)
    udms = input("Apply a usable data mask? Filepath if yes, otherwise n: ")
    assert isinstance(udms, str)
    if udms == "n":
        udms = False
    allowRegistration = input("Allow target image to be re-registered if needed? y/n: ")
    assert isinstance(allowRegistration, str)
    while (allowRegistration != "y") and (allowRegistration != "n"):
        allowRegistration = input("Try again. Allow re-registration? y/n: ")
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
    while (view_radcal_fits != "y") and (view_radcal_fits != "n"):
        view_radcal_fits = input("Try again. Show radcal fits? y/n: ")
        assert isinstance(view_radcal_fits, str)
    if view_radcal_fits == "y":
        view_radcal_fits = True
    elif view_radcal_fits == "n":
        view_radcal_fits = False
    else:
        print("Must choose y or n. Try again.")
        quit()
    main(image1, image_reg_ref, image2, udm=udm, allow_reg=allowRegistration,
         view_radcal_fits=view_radcal_fits, outdir=output_dir)
