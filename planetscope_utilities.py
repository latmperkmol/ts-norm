import numpy as np
import gdal
import os
import warnings
import json


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

    bands_array = []
    nodata_array = np.zeros([rows, cols])    # will store locations where a pixel is no-data. If no-data, then 1.0.
    for band in range(bands):
        temp_array = np.array(img_src.GetRasterBand(band+1).ReadAsArray()).astype(np.float)
        size = temp_array.size
        upper_cutoff = np.sort(temp_array, axis=None)[int(size*(1-count_thresh))]
        np.place(temp_array, temp_array >= upper_cutoff, nodata)
        bands_array.append(temp_array / np.max(temp_array)*100.)
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


def check_for_clouds(directory=".", tolerance=0.5):
    """
    Check the PlanetSCope metadata file for the target image and assess the fraction of cloud.
    Images exceeding cloud tolerance will not be processed.
    If multiple metadata files are present in the directory, the mean cloud fraction is used (suitable for mosaics)

    :param directory: (string) directory containing JSON files with image metadata
    :param tolerance: (float) values 0-1.0 (inclusive) are valid. Permissible fraction of cloud cover.
    :return: (boolean) True for a usable image, False otherwise.
    """
    cloud = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith("metadata.json")]:
            with open(os.path.join(dirpath, filename)) as file:
                metadata = json.load(file)
                cloud.append(metadata["properties"]["cloud_cover"])
    if not cloud:
        warnings.warn("Couldn't find the Planet metadata.json file. Assuming cloud cover is at an acceptable level. ")
        return True
    cloud_frac = sum(cloud)/len(cloud)
    if tolerance >= cloud_frac >= 0.0:
        return True
    else:
        return False