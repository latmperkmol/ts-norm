import numpy as np
import gdal
import os


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