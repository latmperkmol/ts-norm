
"""
Purpose: Set of tools for finding zonal statistics for a set of rasters and preparing for further analysis

In general, this should be run from terminal.
User will be prompted for directory containing TIF files and location of shapefile containing zones of interest.

Some functions are better called from a python script.
E.g. generating VIs using save_VIs() is most efficiently executed from a python script.

Once you've saved a JSON with all the cell statistics, it can easily be loaded back in as a dictionary:
with open(filelocation) as fp:
    cell_data_dict = json.load(fp)

OUTPUT DICTIONARY FORMAT:
dictionary[rastername][band][zone][stat]
"""


from rasterstats import zonal_stats
import os
import gdal
import numpy as np
import json
import collections
import math
import despike
from segment_fitter import seg_fit
from scipy.signal import savgol_filter


def main(direc, shpfile, out_dir, outfilename="cell_data.json", do_despike=False, fit_segments=False, doy_array=[], nodata=0.0,
         num_segs=10, window_len=11, polyorder=4):
    """
    Run through each raster in the directory. For each raster, calculate some statistics for each zone and each band.
    A dictionary is generated containing each statistic for each raster, band, and zone.
    The dictionary is saved to a JSON file. It is also returned by the function for further use if needed.
    After calculating stats, despike the means.
    :param direc: (string) directory containing raster files
    :param shpfile: (string) .shp shapefile containing zones to analyze
    :param out_dir: (string) directory to store output files
    :param outfilename: (string) filename of the dictionary of zonal statistics to be saved in direc
    :param fit_segments: (bool) whether or not to fit segments to despiked data. If yes, a csv with segments will be
        saved in direc
    :param doy_array: (1D numpy array) vector with day of year for each raster in direc
    :param nodata: (float) no-data value in input rasters
    :return rasters_dict: dictionary with all stats for all bands, all zones, all rasters
    """

    raster_list = []
    zones = 0  # this will get overwritten later.
    bands = 1  # this will get overwritten later.
    load_saved = "n"

    # path of dictionary of zonal stats
    outfile = os.path.join(direc, outfilename)

    # This is not recursive due to break. Does not search subdirectories.
    for dirpath, dirnames, filenames in os.walk(direc):
        for filename in [f for f in filenames if f.endswith(".tif")]:
            raster_list.append(filename)  # this should include filenames that are machine-callable
        break

    # initialize a dictionary, which will hold names of each raster and their corresponding stats for each band+zone.
    rasters_dict = {}

    if os.path.isfile(outfile) == True:
        print(outfilename + " already exists. ")
        load_saved = raw_input("Load this file? If not, it will be overwritten. y or n: ")

    if load_saved != "y":
    # begin looping over rasters, calculating rasterstats and adding to dictionary
        for rasterfile in raster_list:
            rasterpath = os.path.join(direc, rasterfile)
            raster = gdal.Open(rasterpath)
            rastername = rasterfile[:-4]
            rasterstat = []
            bands = raster.RasterCount
            for b in range(bands):
                b+=1
                rasterstat.append(zonal_stats(shpfile, rasterpath, band=b, stats=['min', 'max', 'mean', 'median', 'count']))
            rasters_dict[rastername] = rasterstat

        sorted_dict = collections.OrderedDict(sorted(rasters_dict.items()))

        # save the dictionary with the zonal stats
        with open(outfile, 'w') as fp:
            json.dump(sorted_dict, fp)

    else:
        print("Loading saved file " + outfile + " ... ")
        with open(outfile, 'r') as fp:
            cell_data = json.load(fp)
        sorted_dict = collections.OrderedDict(sorted(cell_data.items()))
        # check the number of bands
        raster_file = raster_list[0]
        raster_path = os.path.join(direc, raster_file)
        raster = gdal.Open(raster_path)
        bands = raster.RasterCount

    # grab the number of zones in a really, really horrible way. I am so sorry.
    for raster in sorted_dict:
        zones = len(sorted_dict[raster][0])
        break

    # create a number of CSV files = number of zones. Currently only writes out the means.
    outnames = []   # filenames of the CSVs for each zone
    stat = 'mean'
    for zone in range(zones):
        outname_csv = "zone" + str(zone) + "means.csv"
        outnames.append(outname_csv)
        export_dict_to_csv(sorted_dict, zone, bands=bands, stat=stat, outfilename=outname_csv, out_dir=direc)

    # start by checking how many bands are present
    if do_despike:
        threshold = 0.10
        if bands == 1:
            for zone in range(zones):
                vector = np.loadtxt(os.path.join(direc, outnames[zone]), delimiter=',')    # load in the vectors
                # take the vector and compress to remove missing data values
                compressed_vector, compressed_doy = compress_values(vector, doy_array, nodata=nodata)
                # FIRST we despike, SECOND we interpolate, THIRD we filter.
                despiked_vector = despike.despike(compressed_vector, threshold)[1]
                # resample and interpolate so that the values are evenly spaced
                resamp_doy = np.arange(compressed_doy.max())
                resamp_vec = np.interp(resamp_doy, compressed_doy, despiked_vector)
                # apply savitsky-golay filter, then despike
                smoothed_vector = savgol_filter(resamp_vec, window_len, polyorder, mode='mirror')

                np.savetxt(os.path.join(direc, outnames[zone][:-4] + '_comp_despiked.csv'), smoothed_vector,
                           delimiter=',')
                np.savetxt(os.path.join(direc, outnames[zone][:-4] + '_rel_doy.csv'), compressed_doy, delimiter=',')
                np.savetxt(os.path.join(direc, outnames[zone][:-4] + '_resamp_doy.csv'), resamp_doy, delimiter=',')
                if fit_segments:
                    segments = seg_fit(smoothed_vector, num_segs, maxerror=0.1, spacing_array=resamp_doy, max_iter=1000)
                    seg_outfile = os.path.join(direc, outnames[zone][:-4] + '_segments.csv')
                    np.savetxt(seg_outfile, segments, delimiter=',')

        # if multiple bands, check for concurrent spikes first, then despike
        # run the despike, but only retrieve the first output (spike locations)
        # then compare each of these outputs to see where the concurrent spikes are. store locations in a numpy vector
        # then re-run the despike script, with known_spikes = spike locations from above
        # NB: no compression of missing values occurs here!
        # TODO: add compression of missing values for multi-band
        # TODO: add segment fitting for multi-band
        if bands > 1:
            concurrence_num = min(int(math.floor(bands*0.75)), bands-1)   # number of bands with spikes for it to count
            spike_locations = []
            for zone in range(zones):
                means_array = np.loadtxt(os.path.join(direc, outnames[zone]), delimiter=',')
                zone_data = np.zeros([len(means_array[:, 0]), bands])
                spikes = np.zeros([len(means_array[:, 0]), bands])  # initialize
                for band in range(bands):
                    spike_locations.append(despike.despike(means_array[:, band], threshold)[0])
                    spikes[:, band] = spike_locations[band]     # boolean array with location of spikes in each band
                spikes_fname = "zone" + str(zone) + "_spike_locations.csv"
                np.savetxt(os.path.join(direc, spikes_fname), spikes, delimiter=',')
                find_concurrent_spikes(os.path.join(direc, spikes_fname),
                                       os.path.join(direc, "zone" + str(zone) + "_concurrent_spikes.csv"), concurrence_num)
                concurrent_spikes = np.loadtxt(os.path.join(direc, "zone" + str(zone) + "_concurrent_spikes.csv"))
                for band in range(bands):
                    conc_spikes_removed = despike.despike(means_array[:, band], threshold, known_spikes=concurrent_spikes)[1]
                    zone_data[:, band] = conc_spikes_removed
                np.savetxt(os.path.join(direc, outnames[zone][:-4] + '_despiked.csv'), zone_data, delimiter=',')
    return rasters_dict


def export_dict_to_csv(dictionary, zone, bands=4, stat='mean', outfilename="raster_means.csv", out_dir=os.getcwd()):
    # goal: create a 2D array: each column is a band, each row is a raster. Values are for given statistics.

    # just in case the sorted version wasn't passed
    sorted_dict = collections.OrderedDict(sorted(dictionary.items()))
    rows = len(sorted_dict)   # number of rasters
    cols = bands   # number of bands
    vals = np.zeros([rows, cols])   # array where stat values will be assigned

    r = 0
    for raster in sorted_dict:
        for band in range(bands):
            vals[r][band] = sorted_dict[raster][band][zone][stat]
        r+=1
    # replace any NaN values with 0.0
    np.nan_to_num(vals, copy=False)
    outfilepath = os.path.join(out_dir, outfilename)
    np.savetxt(outfilepath, vals, delimiter=",")
    return


def find_concurrent_spikes(csv_file, outfilepath, concurrence, header=False):
    """
    Finds rasters where at least three bands have spikes. Identifies these areas as spikes.
    Expects columnar data (each column is a band, each row is a raster).
    :param csv_file: columnar comma-separated file with location of each spike in each band
    :param outfilepath: location to save single column of concurrent spikes
    :param concurrence: the number of spikes that must appear concurrently to count as a true spike
    :return:
    """
    if header == False:
        header = 0
    else:
        header = 1
    data = np.loadtxt(csv_file, delimiter=',', skiprows=header)
    spikes = (np.sum(data, 1) >= concurrence)   # anywhere that the sum of a row >= 'concurrence', mark a spike
    np.savetxt(outfilepath, spikes, delimiter=",")
    return spikes


def calculate_raster_properties(dictionary, rasterpath, shapefile):
    """

    :param dictionary: dictionary containing name of each raster and stats by zone and band
    :param rasterpath: machine path to raster
    :param shapefile:
    :return: updated dictionary with new raster information appended
    """
    raster = gdal.Open(rasterpath)
    rasterfile = os.path.split(rasterpath)[1]
    # Remove the extension to make the raster name more easily human-readable:
    rastername = rasterfile[:-4]
    rasterstat = []
    for b in range(raster.RasterCount):
        b+=1
        rasterstat.append(zonal_stats(shapefile, rasterpath, band=b,
                                      stats=['min', 'max', 'mean', 'median', 'count']))
    dictionary[rastername] = rasterstat
    return dictionary


def plot_custom(rasters_dict, DOY_array=[], choose_lower_lim=False, choose_upper_lim=False):
    """

    :param rasters_dict: Dictionary containing stats for each band and zone for each raster.
    :param DOY_array: Optional array containing the day of year for each raster.
                      Makes plotting more meaningful; DOY will be on x-axis instead of raster number.
    :return:
    """
    import matplotlib.pyplot as plt
    band_num = int(raw_input("Band number? Count from 0. "))
    zone_num = int(raw_input("Feature number? Count from 0. "))
    stat = raw_input("Which statistic? ")
    stat_value = []
    for raster in sorted(rasters_dict):
        stat_value.append(rasters_dict[raster][band_num][zone_num][stat])

    fig, ax = plt.subplots()
    plt.ylabel(stat)
    plt.title(stat + " of zone " + str(zone_num) + ", " "band " + str(band_num+1))
    if DOY_array != []:
        ax.set_xlabel('DOY')
        ax.plot(DOY_array, stat_value, 'ko')
    else:
        ax.set_xlabel('raster number')
        ax.plot(stat_value, 'ko')

    if choose_upper_lim == True:
        upper_lim = int(raw_input("Upper y-limit of plot? "))
    if choose_lower_lim == True:
        lower_lim = int(raw_input("Lower y-limit of plot? "))

    if (choose_lower_lim == False) and (choose_upper_lim == False):
        ax.set_ylim(bottom=0)
    elif (choose_lower_lim == True) and (choose_upper_lim == False):
        ax.set_ylim(bottom=lower_lim, top=True)
    elif (choose_lower_lim == False) and (choose_upper_lim == True):
        ax.set_ylim(bottom=True, top=upper_lim)
    else:
        ax.set_ylim(bottom=lower_lim, top=upper_lim)
    plt.show()


def get_all_DOY(directory):
    """
    Builds a numpy array of the relative day of year (relative to first raster) based on Planet metadata.
    :param directory: (string) directory containing the Planet metadata JSON files. Files can be in subdirectories.
    :return rel_DOY: (numpy array) relative day of year, starting at 0
    """
    from datetime import date
    DOY_dict = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".json")]:
            with open(os.path.join(dirpath, filename)) as file:
                try:
                    meta = json.load(file)
                    ID = meta["id"]
                    sat_ID = meta["properties"]["satellite_id"]
                    raw_date = meta["properties"]["acquired"]
                    year = int(raw_date[:4])
                    month = int(raw_date[5:7])
                    day = int(raw_date[8:10])
                    date_class = date(year, month, day)
                    DOY_dict[ID] = {"date": date_class, "sat_ID": sat_ID}
                except KeyError:
                    print "Invalid JSON at " + dirpath + "\\" + filename
            dirnames[:] = []

    DOY_list = []
    sat_ID_list = []
    for key in DOY_dict:
        if (DOY_dict[key]["date"] in DOY_list) == False and (DOY_dict[key]["sat_ID"] in sat_ID_list) == False:
            DOY_list.append(DOY_dict[key]["date"])
            sat_ID_list.append(DOY_dict[key]["sat_ID"])
    delta_DOY = np.zeros(len(DOY_list)-1)   # array containing the number of days between each scene as datetime objects
    DOY_list = sorted(DOY_list)   # put the list of DOY in order from earliest to latest

    for i in range(len(delta_DOY)):   # get difference between each date
        delta_DOY[i] = (DOY_list[i+1] - DOY_list[i]).days
    rel_DOY = np.zeros(len(DOY_list))   # a list of the relative DOYs, counting first entry as day 0
    for i in xrange(1, len(rel_DOY)):
        rel_DOY[i] = rel_DOY[i-1] + delta_DOY[i-1]
    return rel_DOY


def compress_values(data_arr, spacing_arr=[], nodata=0.0):
    """
    Finds locations in the data array that having missing values (e.g. from no overlap between shpfile feature and raster).
    Removes these values from the data array, as well as the spacing array.
    Returns compressed versions of the data array and spacing array.
    :param data_arr: (ndarray) 1D array of raw data values
    :param spacing_arr: (ndarray) 1D array of relative spacing between data values (i.e. relative day of year)
    :param nodata: (float) value signifying a bad data point
    :return:
    """
    masked_data_arr = np.ma.masked_equal(data_arr, nodata)
    compressed_data_arr = masked_data_arr.compressed()
    if len(spacing_arr) == len(data_arr):
        masked_spacing_arr = np.ma.array(spacing_arr, mask=masked_data_arr.mask)
        compressed_spacing_arr = masked_spacing_arr.compressed()
        return compressed_data_arr, compressed_spacing_arr
    else:
        return compressed_data_arr, spacing_arr


if __name__ == '__main__':
    directory = raw_input("Directory of tif files: ")
    shapefile = raw_input("Path name of shapefile: ")
    output_dir = raw_input("Directory for outputs: ")
    dict = main(directory, shapefile)
