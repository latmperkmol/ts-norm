
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


def main(direc, shpfile, outfilename="cell_data.json", do_despike=False, fit_segments=False, doy_array=[]):
    """
    Run through each raster in the directory. For each raster, calculate some statistics for each zone and each band.
    A dictionary is generated containing each statistic for each raster, band, and zone.
    The dictionary is saved to a JSON file. It is also returned by the function for further use if needed.
    After calculating stats, despike the means.
    :param direc: directory containing raster files
    :param shpfile: .shp shapefile containing zones to analyze
    :param outfilename: filename to be saved in direc
    :param fit_segments: (bool) whether or not to fit segments to despiked data. If yes, a csv with segments will be
        saved in direc
    :param doy_array: (1D numpy array) vector with day of year for each raster in direc
    :return rasters_dict: dictionary with all stats for all bands, all zones, all rasters
    """

    raster_list = []
    bands = 1
    zones = 0

    # This is not recursive due to break. Does not search subdirectories.
    for dirpath, dirnames, filenames in os.walk(direc):
        for filename in [f for f in filenames if f.endswith(".tif")]:
            raster_list.append(filename)  # this should include filenames that are machine-callable
        break

    # initialize a dictionary, which will hold names of each raster and their corresponding stats for each band+zone.
    rasters_dict = {}

    # begin looping over rasters, calculating rasterstats and adding to dictionary
    for rasterfile in raster_list:
        rasterpath = os.path.join(direc, rasterfile)
        raster = gdal.Open(rasterpath)
        rastername = rasterfile[:-4]
        rasterstat = []
        bands = raster.RasterCount
        for b in range(bands):
            b+=1
            rasterstat.append(zonal_stats(shpfile, rasterpath, band=b, stats=['min', 'max', 'mean', 'median', 'std']))
        rasters_dict[rastername] = rasterstat

    sorted_dict = collections.OrderedDict(sorted(rasters_dict.items()))

    outfile = os.path.join(direc, outfilename)
    if os.path.isfile(outfile) == False:
        with open(outfile, 'w') as fp:
            json.dump(sorted_dict, fp)
    else:
        overwrite = raw_input(outfilename + " already exists. Overwrite? y or n: ")
        if overwrite == "y":
            print("Overwriting. ")
            with open(outfile, 'w') as fp:
                json.dump(sorted_dict, fp)
        else:
            print("Not saving file. ")

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
        export_dict_to_csv(sorted_dict, zone, bands=b, stat=stat, outfilename=outname_csv, out_dir = direc)

    # start by checking how many bands are present
    if do_despike:
        threshold = 0.5
        if bands == 1:
            for zone in range(zones):
                vector = np.loadtxt(os.path.join(direc, outnames[zone]), delimiter=',')    # load in the vectors
                despiked_vector = despike.despike(vector, threshold)[1]
                np.savetxt(os.path.join(direc, outnames[zone][:-4] + '_despiked.csv'), despiked_vector, delimiter=',')
                # TODO: get segment fitting to work!!
                if fit_segments:
                    segments = seg_fit(despiked_vector, 5, 0.01, doy_array)
                    seg_outfile = os.path.join(direc, outnames[zone][:-4] + '_segments.csv')
                    np.savetxt(seg_outfile, segments, delimiter=',')

        # if multiple bands, check for concurrent spikes first, then despike
        # run the despike, but only retrieve the first output (spike locations)
        # then compare each of these outputs to see where the concurrent spikes are. store locations in a numpy vector
        # then re-run the despike script, with known_spikes = spike locations from above
        # TODO: add segment fitting for despiked data
        if bands > 1:
            concurrence_num = min(int(math.floor(b*0.75)), bands-1)     # number of bands with spikes for it to count
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

    # perform segment fitting

    return rasters_dict


def export_dict_to_csv(dictionary, zone, bands=4, stat='mean', outfilename = "raster_means.csv", out_dir=os.getcwd()):
    # goal: create a 2D array: each column is a band, each row is a raster. Values are for given statistics.

    # just in case the sorted version wasn't passed
    sorted_dict = collections.OrderedDict(sorted(dictionary.items()))
    rows = len(sorted_dict)   # number of bands
    cols = bands   # number of rasters
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
                                      stats=['min', 'max', 'mean', 'median', 'std', 'count']))
    dictionary[rastername] = rasterstat
    return dictionary


def plot_custom(rasters_dict, DOY_array = [], choose_lower_lim = False, choose_upper_lim = False):
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


def indices_by_zone(rasters_dict, zone_num):
    red_mean = np.empty(0)
    NIR_mean = np.empty(0)
    red_std = np.empty(0)
    NIR_std = np.empty(0)
    blue_mean = np.empty(0)
    blue_std = np.empty(0)
    for raster in sorted(rasters_dict):
        red_mean = np.append(red_mean, (rasters_dict[raster][2][zone_num]['mean']))
        NIR_mean = np.append(NIR_mean, (rasters_dict[raster][3][zone_num]['mean']))
        blue_mean = np.append(blue_mean, (rasters_dict[raster][0][zone_num]['mean']))
        red_std = np.append(red_std, (rasters_dict[raster][2][zone_num]['std']))
        NIR_std = np.append(NIR_std, (rasters_dict[raster][3][zone_num]['std']))
        blue_std = np.append(blue_std, (rasters_dict[raster][0][zone_num]['std']))

    zone_NDVI_mean = (NIR_mean - red_mean)/(NIR_mean + red_mean)
    zone_NDVI_std = (NIR_std - red_std)/(NIR_std + red_std)

    G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
    zone_EVI_mean = G*(NIR_mean - red_mean)/(NIR_mean + C1*red_mean - C2*blue_mean + L)
    zone_EVI_std = G*(NIR_std - red_std)/(NIR_std+ C1*red_std - C2*blue_std + L)

    return zone_NDVI_mean, zone_NDVI_std, zone_EVI_mean, zone_EVI_std


def get_all_DOY(directory):
    """

    :param directory:
    :return:
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


if __name__ == '__main__':
    directory = raw_input("Directory of tif files: ")
    shapefile = raw_input("Path name of shapefile: ")
    dict = main(directory, shapefile)
