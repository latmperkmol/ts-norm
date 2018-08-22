"""
Purpose: remove spikes from a dataset
"""

import numpy as np
import os

def despike(vector, threshold, known_spikes=[]):
    """
    IMPORTANT - has not been tested with known spikes!!  Suspect there are errors in that section of code.
    Only despikes one vector currently. Must be run on each band individually
    :param vector: (ndarray) raw data to be despiked.
    :param threshold: (float) number 0-1 determining how much to despike. Lower number -> more despiking
    :param known_spikes:
    :return:
    """
    # TODO: test with known spikes
    # TODO: add ability to despike multiband (2D array) data
    vec = vector
    if isinstance(vec, basestring):
        vec = np.loadtxt(vec)

    # If an array containing known spikes is given, only despike those locations
    if known_spikes != []:
        spike_locations = np.loadtxt(known_spikes, delimiter=',')
        trozo = spike_locations[1:len(spike_locations)-2]       # suspect that this line is bad. may need -1 instead
        # intializing other arrays to match trozo
        spike0 = trozo
        spike = trozo
        trozi = trozo*0.0

        if np.sum(trozo) != 0:
            indexMAX = np.argwhere(trozo)

            for i in range(0, int(np.sum(trozo)-1)):
               vec[indexMAX[i]+1] = np.mean([vec[indexMAX[i]], vec[indexMAX[i]+2]])
               spike[indexMAX[i]] = 0
               trozi[indexMAX[i]] = 1

    # If the spikes need to be detected, proceed as below
    else:
        left = vec[0:len(vec)-2]
        right = vec[2:len(vec)]
        spike = abs(vec[1:len(vec)-1]-((left+right)/2.))        # could use better indexing to make this more pythonic
        spike0 = spike
        dif = abs(left-right)/(spike+0.0000000000001)
        # TODO: check this next line since it looks very weird
        trozo = (dif < (1-threshold))*(spike != 0.)  # if the difference < 1-threshold, mark as binary spike, unless 'spike' is 0 there??

        trozi = trozo*0

        while np.sum(trozo) != 0:
            indexMAX = np.argmax(spike*trozo)  # need to assign 'indexMAX' the location of the largest value of spike*trozo

            vec[indexMAX+1] = np.mean([vec[indexMAX], vec[indexMAX+2]])
            spike[indexMAX] = 0
            trozi[indexMAX] = 1

            left = vec[0:len(vec)-3]
            right = vec[2:len(vec)-1]
            spike = abs(vec[1:len(vec)-2] - ((left+right)/2.))
            dif = abs(left-right)/(spike+0.0000000000001)
            trozo = (dif < (1. - threshold)) * (spike != 0.)  # if the difference < 1-threshold, mark as binary spike, unless 'spike' is 0 there??

    trozo = trozi
    if known_spikes != []:
        spike_locations = np.loadtxt(known_spikes, delimiter=',')
        trozo = np.append(trozo, spike_locations[len(spike_locations)-1])
        trozo = np.insert(trozo, 0, 0)
        if spike_locations[len(spike_locations)-1] == 1:
            vec[len(vec)-1] = vec[len(vec)-2]
    else:
        trozo = np.insert(trozo, 0, 0)
        if np.std([vec[len(vec)-2:len(vec)-1]]) > np.std([vec[len(vec)-3:len(vec)-2]]):
            vec[len(vec)-1] = vec[len(vec)-2]
            trozo = np.append(trozo, 1)
        else:
            trozo = np.append(trozo, 0)

    valuest = vec
    return trozo, valuest


if __name__ == '__main__':
    # expects files to be despiked to be .CSV
    location = raw_input("Name of directory or file to despike: ")
    threshold = float(raw_input("Despike threshold (between 0 and 1): "))
    spikes = raw_input("Are there known spike locations? y/n: ")
    if spikes == 'y':
        spikes = raw_input("File path of file with spikes: ")
    elif spikes == 'n':
        spikes = None
    else:
        print "Only y or n. Try again. "
        exit()

    if os.path.isfile(location):
        output = despike(location, threshold, spikes)
        np.savetxt(os.path.join(os.path.split(location)[0], os.path.split(location)[1][:-4] + '_despiked.csv'), output[1])
    elif os.path.isdir(location):
        all_despiked_vectors = []
        for dirpath, dirnames, filenames in os.walk(location):
            for filename in [f for f in filenames if f.endswith(".csv")]:
                print "Despiking " + filename
                output = despike(os.path.join(dirpath, filename), threshold, spikes)
                all_despiked_vectors.append(output[1])
                np.savetxt(os.path.join(location, filename[:-4] + "_despiked.csv"), output[1])


