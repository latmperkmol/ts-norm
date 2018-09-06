
import numpy as np
import warnings


def single_band_change_detect(vector, spacing):
    """

    :param vector: 1D numpy vector with value of a pixel through
    :param spacing: 1D numpy vector specifying dates of each value in vector
    :return:
    """
    rise = vector[1:] - vector[:-1]
    run = spacing[1:] - spacing[:-1]
    flags = np.zeros(len(rise))
    try:
        slope = rise/run
    except ZeroDivisionError:
        warnings.warn("Found divide by 0 in slope calculation. Switching to looped processing. ")
        slope = np.zeros(len(rise))  # this is temp, fix code later
        # TODO abandon array-based operation and loop through array instead to catch the divide by 0 error and replace
        for i in range(0, len(slope)):
            try:
                slope[i] = rise[i]/run[i]
            except ZeroDivisionError:
                warnings.warn("Divide by 0 occurs at spot {}. Replacing with slope=0. ".format(i))
                slope[i] = 0

    for i in range(1, len(flags)):
        if slope[i-1] != slope[i] and slope[i]<0:
            flags[i] = 1

    return flags
