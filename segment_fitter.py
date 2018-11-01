import numpy as np


def seg_fit(vector, maxsegs, maxerror, spacing_array=[], max_iter=1000):
    """
    # Purpose: Given a set of values, fit a series in linear segments
    :param vector: 1D array of values to fit using linear segments
    :param maxsegs: maximum number of segments to use in fit
    :param maxerror: maximum permissible error for a segment
    :param spacing_array: 1D array defining the space between rasters, e.g. list of day-of-years
    :param max_iter: max number of iterations to go through in segment fitting process
    :return: 1D array with segments evaluated at each value in spacing array (or evenly spaced if no spacing array)
    """
    errorreal = 0.0
    initial = np.arange(len(vector)-1)    # start of each segment
    final = np.arange(len(vector)-1)+1    # end of each segment
    n_segs = len(vector)
    if spacing_array != []:
        xvals = spacing_array
    else:
        xvals = range(0, len(vector))

    iterations = 0
    while (errorreal <= maxerror) and (n_segs != 1):
        initial2 = np.zeros(len(initial)-1, dtype=int)
        final2 = np.zeros(len(final)-1, dtype=int)
        error = np.zeros(len(initial2), dtype=float)

        for i in range(0, len(initial2)):
            initial2[i] = initial[i]
            final2[i] = final[i+1]
            # y-values are values of 'vector' at initial2[i] and final2[i] (just two points)
            # x-values are values of 'xvals' at initial2[i] and final2[i] (just two points)
            # places where we want to interpret are all the 'xvals' locations between initial2[i] and final2[i]
            segment = np.interp(xvals[initial2[i]:final2[i]+1], [xvals[initial2[i]],xvals[final2[i]]], [vector[initial2[i]],vector[final2[i]]])
            # error between point and segment value at that location
            error[i] = np.sqrt(np.mean((vector[np.arange(final2[i]-initial2[i]+1)+initial2[i]]-segment)**2.0))

        minimum = np.min(error)
        minimum_loc = np.argmin(error)
        errorreal = minimum
        if minimum_loc == len(error)-1:
            # need some special handling for when the last term has the lowest error
            # this will mean that there is no value in initial corresponding the final2[minimum_loc]
            # this is what IDL does by default
            find = -1

        else:
            # find is the location in the initial's vector where the error is a minimum in the final2 vector
            # issue is that the final2 vector can have values beyond the initial vector (i.e. end of the last segment)
            find = np.argwhere(initial == final2[minimum_loc])
            find = find.flatten()[0]
        # ind is the location in the initial's vector where the error is a minimum in the initial2 vector
        ind = np.argwhere(initial == initial2[minimum_loc])     # outputs an array
        ind = ind.flatten()[0]

        if (minimum >= maxerror) and (len(error) >= maxsegs):
            maxerror = minimum

        if minimum <= maxerror:
            # construct the new initial position vector by dropping the lowest error term (initial[minimum_loc+1])
            initial3 = np.concatenate((initial[0:ind+1], initial[find:len(initial)]))
            if final2[minimum_loc] == np.max(final2):
                initial3 = initial[0:ind+1]

            """
            # construct the new final position vector
            if final2[minimum_loc] == np.max(final2):
                # if the place with the lowest error is the last value in the final2 vector, do this.
                # TODO: fix TypeError: 'only integer scalar arrays can be converted to a scalar index'
                # TODO: happening when second term is an empty array
                final3 = final[0:-1]
            elif ind == 0:
                final3 = np.insert(final[find:len(final)], 0, int(final2[minimum_loc]))
            else:
                final3 = np.concatenate((final[0:ind], final[find-1:len(final)]))
            """
            # construct the new final position vector
            final3 = np.concatenate((final[0:ind], final[find-1:len(final)]))
            if ind == 0:
                final3 = np.insert(final[find:len(final)], 0, int(final2[minimum_loc]))
            if final2[minimum_loc] == np.max(final2):
                #final3 = np.concatenate(final[0:ind], final[find:len(final)])
                #print("final2[minimum_loc] == np.max(final2)")
                final3 = np.append(final[0:ind], final[find])

            # update the segment vectors
            initial = initial3
            final = final3
            n_segs = len(initial3)

            if n_segs == 1:
                initial = [0]
                final = [len(vector)-1]
        iterations += 1
        if iterations > max_iter:
            print("Max iterations reached. ")
            break

    # where TX used VECTSIMPLIFIED, use "output_vectors". Move to a mandatory output instead of optional argument
    output_vectors = []
    for i in range(0, len(initial)):
        segment = np.interp(xvals[initial[i]:final[i]+1], [xvals[initial[i]], xvals[final[i]]], [vector[initial[i]], vector[final[i]]])
        if i == 0:
            output_vectors = segment
        else:
            output_vectors = np.append(output_vectors, segment[1:])

    return output_vectors
