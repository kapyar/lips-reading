# -*- coding: utf-8 -*-

import data_loader as dl
import numpy as np
import math
import math_utils as mutils
import feature_extractor as fe

'''

    Aspect ration methods
    
'''


def find_word_exactly(mouth_points, path_to_data):
    data = dl.load(path_to_data)

    best_candidate = "Unknown"
    best_error = 10000000000

    for key in data.keys():
        for word in data[key]:
            if len(word) == len(mouth_points):
                temp_error = 0
                for a, b in zip(mouth_points, word):
                    temp_error += math.fabs(a - b)
                    if temp_error < best_error:
                        best_candidate = key
                        best_error = temp_error
            else:
                # subarray all available with smallest len
                if len(word) < len(mouth_points):
                    small = word
                    big = mouth_points
                else:
                    small = mouth_points
                    big = word

                for i in range(0, (len(big) - len(small))):
                    candidate = big[i: (len(small) + i)]
                    temp_error = 0
                    for a, b in zip(small, candidate):
                        temp_error += math.fabs(a - b)
                        if temp_error < best_error:
                            best_candidate = key
                            best_error = temp_error

    return best_candidate


def find_by_mean(feature_list, path_to_data):

    data = dl.load(path_to_data)
    mean = mutils.mean(feature_list)

    best_candidate = ""
    best_diff = 10

    for key in data.keys():
        words = data[key]
        for word in words:
            if math.fabs(word - mean) < best_diff:
                best_candidate = key
                best_diff = math.fabs(word - mean)

    return best_candidate


def find_by_mean_min_max_removed(feature_list, path_to_data):
    feature_list.remove(min(feature_list))
    feature_list.remove(max(feature_list))

    return find_by_mean(feature_list, path_to_data)



'''
    Spacial methods
'''

# -m full
def spacial_full(mouth_points, path_to_data):
    data = dl.load(path_to_data)

    # todo

    return "Unknown"


# use -m mean
def spacial_mean_points_shift(mouth_points, path_to_data):
    data = dl.load(path_to_data)

    mean = fe.spacial_mean_points(mouth_points)

    best_candidate = ""
    best_error = 1000000000

    for key in data.keys():
        words = data[key]
        for word in words:
            temp_error = 0
            for a, b in zip(mouth_points, word):
                temp_error += np.linalg.norm(a - b)
                if temp_error < best_error:
                    best_candidate = key
                    best_error = temp_error

    return best_candidate


# -m centroid
def spacial_centroid(mouth_points, path_to_data):
    data = dl.load(path_to_data)

    best_candidate = "Unknown"
    best_error = 10000000000

    for key in data.keys():
        for word in data[key]:
            if len(word) == len(mouth_points):
                temp_error = 0
                for a, b in zip(mouth_points, word):
                    temp_error += math.fabs(a - b)
                    if temp_error < best_error:
                        best_candidate = key
                        best_error = temp_error
            else:
                # subarray all available with smallest len
                if len(word) < len(mouth_points):
                    small = word
                    big = mouth_points
                else:
                    small = mouth_points
                    big = word

                for i in range(0, (len(big) - len(small))):
                    candidate = big[i: (len(small) + i)]
                    converted = fe.spacial_centroid(candidate)
                    temp_error = 0
                    for a, b in zip(small, converted):
                        temp_error += math.fabs(a - b)
                        if temp_error < best_error:
                            best_candidate = key
                            best_error = temp_error

    return best_candidate