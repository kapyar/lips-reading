# -*- coding: utf-8 -*-

import data_loader as dl
import numpy as np
import math
from scipy.stats import pearsonr
import math_utils as mutils

def find_word_exactly(feature_list, path_to_data):
    data = dl.load(path_to_data)

    for key in data.keys():
        for word in data[key]:
            if len(word) == len(feature_list):
                if np.allclose(feature_list, word, 0.2):
                    print("[INFO] founded {}".format(key))
                    return key
            else:
                # subarray all available with smallest len
                if len(word) < len(feature_list):
                    small = word
                    big = feature_list
                else:
                    small = feature_list
                    big = word

                for i in range(0, (len(big) - len(small))):
                    candidate = big[i : (len(small)+i)]
                    if np.allclose(candidate, small, 0.2):
                        print("[INFO] founded {}".format(key))
                        return key


    return "Unknown"




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
