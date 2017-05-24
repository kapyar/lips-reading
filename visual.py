# -*- coding: utf-8 -*-
# USAGE
# python visual.py --data data/aspect.dat


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
import data_loader as dl


def plot_all_frames_in_word(word):
    for frame in word:
        for points in frame:
            plt.scatter(points[0], points[1])

    plt.show()


def plot_all_mean(words):
    for key in words.keys():
        plt.title(key)
        for word in words[key]:
            plt.scatter(word[0], word[1])
            plt.show()

data = dl.load("data/dictionary_spacial_frame_mean.dat")
plot_all_mean(data)
