# USAGE
# python data_loader.py -o data/dictionary_spacial_mean.dat


import pickle
import argparse


def load(path_to_data):
    with open(path_to_data, 'rb') as handle:
        data = pickle.load(handle)

    return data


def show_dictionary(path_to_data):

    data = load(path_to_data)

    print("[INFO] ==== DICTIONARY START ===")
    print (data)
    print("[INFO] ==== DICTIONARY FINISH ===")


def min_word_len_in(path_to_data):

    data = load(path_to_data)
    min = 1000000000000

    for key in data.keys():
        words = data[key]
        for word in words:
            if len(word) < min:
                min = len(word)

    return min


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--origin", required=True,
        help="path to the raw video")

    args = vars(ap.parse_args())

    show_dictionary(args["origin"])


