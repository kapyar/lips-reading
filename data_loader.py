import pickle


def load(path_to_data):
    with open(path_to_data, 'rb') as handle:
        data = pickle.load(handle)

    return data