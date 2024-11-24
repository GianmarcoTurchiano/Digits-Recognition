"""
Code for loading data in the pickle format.
"""
import pickle


def load_pickle_data(path):
    """
    Returns the content of a pickle file.
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data
