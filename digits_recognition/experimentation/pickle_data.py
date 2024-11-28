"""
Code for loading and saving data in the pickle format.
"""
import pickle


def load_pickle_data(path):
    """
    Returns the content of a pickle file.
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def save_pickle_data(out_path, features, labels):
    """
    Saves a pickle file with keys 'X' and 'y'.
    """
    with open(out_path, 'wb') as file:
        pickle.dump({
            'X': features,
            'y': labels
        }, file)
