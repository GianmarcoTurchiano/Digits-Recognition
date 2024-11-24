"""
Code for saving the data in the pickle format.
"""
import pickle


def save_pickle_data(out_path, features, labels):
    """
    Saves a pickle file with keys 'X' and 'y'.
    """
    with open(out_path, 'wb') as file:
        pickle.dump({
            'X': features,
            'y': labels
        }, file)
