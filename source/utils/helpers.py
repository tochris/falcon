import os
import pickle

def save_obj(obj, path, filename):
    """save pickeled object"""
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(path, filename):
    """load pickeled object"""
    file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
