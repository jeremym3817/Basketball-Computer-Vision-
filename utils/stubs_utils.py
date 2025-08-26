import os
import pickle

def save_stub(stub_path, object):
    # create stub directory
    stub_dirname = os.path.dirname(stub_path)
    if not os.path.exists(stub_dirname):
        os.mkdir(stub_dirname)
    
    # pickles object (stores it for later, serialize)
    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(object, f)

def read_stub(stub_path, read_from_stub=True):
    # if we're reading from stub path and it exists, unpickle (deserialize) and return object
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    return None
