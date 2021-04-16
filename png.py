from glob import glob
import numpy as np
from sklearn.externals._pilutil import imread


def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        print('Processing file: ', f)
        features[f] = imread(f, True).astype(np.float64)
    return features
