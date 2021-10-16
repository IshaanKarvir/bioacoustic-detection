import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
import warnings

def read_wavfile(fpath, normalize=True, dtype=np.float32, verbose=False):
    """
    Reads a.wav file from a path then returns the file's sample rate and data.
    
    If normalize is set to True, then the data is first scaled to [-1.0, 1.0]
    before returning.

    Parameters
    fpath : str
        path to the .wav file (eg: '/foo/bar/example.wav')
    normalize : bool, optional (default: True)
        flag to control whether data is normalized
    dtype : numpy.dtype, optional (default: np.float32)
        type to cast .wav file's data (should be a float type)
    verbose : bool, optional (default: False)
        flag to control whether debug information is printed
    
    Returns
    sr : int
        number of samples taken per second
    data : numpy array
        NDarray containing .wav file's data
    """

    if verbose:
        print("Reading {}".format(fpath))
    sr, data = wavfile.read(fpath)
    if verbose:
        print(
            "{} samples at {} samples/sec --> {} seconds".format(
                data.shape[0],
                sr,
                data.shape[0]/sr
            )
        )

    data = data.astype(dtype)

    if normalize:
        data = data - data.min()
        data = data / data.max() * 2.0
        data = data - 1.0
    
    return sr, data


def read_annotations(fpath, verbose=False):
    """
    Reads an annotation file and returns the annotations as a pandas DataFrame.

    This function does not ensure columns are returned in any particular order
    or having any particular types.

    Parameters
    fpath : str
        path to the .txt file (eg: '/foo/bar/annots.txt')
    verbose : bool, optional (default: False)
        flag to control whether debug information is printed
    
    Returns
    annotations : pandas DataFrame
        DataFrame containing one annotation per row
    """
    annotations = pd.read_csv(fpath, sep="\t")
    if verbose:
        print("Read {} annotations from {}".format(len(annotations), fpath))
        print(
            "Columns:",
            ",".join(
                [" {} ({})".format(c, type(c)) for c in annotations.columns]
            )
        )
    return annotations


def save_annotations(annots, fpath, verbose=False):
    """
    Saves a DataFrame of annotations to a file.

    Parameters
    annots : DataFrame
        annotations to be saved
    fpath : str
        path to save annotations to (eg: '/foo/bar/annots.txt')
    verbose : bool, optional (default: False)
        flag to control whether debug information is printed
    """
    if not fpath.endswith(".txt"):
        warnings.warn(
            "({}) Saving as '{}' does not match existing " \
                "annotations.".format("save_annotations", fpath)
        )


    annots.to_csv(fpath, index=False, sep="\t", float_format='%g')
    if verbose:
        print("Saved {} annotations to {}".format(len(annots), fpath))
