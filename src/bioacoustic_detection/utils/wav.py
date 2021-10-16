from scipy.io import wavfile
from scipy import signal
import warnings
import numpy as np
import os
from os import path
from tqdm import tqdm

def process_wav_file(fpath, rate=12, norm=0.5):
    """
    Decimates and normalizes a wav file to the range [-norm, norm].

    Parameters
    fpath : str
        path to the .wav file to process (eg: '/foo/bar/recording.wav')
    rate : int, optional (default: 12)
        decimation rate (by default reduces samples by a factor of 12)
    norm : float, optional (default: 0.5)
        absolute value of the minimum and maximum sample
    
    Returns
    sr : int
        new sample rate after decimation
    data : np.ndarray
        array of processed data
    """

    sr, data = wavfile.read(fpath)
    data = signal.decimate(data, rate)
    data = data.astype(float)
    data = data - data.min()
    data = (data / data.max() * 2.0) - 1.0
    data = data * norm
    sr = sr // rate
        
    return sr, data


def process_directory_wav_files(
        input_directory,
        output_directory,
        rate=12,
        norm=0.5,
        dtype=np.int16,
        show_progress=True):
    """
    Decimates and normalizes every wav file in a directory then saves to an
    output directory.

    Parameters
    input_directory : str
        path to the input directory containing .wav files
    output_directory : str
        path to the output directory to save processed .wav files
    rate : int, optional (default: 12)
        decimation rate (by default reduces samples by a factor of 12)
    norm : float, optional (default: 0.5)
        absolute value of the minimum and maximum sample
    dtype : integer data type, optional (default: np.int16)
        integer data type to convert wav samples to
    show_progress : bool, optional (default: True)
        flag to control whether progress bar is shown or hidden
    """
    
    os.makedirs(output_directory, exist_ok=True)

    if norm < 0.0 or norm > 1.0:
        new_norm = np.clip(norm, 0.0, 1.0)
        warnings.warn(
            "({}) Norm must be between 0.0 and 1.0, not {:g}. " \
            "Clipping to {:g}.".format(
                "process_directory_wav_files",
                norm,
                new_norm)
        )
        norm = new_norm

    fnames = [
        fname for fname in os.listdir(input_directory) if fname.endswith(".wav")
    ]
    file_iter = tqdm(fnames) if show_progress else fnames
    for fname in file_iter:
        fpath = path.join(input_directory, fname)
        sr, data = process_wav_file(fpath, rate=rate, norm=norm)
        data = (data * np.iinfo(dtype).max).astype(dtype)
        # Data now spans half of the dtype's span and is 0-centered.
        out_fname = "{}_processed.wav".format(path.splitext(fname)[0])
        wavfile.write(path.join(output_directory, out_fname), sr, data)


# TODO: Implement some unit tests for PCEN
def PCEN(spec, M_return_timestep, init_M=None, epsilon=1e-6, s=0.001, alpha=0.80, delta=2.0, r=0.5):
    """
    Per-channel energy normalization.
    
    Removes background noise by keeping track
    of a smoothed intensity in each frequency bin (see [1] and [2] for more
    information). Default values were handcrafted and should be examined
    more thoroughly.

    Since this function needs to be applied to (potentially overlapping) chunks
    of a very long signal, it has been adapted to also return the smoothed
    intensity at a requested time step.

    Parameters
    spec : numpy array
        the input spectrogram
    M_return_timestep : int
        step in time to return the smoothed intensity
    init_M : numpy array, optional (default: None)
        initial state of the smoothed intensity
    epsilon : float, optional (default: 1e-6)
        Very small number used to prevent division by zero
    s : float, optional (default: 0.001)
        IIR smoothing coefficient
    alpha : float, optional (default: 0.80)
        gain normalization parameter [0.0, 1.0]
    delta : float, optional (default: 2.0)
        stabilized root compression offset
    r : float, optional (default: 0.5)
        stabilized root compression exponent
    
    Returns
    output : numpy array
        spectrogram result of applying PCEN to input spec
    out_M : ...
        smoothed intensity of input spec at timestep 'M_return_timestep'

    [1] - https://research.google/pubs/pub45911.pdf
    [2] - https://www.justinsalamon.com/uploads/4/3/9/4/4394963/lostanlen_pcen_spl2018.pdf
    """
    
    assert alpha > 0.0 and alpha < 1.0, "alpha must be between 0.0 and 1.0"
    
    output = np.zeros_like(spec)
    if init_M is None:
        M = np.zeros(shape=(output.shape[0]))
    else:
        M = np.array(init_M)
    assert M.shape[0] == output.shape[0]
    for t in range(output.shape[1]):
        M = (1 - s) * M + s * spec[:,t]
        output[:,t] = ((spec[:,t] / ((M + epsilon) ** alpha)) ** r) - (delta ** r)
        if t == M_return_timestep:
            out_M = M
    return output, out_M