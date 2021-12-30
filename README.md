# Bioacoustic Detection

(A pre-trained model and API for detecting humpback whale vocalizations in
hydrophone recordings.)

Detecting and localizing biological sounds from hydrophone recordings is useful to remotely study the social behavior of marine mammals. Manually listening through hydrophone recordings to label the occurence of each vocalization is an incredibly time consuming process and is not feasible for annotating hundreds to thousands of hours of audio data. In order to reduce the amount of time that researchers must spend simply locating the signals which they wish to study, this repository aims to automatically detect and localize vocalizations in hydrophone recordings. To that end, this repository trains a convolutional neural network to detect humpback whale vocalizations in spectrograms (a visual representation of audio). Using CNNs to detect objects in images is a well-studied problem and therefor by using spectrograms as input to our model, we can benefit from the numerous successful approaches published in the object detection literature as well as open-source libraries which support this task.

First, we'll review object detection. The [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MSCOCO](https://arxiv.org/abs/1405.0312) datasets are both cannonical examples of the object detection task: given an image as input, produce a set of bounding boxes and class labels which identify and localize each object in the image. In the context of our task: given a spectrogram of an audio recording, produce a set of bounding boxes and class labels which identify the species of each vocalization and localize each in both time and frequency.

(TODO: include example spectrogram and labels)

(TODO: more specifically, which CNN architecture does this repo currently use and how is it trained?)

(TODO: Provide an example workflow from input files to detections and performance evaluations.)

## Organization

The bioacoustic-detection package is organized with all of the source code in `src/bioacoustic_detection`, executable scripts in `scripts/`, and tests in `tests/`. Nearly all of the interesting code will be contained in the package's source code while the executable scripts are only meant to provide a convenient command line interface (CLI) for using the package. The bioacoustic-detection package itself is composed of:

  - `utils/`
      - `io.py` contains code for reading and writing annotation and wav files.
      - `annot.py` contains code for cleaning raw annotations and other annotation-related utilities
      - `wav.py` contains code for pre-processing wav files, including decimation
      - `vis.py` contains code for constructing and saving visualizations
  - `spectrogram_dataset_generator.py` contains the code for generating spectrogram datasets from annotations and recordings.
  - (TODO) `run_inference.py` contains the code for running inference on a recording.
  - (TODO) `evaluate.py` contains code for evaluating a trained model on a spectrogram dataset with annotations.

## Install
  1. Install python3 and python3-venv
  2. Clone the repo with `git clone --recursive git@github.com:jackson-waschura/bioacoustic-detection.git` (the `--recursive` flag is important so that you also retrieve the TF `models` submodule which is used to train object detection models)
  3. Change the current directory with `cd bioacoustic-detection`.
  4. Run the setup script with `./setup.sh`. This will create a python3 virtual environment (venv) in the directory "env/" and install the prerequisite packages there along with the bioacoustic-detection package.
  5. Run `source env/bin/activate` to enter the python venv. When you are done using the bioacoustic-detection package, you can exit the python venv by running `deactivate`. (You must do this every time you want to use the bioacoustic-detection package unless you prefer to install the package directly onto your local machine.)

## Scripts

The following scripts provide helpful interfaces for using the bioacoustic-detection package from the terminal:
  - `clean_annotations.py [-h] [-i IN_DIR] [-o OUT_DIR] [-q]`
  - `generate_spectrogram_dataset.py [-h] --splits SPLITS -o OUT_DIR [optional parameters...]`

## TODO:
  1. `DONE` Add a very permissive license.
  2. `STARTED` Translate Jupyter Notebooks into python scripts
      - `DONE` IO utility methods
      - `DONE` Annotation utility methods
      - `DONE` Data cleaning methods
      - `STARTED` Visualization utility methods
      - `STARTED` Spectrogram Dataset Generation
      - Training methods (adapted from TF's Object Detection)
      - Inference methods
      - Evaluation methods
  3. Create unit tests for python scripts
      - IO utility methods
      - `DONE` Annotation utility methods
      - Wav utility methods (PCEN)
      - Visualization utility methods
      - `DONE` Data cleaning methods
      - Spectrogram Dataset Generation
      - Inference methods
      - Evaluation methods
  4. `DONE` List all dependencies with specific versions in a requirements.txt file
  5. `DONE` Allow this codebase to be installed with pip (setuptools).
  6. `DONE` Create a setup script which creates and sets up a python virtual environment (installing requirements and then this package).
  7. Write notebooks in colab which others can use to execute tasks such as running inference or evaluation. (Pulls this source from github, installs all requirements, installs package, calls necessary functions to perform task, returns / visualizes results).
  8. `STARTED` Add detailed documentation in this README as well as within each of the notebooks which use it. (Look into expandable items in markdown)
  9. `DONE` Add a .gitignore to ignore all of the `__pycache__` files etc.
