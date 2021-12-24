# Bioacoustic Detection

A pre-trained model and API for detecting humpback whale vocalizations in
hydrophone recordings.

(TODO: Add high-level description of the task and how this codebase accomplishes it. Link to relevant papers in object detection / nueral networks. Provide an example workflow from input files to detections and performance evaluations.)

## Organization

The bioacoustic-detection package is organized with all of the source code in `src/bioacoustic_detection`, executable scripts in `scripts/`, and tests in `tests/`. Nearly all of the interesting code will be contained in the package's source code while the executable scripts are only meant to provide a convenient command line interface (CLI) for using the package. The bioacoustic-detection package itself is composed of:

  - `utils/`
      - `io.py` contains code for reading and writing annotation and wav files.
      - `annot.py` contains code for cleaning raw annotations and other annotation-related utilities
      - `wav.py` contains code for pre-processing wav files, including decimation
      - `vis.py` contains code for constructing and saving visualizations
  - `spectrogram_dataset_generator.py` contains the code for generating spectrogram datasets from annotations and recordings.
  - (TODO) `train.py` contains the code for training a detection model from a spectrogram dataset.
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
  - (TODO) `generate_dataset.py ...`

## TODO:
  1. `DONE` Add a very permissive license.
  2. `STARTED` Translate Jupyter Notebooks into python scripts
      - `DONE` IO utility methods
      - `DONE` Annotation utility methods
      - `DONE` Data cleaning methods
      - `STARTED` Visualization utility methods
      - Spectrogram Dataset Generation
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
