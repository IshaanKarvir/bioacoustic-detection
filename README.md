# Bioacoustic Detection

A pre-trained model and API for detecting humpback whale vocalizations in
hydrophone recordings.

## Install
TODO: Describe install instructions here.
  1. Install python3 and python3-venv
  2. Clone the repo and `cd bioacoustic_detection`
  3. Run `./setup.sh` (Describe what this does)
  4. Run `source env/bin/activate`

## TODO:
  1. `DONE` Add a very permissive license.
  2. `STARTED` Translate Jupyter Notebooks into python scripts
      - `DONE` IO utility methods
      - `STARTED` Annotation utility methods
      - Visualization utility methods
      - Data cleaning methods
      - Spectrogram Dataset Generation
      - Inference methods
      - Evaluation methods
  3. Create unit tests for python scripts
      - IO utility methods
      - `STARTED` Annotation utility methods
      - Visualization utility methods
      - Data cleaning methods
      - Spectrogram Dataset Generation
      - Inference methods
      - Evaluation methods
  4. `DONE` List all dependencies with specific versions in a requirements.txt file
  5. `DONE` Allow this codebase to be installed with pip (setuptools).
  6. `DONE` Create a setup script which creates and sets up a python virtual environment (installing requirements and then this package).
  7. Write notebooks in colab which others can use to execute tasks such as running inference or evaluation. (Pulls this source from github, installs all requirements, installs package, calls necessary functions to perform task, returns / visualizes results).
  8. Add detailed documentation in this README as well as within each of the notebooks which use it.