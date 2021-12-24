#!/bin/bash

echo "-------------------------"
echo "    Running setup.sh     "
echo "-------------------------"

mkdir env
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel

sudo apt-get install -y protobuf-compiler
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error installing protobuf-compiler (protoc)"
    echo "Terminating early."
    exit $retVal
fi

# ============= TF Object Detection API =============
pushd models/research

# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

popd

# ============= bioacoustic-detection =============
python -m pip install -r requirements.txt
python -m pip install -e .

echo "-------------------------"
echo "Finished running setup.sh"
echo "-------------------------"