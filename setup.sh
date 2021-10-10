#!/bin/bash

echo "-------------------------"
echo "    Running setup.sh     "
echo "-------------------------"

mkdir env
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .

echo "-------------------------"
echo "Finished running setup.sh"
echo "-------------------------"

