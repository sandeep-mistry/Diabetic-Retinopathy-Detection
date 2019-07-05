# !/bin/bash

echo Running environment setup
echo Installing Python
conda install --yes python=3.6 conda pip
echo Installing packages
pip install -r requirements.txt
echo Environment setup complete
