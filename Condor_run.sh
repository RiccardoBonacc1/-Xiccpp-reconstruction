#!/bin/bash

## This is a simple bash script to run the python script in the correct conda environment  

NUMBER=$1
POLARITY=$2

set -- 

# Activate conda environment
source /home/bonacci/Work/miniconda3/bin/activate base
conda activate Env

# Run the Python script in the activated conda environment
python /home/bonacci/Work/Mass_Fit_script.py -n $NUMBER -p $POLARITY

echo "Done"
