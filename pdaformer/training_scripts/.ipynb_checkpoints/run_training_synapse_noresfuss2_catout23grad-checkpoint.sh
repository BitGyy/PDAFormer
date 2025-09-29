#!/bin/sh

DATASET_PATH=../DATASET_Synapse

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_synapse_noresfuss2_catout123
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python3 ../unetr_pp/run/run_training_gradcam.py 3d_fullres unetr_pp_trainer_synapse 2 0 
