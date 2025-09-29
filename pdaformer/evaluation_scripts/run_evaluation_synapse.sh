#!/bin/sh

DATASET_PATH=../DATASET_Synapse
CHECKPOINT_PATH=../output_synapse

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../pdaformer/run/run_training.py 3d_fullres unetr_pp_trainer_synapse 2 0 -val
