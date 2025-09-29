#!/bin/sh

DATASET_PATH=../DATASET_CTA
CHECKPOINT_PATH=../output_CTA/

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task501_CTA
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../pdaformer/run/run_training.py 3d_fullres unetr_pp_trainer_CTA 501 0 -val
